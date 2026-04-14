"""
preprocess_data.py — Batch-process all .vital files → one HDF5 file.

Performance architecture (24-core / 205 GB RAM / RTX 5090 machine):
  ┌─────────────────────────────────────────────────────────────────┐
  │  ProcessPoolExecutor (n_workers ≈ 16)                           │
  │    Each worker: one .vital file                                 │
  │      1. VitalDB load + MNE FIR filter  (~0.8 s)                │
  │      2. Collect valid windows → (N, n_ch, T) array             │
  │      3. BatchProcessor.compute()   → vectorized numpy FFT       │
  │         replaces N × scipy.signal.welch calls (~0.5 s vs 30 s) │
  │      4. Write → temp HDF5 file in outputs/preprocessed/tmp/     │
  │  Main process: merge all temp HDF5 → final dataset.h5          │
  │  PipelineValidator: QA on merged file                          │
  └─────────────────────────────────────────────────────────────────┘

Estimated runtime:
  Sequential (old): 75 files × ~33 s = ~41 min
  Parallel   (new): ceil(75/16) × ~2 s = ~10 s  (wall-clock)

Usage:
    # Standard EEG preprocessing (v1–v8)
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --config configs/pipeline_v1.yaml
                                       --raw_dir raw_data
                                       --out outputs/preprocessed/dataset.h5
                                       --workers 16

    # v9 MERIDIAN: EEG preprocessing + multimodal annotation (drug/vitals/stim)
    python scripts/preprocess_data.py --config configs/pipeline_v9.yaml --v3
    python scripts/preprocess_data.py --config configs/pipeline_v9.yaml --v3
                                       --out outputs/preprocessed/dataset.h5
                                       --v3-h5 outputs/preprocessed/dataset_v3.h5

    # Only run multimodal annotation on an existing HDF5 (skip EEG reprocessing)
    python scripts/preprocess_data.py --config configs/pipeline_v9.yaml
                                       --v3-only
                                       --out outputs/preprocessed/dataset.h5
                                       --v3-h5 outputs/preprocessed/dataset_v3.h5
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import yaml
from tqdm import tqdm

from src.data.loader import VitalLoader, load_config
from src.pipeline.validator import PipelineValidator
from src.data.dataset_v3 import build_multimodal_hdf5


# ──────────────────────────────────────────────────────────────────────────────
# Worker function (runs in a subprocess)
# ──────────────────────────────────────────────────────────────────────────────

def _worker(args: Tuple[str, str, Dict[str, Any], str]) -> Tuple[str, int, str]:
    """
    Process one .vital file and write result to a temp HDF5.

    Returns (case_id, n_windows, tmp_h5_path_or_error_message).
    On failure returns (case_id, 0, "ERROR: ...").
    """
    vital_path, case_id, cfg, tmp_dir = args
    tmp_h5 = os.path.join(tmp_dir, f"{case_id}.h5")
    try:
        loader = VitalLoader(cfg)
        with h5py.File(tmp_h5, "w") as hf:
            n = loader.process_file(vital_path, hf, case_id)
        return case_id, n, tmp_h5
    except Exception as e:
        # Clean up partial file
        if os.path.exists(tmp_h5):
            try:
                os.remove(tmp_h5)
            except OSError:
                pass
        return case_id, 0, f"ERROR: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 merge
# ──────────────────────────────────────────────────────────────────────────────

def _merge_h5_files(
    tmp_files: list,   # list of (case_id, tmp_h5_path)
    existing_h5: h5py.File,
    overwrite: bool,
) -> int:
    """
    Copy each case group from its temp HDF5 into the final output HDF5.
    Returns total number of windows written.
    """
    total = 0
    for case_id, tmp_h5 in tqdm(tmp_files, desc="Merging HDF5", leave=False):
        if case_id in existing_h5:
            if overwrite:
                del existing_h5[case_id]
            else:
                total += int(existing_h5[case_id].attrs.get("n_windows", 0))
                continue
        with h5py.File(tmp_h5, "r") as src:
            if case_id not in src:
                continue
            src.copy(case_id, existing_h5)
            total += int(existing_h5[case_id].attrs.get("n_windows", 0))
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel EEG preprocessing: .vital → HDF5"
    )
    parser.add_argument("--config",   default="configs/pipeline_v1.yaml")
    parser.add_argument("--raw_dir",  default="raw_data")
    parser.add_argument("--out",      default="outputs/preprocessed/dataset.h5")
    parser.add_argument("--workers",  type=int, default=None,
                        help="Number of parallel workers. Default: min(16, cpu_count-2)")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Run dataset validation after preprocessing")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Reprocess cases already in the output HDF5")
    # v3 MERIDIAN multimodal annotation
    parser.add_argument("--v3", action="store_true", default=False,
                        help="After EEG preprocessing, add multimodal annotations "
                             "(drug CE / vitals / stim_cv) to a v3 HDF5 copy")
    parser.add_argument("--v3-h5", default=None,
                        help="Output path for the multimodal HDF5. Defaults to "
                             "paths.multimodal_h5 in config, or dataset_v3.h5")
    parser.add_argument("--v3-only", action="store_true", default=False,
                        help="Skip EEG preprocessing; only run multimodal annotation "
                             "on an existing HDF5 (--out must already exist)")
    args = parser.parse_args()

    # Auto-enable v3 when using a v3 config
    if not args.v3 and not args.v3_only:
        import yaml as _yaml_probe
        with open(args.config, "r", encoding="utf-8") as _f:
            _cfg_probe = _yaml_probe.safe_load(_f)
        if _cfg_probe.get("training", {}).get("model_version") == "v3":
            args.v3 = True
            print("[INFO] Detected model_version=v3 in config — enabling --v3 automatically")

    cfg = load_config(args.config)
    raw_dir  = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --v3-only: skip EEG preprocessing, jump straight to multimodal annotation
    if args.v3_only:
        if not out_path.exists():
            print(f"ERROR: --v3-only requires an existing HDF5 at {out_path}")
            sys.exit(1)
        print(f"--v3-only mode: skipping EEG preprocessing, "
              f"annotating existing HDF5 at {out_path}")
        _run_v3_annotation(args, cfg, out_path)
        return

    vital_files = sorted(raw_dir.glob("*.vital"))
    print(f"Found {len(vital_files)} .vital files in {raw_dir}")
    print(f"Output: {out_path}")

    # Filter already-done cases (unless --overwrite)
    existing_cases: set = set()
    if out_path.exists() and not args.overwrite:
        with h5py.File(str(out_path), "r") as hf:
            existing_cases = set(hf.keys())
        if existing_cases:
            print(f"Skipping {len(existing_cases)} already-processed cases "
                  f"(use --overwrite to reprocess)")

    todo = [vf for vf in vital_files if vf.stem not in existing_cases]
    skip = [vf for vf in vital_files if vf.stem in existing_cases]
    print(f"  To process: {len(todo)}   Already done: {len(skip)}")

    if not todo:
        print("Nothing to process.")
        _print_summary(str(out_path))
        return

    # Determine worker count
    cpu = os.cpu_count() or 4
    n_workers = args.workers if args.workers else min(16, max(1, cpu - 2))
    print(f"Using {n_workers} parallel workers (cpu_count={cpu})")

    # Build worker argument list
    tmp_dir = out_path.parent / "tmp_workers"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (str(vf), vf.stem, cfg, str(tmp_dir))
        for vf in todo
    ]

    # ── Parallel processing ───────────────────────────────────────────────
    t0 = time.time()
    results: list = []   # (case_id, n_windows, tmp_h5_or_error)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, wa): wa[1] for wa in worker_args}
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for fut in as_completed(futures):
                case_id, n, path_or_err = fut.result()
                if path_or_err.startswith("ERROR:"):
                    tqdm.write(f"  {case_id}: {path_or_err}")
                else:
                    tqdm.write(f"  {case_id}: {n} windows")
                results.append((case_id, n, path_or_err))
                pbar.update(1)

    proc_elapsed = time.time() - t0
    ok_results  = [(cid, n, p) for cid, n, p in results if not p.startswith("ERROR:") and n > 0]
    err_results = [(cid, n, p) for cid, n, p in results if p.startswith("ERROR:") or n == 0]

    print(f"\nProcessing: {len(ok_results)} ok, {len(err_results)} failed/empty "
          f"— {proc_elapsed:.1f}s ({proc_elapsed/60:.1f} min)")

    # ── Merge temp files into final HDF5 ─────────────────────────────────
    t1 = time.time()
    merge_list = [(cid, path) for cid, _, path in ok_results]

    with h5py.File(str(out_path), "a") as hf:
        total_windows = _merge_h5_files(merge_list, hf, args.overwrite)
        # Add previously-done windows
        for vf in skip:
            cid = vf.stem
            if cid in hf:
                total_windows += int(hf[cid].attrs.get("n_windows", 0))

    merge_elapsed = time.time() - t1

    # ── Clean up temp directory ───────────────────────────────────────────
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    total_elapsed = time.time() - t0
    print(f"Merge: {merge_elapsed:.1f}s")
    print(f"\nPreprocessing complete.")
    print(f"  Total windows : {total_windows:,}")
    print(f"  Wall time     : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    _print_summary(str(out_path))

    # ── Dataset validation ────────────────────────────────────────────────
    if args.validate and total_windows > 0:
        print("\nRunning dataset validation...")
        validator = PipelineValidator.from_config(cfg, hard_fail=False)
        validator.validate_dataset(str(out_path), n_sample_per_case=3, verbose=True)

    # ── V3 multimodal annotation ──────────────────────────────────────────
    if args.v3 or args.v3_only:
        _run_v3_annotation(args, cfg, out_path)


def _run_v3_annotation(args, cfg: dict, base_h5: Path) -> None:
    """
    构建 MERIDIAN-v9 多模态 HDF5 注释。

    流程：
      1. 确定 v3 HDF5 路径（参数 > 配置 > 默认）
      2. 如果 v3 HDF5 不存在或与 base_h5 不同，复制 base_h5 → v3 HDF5
      3. 调用 build_multimodal_hdf5() 添加药物/心血管刺激/生命体征标注
    """
    import shutil

    # 确定 v3 HDF5 输出路径 (argparse converts --v3-h5 → args.v3_h5)
    v3_h5_str = getattr(args, "v3_h5", None)
    if not v3_h5_str:
        v3_h5_str = cfg.get("paths", {}).get(
            "multimodal_h5",
            str(base_h5.parent / "dataset_v3.h5"),
        )
    v3_h5 = Path(v3_h5_str)
    v3_h5.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = getattr(args, "raw_dir", "raw_data")

    # 复制 base_h5 → v3_h5（幂等：如果 v3_h5 已存在则跳过复制）
    if str(v3_h5.resolve()) != str(base_h5.resolve()):
        if not v3_h5.exists():
            print(f"\nCopying {base_h5} → {v3_h5} ...")
            t_copy = time.time()
            shutil.copy2(str(base_h5), str(v3_h5))
            print(f"Copy done in {time.time()-t_copy:.1f}s")
        else:
            print(f"\nV3 HDF5 already exists at {v3_h5} (skipping copy)")

    # 添加多模态注释（build_multimodal_hdf5 是幂等的）
    print(f"\nRunning multimodal annotation on {v3_h5}")
    t_mm = time.time()
    build_multimodal_hdf5(str(v3_h5), raw_dir, verbose=True)
    print(f"Multimodal annotation done in {time.time()-t_mm:.1f}s ({(time.time()-t_mm)/60:.1f} min)")
    _print_summary(str(v3_h5))


def _print_summary(out_path: str) -> None:
    if not Path(out_path).exists():
        return
    with h5py.File(out_path, "r") as hf:
        cases  = list(hf.keys())
        total  = sum(int(hf[c].attrs.get("n_windows", 0)) for c in cases)
        print(f"\nHDF5 summary:")
        print(f"  Cases  : {len(cases)}")
        print(f"  Windows: {total:,}")
        if cases:
            c0 = cases[0]
            print(f"  Shapes ({c0}):")
            for k in hf[c0]:
                print(f"    {k}: {hf[c0][k].shape}  dtype={hf[c0][k].dtype}")


if __name__ == "__main__":
    # Required on Windows for ProcessPoolExecutor
    import multiprocessing
    multiprocessing.freeze_support()
    main()
