"""eegrec_to_ads1299_replay.py

Convert an EEGRecorder raw session (eeg.bin / ppg.bin / meta.json) into a fully-playable
ADS1299 session for the EEGMonitor.Ads1299 数据回放 dialog.

It re-runs the raw EEG through the SAME Python inference service the live app uses
(/reset + /process, 250-sample chunks @ sample_rate=250 — identical to a live session),
attaches PPG HR/SpO2 per chunk, and writes:

    {out}/{name}_replay/
        inference.jsonl       per-epoch qcon(bis)/qnox(fnox)/sqi/hr/spo2/bands  (what the trend reads)
        raw_signal.bin        tagged LE: [i8 ticks][u8 1][f32 uv]  (+ optional PPG [u8 3][i4 ir][i4 red])
        raw_signal.meta.json  eeg_sample_rate_hz
        events.jsonl          (empty)

Usage:
    python scripts/eegrec_to_ads1299_replay.py <eegrec_session_folder> [--out <ads1299_recordings_dir>]
                                               [--service http://localhost:8765]

The inference service must be reachable; if it is not, this script tries to start it
(python EEGMonitor/EEGProcessingService/main.py) and waits for /health.
"""
import sys, os, json, time, struct, argparse, subprocess
import urllib.request, urllib.error
from datetime import datetime, timedelta
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SERVICE_DIR = os.path.join(REPO, "EEGMonitor", "EEGProcessingService")
ADS_APPSETTINGS = os.path.join(REPO, "EEGMonitor", "EEGMonitor.Ads1299", "appsettings.json")
DEFAULT_OUT = os.path.join(REPO, "recordings")

PPG_DTYPE = np.dtype([("ticks", "<i8"), ("ir", "<i4"), ("red", "<i4"), ("spo2", "u1"), ("hr", "u1")])
TICKS_PER_DAY = 864_000_000_000
CHUNK = 250          # samples/chunk — matches DataPipeline ChunkSize for nominal 250 Hz
REQ_RATE = 250       # sample_rate sent to the service — matches the live app


def dotnet_ticks(dt: datetime) -> int:
    """Wall-clock .NET DateTime.Ticks (100 ns since 0001-01-01), exact integer math."""
    d = dt - datetime(1, 1, 1)
    return d.days * TICKS_PER_DAY + d.seconds * 10_000_000 + d.microseconds * 10


def parse_started(s: str) -> datetime:
    # meta 'started' looks like 2026-06-11T21:07:57.5956539+08:00 — keep the local wall clock, drop tz.
    s = s.strip()
    for tz in ("+", ):
        i = s.rfind("+")
        if i > 10:
            s = s[:i]
    if s.endswith("Z"):
        s = s[:-1]
    # trim fractional seconds to 6 digits for fromisoformat
    if "." in s:
        head, frac = s.split(".", 1)
        s = head + "." + frac[:6]
    return datetime.fromisoformat(s)


# ── HTTP helpers (stdlib only) ──
def http_get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def http_post_json(url, obj, timeout=60):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ensure_service(base):
    try:
        st, _ = http_get(base + "/health")
        if st == 200:
            print(f"  推理服务在线: {base}")
            return None
    except Exception:
        pass
    print("  推理服务未在线 — 尝试启动 main.py …")
    py = sys.executable
    proc = subprocess.Popen([py, "main.py"], cwd=SERVICE_DIR,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        time.sleep(1)
        try:
            st, _ = http_get(base + "/health")
            if st == 200:
                print("  推理服务已启动")
                return proc
        except Exception:
            continue
    raise SystemExit("无法连接/启动推理服务,请先手动运行 EEGProcessingService/main.py")


def default_out():
    try:
        with open(ADS_APPSETTINGS, encoding="utf-8-sig") as f:
            return json.load(f)["Recording"]["OutputDirectory"]
    except Exception:
        return DEFAULT_OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="EEGRecorder 会话文件夹 (含 eeg.bin/ppg.bin/meta.json)")
    ap.add_argument("--out", default=None, help="ADS1299 回放根目录 (默认读 ads1299 appsettings)")
    ap.add_argument("--service", default="http://localhost:8765")
    args = ap.parse_args()

    src = os.path.abspath(args.session)
    with open(os.path.join(src, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    eeg = np.fromfile(os.path.join(src, "eeg.bin"), dtype="<f4")
    ppg_path = os.path.join(src, "ppg.bin")
    ppg = np.fromfile(ppg_path, dtype=PPG_DTYPE) if os.path.exists(ppg_path) and os.path.getsize(ppg_path) else np.empty(0, PPG_DTYPE)

    fs = float(meta["eeg"].get("measured_rate_hz") or meta["eeg"]["nominal_rate_hz"])
    start = parse_started(meta["started"])
    base_ticks = dotnet_ticks(start)
    n_chunks = eeg.size // CHUNK
    if n_chunks == 0:
        raise SystemExit(f"EEG 太短 ({eeg.size} 采样),不足一个 {CHUNK} 块")

    out_root = args.out or default_out()
    name = os.path.basename(src.rstrip("\\/")) + "_replay"
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"源: {src}")
    print(f"EEG {eeg.size} 采样 @ {fs:.2f}Hz → {n_chunks} 块;PPG {ppg.size} 帧")
    print(f"输出: {out_dir}")

    base = args.service.rstrip("/")
    ensure_service(base)
    try:
        http_post_json(base + "/reset", {}, timeout=10)
    except Exception:
        pass

    # ── replay chunks through /process ──
    inf_lines = []
    for j in range(n_chunks):
        seg = eeg[j * CHUNK:(j + 1) * CHUNK]
        lo_tick = base_ticks + round(j * CHUNK / fs * 1e7)
        hi_tick = base_ticks + round((j + 1) * CHUNK / fs * 1e7)
        t_chunk = start + timedelta(seconds=j * CHUNK / fs)

        hr_val = spo2_val = None
        if ppg.size:
            m = (ppg["ticks"] >= lo_tick) & (ppg["ticks"] < hi_tick)
            hrw = ppg["hr"][m]; hrw = hrw[hrw > 0]
            spw = ppg["spo2"][m]; spw = spw[spw > 0]
            if hrw.size:  hr_val = float(np.median(hrw))
            if spw.size:  spo2_val = float(np.median(spw))

        req = {
            "sample_rate": REQ_RATE,
            "channel_count": 1,
            "start_time": t_chunk.isoformat(),
            "eeg_data": [[float(v)] for v in seg],
            "pulse_wave": [],
            "spo2": spo2_val,
            "heart_rate": hr_val,
        }
        try:
            r = http_post_json(base + "/process", req, timeout=60)
        except Exception as e:
            print(f"  块 {j} 推理失败: {e}")
            continue

        inf_lines.append(json.dumps({
            "t": t_chunk.isoformat(),
            "qcon": r.get("bis"),
            "qnox": r.get("fnox"),
            "sqi": round(r.get("sqi", 0.0), 1),
            "se": r.get("se"),
            "re": r.get("re"),
            "hr": r.get("heart_rate"),
            "hrv": r.get("hrv_rmssd"),
            "spo2": r.get("spo2"),
            "bands": {
                "DeltaPower": r.get("delta_power", 0.0),
                "ThetaPower": r.get("theta_power", 0.0),
                "AlphaPower": r.get("alpha_power", 0.0),
                "BetaPower":  r.get("beta_power", 0.0),
                "GammaPower": r.get("gamma_power", 0.0),
            },
            "amp_uv": round(r.get("eeg_amplitude_uv", 0.0), 1),
            "dom_hz": round(r.get("eeg_dominant_hz", 0.0), 1),
            "saturated": r.get("hw_is_saturated", False),
            "valid": r.get("signal_valid", True),
            "electrode": r.get("electrode_status", "ok"),
        }))
        if (j + 1) % 5 == 0 or j == n_chunks - 1:
            print(f"  推理 {j + 1}/{n_chunks}  qCON={r.get('bis')}  SQI={r.get('sqi'):.0f}")

    with open(os.path.join(out_dir, "inference.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(inf_lines) + ("\n" if inf_lines else ""))

    # ── raw_signal.bin (tagged): EEG tag1 per sample, then PPG tag3 ──
    with open(os.path.join(out_dir, "raw_signal.bin"), "wb") as f:
        for i in range(eeg.size):
            f.write(struct.pack("<qBf", base_ticks + round(i / fs * 1e7), 1, float(eeg[i])))
        for fr in ppg:
            f.write(struct.pack("<qBii", int(fr["ticks"]), 3, int(fr["ir"]), int(fr["red"])))

    with open(os.path.join(out_dir, "raw_signal.meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "format": "tagged-le-binary",
            "record": "[int64 ticks][byte tag]; tag1 EEG=[float32 uv]; tag3 PPG=[int32 ir][int32 red]",
            "eeg_sample_rate_hz": round(fs),
            "started": start.isoformat(),
            "source": src,
        }, f, indent=2)

    open(os.path.join(out_dir, "events.jsonl"), "w").close()

    print(f"\n完成  {len(inf_lines)} 个推理帧 + {eeg.size} 原始EEG 点")
    print(f"在 ADS1299 程序里点「数据回放」→ 选择会话 “{name}” → 加载")


if __name__ == "__main__":
    main()
