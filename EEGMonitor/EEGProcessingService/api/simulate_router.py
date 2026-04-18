"""
simulate_router.py — VitalDB .vital file simulation endpoints.

Flow:
  POST /simulate/info   → list track names + metadata (fast, no data load)
  POST /simulate/load   → resample all tracks into memory, return session_id
  POST /simulate/chunk  → return [start, start+count) samples from session
  DELETE /simulate/{id} → free session memory
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/simulate", tags=["simulate"])

# ── In-memory session store ───────────────────────────────────────────────────
# Key: session_id (str)
# Value: {eeg: ndarray (N,2), pulse: ndarray (N,), spo2: ndarray (N,),
#         hr: ndarray (N,), n_samples: int, sample_rate: int, duration: float}
_sessions: dict[str, dict] = {}

# ── Schemas ───────────────────────────────────────────────────────────────────

class InfoRequest(BaseModel):
    file_path: str

class TrackMeta(BaseModel):
    name: str
    srate: float
    is_waveform: bool

class InfoResponse(BaseModel):
    duration_seconds: float
    tracks: list[TrackMeta]
    recommended_eeg1: str
    recommended_eeg2: str
    recommended_pulse: str
    recommended_spo2: str
    recommended_hr: str

class LoadRequest(BaseModel):
    file_path: str
    eeg1_track: str = "BIS/EEG1_WAV"
    eeg2_track: str = "BIS/EEG2_WAV"
    pulse_track: str = "SNUADC/PLETH"
    spo2_track:  str = "Solar8000/PLETH_SPO2"
    hr_track:    str = "Solar8000/HR"
    target_sample_rate: int = 256

class LoadResponse(BaseModel):
    session_id: str
    total_samples: int
    duration_seconds: float
    sample_rate: int

class ChunkRequest(BaseModel):
    session_id: str
    start_sample: int
    count: int = 256  # 1 second at 256 Hz

class SamplePoint(BaseModel):
    index: int
    eeg1: float
    eeg2: float
    pulse: float
    spo2: Optional[float] = None
    hr:   Optional[float] = None

class ChunkResponse(BaseModel):
    samples: list[SamplePoint]
    next_sample: int
    is_finished: bool

# ── Helpers ───────────────────────────────────────────────────────────────────

def _auto_recommend(track_names: list[str]) -> dict[str, str]:
    """Pick sensible defaults from the available track list."""
    def first_match(keywords: list[str]) -> str:
        for k in keywords:
            for t in track_names:
                if k.upper() in t.upper():
                    return t
        return track_names[0] if track_names else ""

    return {
        "eeg1":  first_match(["EEG1_WAV", "EEG1", "EEG"]),
        "eeg2":  first_match(["EEG2_WAV", "EEG2"]),
        "pulse": first_match(["PLETH", "PPG", "PULSE_WAVE"]),
        "spo2":  first_match(["PLETH_SPO2", "SPO2", "SAO2"]),
        "hr":    first_match(["PLETH_HR", "/HR", "HEART_RATE"]),
    }

def _load_track(vf, track: str, interval: float) -> Optional[np.ndarray]:
    """Load one track resampled to `interval` seconds/sample. Returns 1-D array or None."""
    if not track:
        return None
    try:
        arr = vf.to_numpy([track], interval)
        if arr is None or arr.size == 0:
            return None
        return arr[:, 0] if arr.ndim == 2 else arr.flatten()
    except Exception as e:
        logger.warning(f"Could not load track '{track}': {e}")
        return None

def _nan_to_none(v: float) -> Optional[float]:
    return None if (v is None or np.isnan(v)) else float(v)

# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/info", response_model=InfoResponse)
async def vital_info(req: InfoRequest):
    """Return track metadata without loading waveform data."""
    try:
        import vitaldb
    except ImportError:
        raise HTTPException(500, "vitaldb package not installed")

    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(404, f"File not found: {req.file_path}")

    try:
        vf = vitaldb.VitalFile(str(path))
    except Exception as e:
        raise HTTPException(400, f"Cannot open vital file: {e}")

    duration = float(vf.dtend - vf.dtstart)
    names = vf.get_track_names()
    rec = _auto_recommend(names)

    # Build track metadata (srate / waveform flag from internal track objects).
    # vf.trks is a dict {name: track_obj} in vitaldb ≥1.4.
    metas: list[TrackMeta] = []
    rate_map: dict[str, float] = {}
    try:
        for name, trk in vf.trks.items():
            srate = float(getattr(trk, "srate", 0) or 0)
            rate_map[name] = srate
    except (AttributeError, TypeError):
        # Older API: trks may be a list of objects with .name / .srate
        for trk in vf.trks:
            name = getattr(trk, "name", str(trk))
            srate = float(getattr(trk, "srate", 0) or 0)
            rate_map[name] = srate

    for n in sorted(names):
        sr = rate_map.get(n, 0)
        metas.append(TrackMeta(name=n, srate=sr, is_waveform=(sr >= 32)))

    return InfoResponse(
        duration_seconds=duration,
        tracks=metas,
        recommended_eeg1=rec["eeg1"],
        recommended_eeg2=rec["eeg2"],
        recommended_pulse=rec["pulse"],
        recommended_spo2=rec["spo2"],
        recommended_hr=rec["hr"],
    )


@router.post("/load", response_model=LoadResponse)
async def load_vital(req: LoadRequest):
    """
    Load and resample the selected tracks into memory.
    Returns session_id used for subsequent /chunk calls.
    This can take a few seconds for long recordings.
    """
    try:
        import vitaldb
    except ImportError:
        raise HTTPException(500, "vitaldb package not installed")

    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(404, f"File not found: {req.file_path}")

    logger.info(f"Loading vital file: {path.name}  fs={req.target_sample_rate}Hz")

    try:
        vf = vitaldb.VitalFile(str(path))
    except Exception as e:
        raise HTTPException(400, f"Cannot open vital file: {e}")

    interval = 1.0 / req.target_sample_rate

    # Load waveforms
    eeg1  = _load_track(vf, req.eeg1_track,  interval)
    eeg2  = _load_track(vf, req.eeg2_track,  interval)
    pulse = _load_track(vf, req.pulse_track, interval)
    spo2  = _load_track(vf, req.spo2_track,  interval)
    hr    = _load_track(vf, req.hr_track,    interval)

    # Use EEG1 length as reference; pad/trim others to match
    ref_len = len(eeg1) if eeg1 is not None else 0
    if ref_len == 0:
        raise HTTPException(400, f"EEG track '{req.eeg1_track}' returned no data")

    def align(arr: Optional[np.ndarray], length: int) -> np.ndarray:
        if arr is None:
            return np.full(length, np.nan)
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "eeg1":        align(eeg1,  ref_len).astype(np.float32),
        "eeg2":        align(eeg2,  ref_len).astype(np.float32),
        "pulse":       align(pulse, ref_len).astype(np.float32),
        "spo2":        align(spo2,  ref_len).astype(np.float32),
        "hr":          align(hr,    ref_len).astype(np.float32),
        "n_samples":   ref_len,
        "sample_rate": req.target_sample_rate,
        "duration":    ref_len / req.target_sample_rate,
        "file_name":   path.name,
    }

    logger.info(f"Session {session_id[:8]} loaded: {ref_len} samples "
                f"({ref_len/req.target_sample_rate:.1f}s)")

    return LoadResponse(
        session_id=session_id,
        total_samples=ref_len,
        duration_seconds=ref_len / req.target_sample_rate,
        sample_rate=req.target_sample_rate,
    )


@router.post("/chunk", response_model=ChunkResponse)
async def get_chunk(req: ChunkRequest):
    """Return a batch of samples starting at start_sample."""
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "Session expired or not found. Please reload.")

    n = sess["n_samples"]
    start = max(0, req.start_sample)
    end   = min(start + req.count, n)

    eeg1  = sess["eeg1"]
    eeg2  = sess["eeg2"]
    pulse = sess["pulse"]
    spo2  = sess["spo2"]
    hr    = sess["hr"]

    samples = [
        SamplePoint(
            index=i,
            eeg1=float(eeg1[i])  if not np.isnan(eeg1[i])  else 0.0,
            eeg2=float(eeg2[i])  if not np.isnan(eeg2[i])  else 0.0,
            pulse=float(pulse[i]) if not np.isnan(pulse[i]) else 0.0,
            spo2=_nan_to_none(float(spo2[i])),
            hr=_nan_to_none(float(hr[i])),
        )
        for i in range(start, end)
    ]

    return ChunkResponse(
        samples=samples,
        next_sample=end,
        is_finished=(end >= n),
    )


@router.delete("/{session_id}")
async def close_session(session_id: str):
    removed = _sessions.pop(session_id, None)
    if removed:
        logger.info(f"Session {session_id[:8]} closed ({removed['file_name']})")
    return {"status": "closed"}
