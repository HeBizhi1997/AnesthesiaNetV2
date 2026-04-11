"""
FastAPI inference service.

Exposes two endpoints for the frontend (WPF/MAUI/HTML):

  POST /infer/stream
    Body: { "eeg": [[ch1_samples], [ch2_samples]], "fs": 128 }
    Returns: { "bis": float, "sqi": [float, float], "features": {...} }

  GET /health
    Returns: { "status": "ok", "model": "AnesthesiaNet" }

Real-time flow:
  1. Frontend collects a 4-second EEG window (512 samples × 2 ch)
  2. POST to /infer/stream every second (sliding window)
  3. Service runs pipeline → model → returns BIS estimate
  4. Frontend plots DSA / BIS trend

The service maintains a per-session hidden state so the LNN has
temporal continuity across consecutive calls. Session ID is passed
as a header (X-Session-Id). Hidden states expire after 60 seconds
of inactivity.

Run with:
    uvicorn src.service.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

try:
    from fastapi import FastAPI, HTTPException, Header
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    # Stub for import-time safety
    class FastAPI:
        def __init__(self, **kw): pass
        def get(self, *a, **kw): return lambda f: f
        def post(self, *a, **kw): return lambda f: f
        def add_middleware(self, *a, **kw): pass
    class BaseModel: pass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.context import EEGContext
from src.pipeline.engine import EEGPipeline
from src.models.anesthesia_net import AnesthesiaNet


# ------------------------------------------------------------------ #
# App setup                                                            #
# ------------------------------------------------------------------ #

app = FastAPI(title="Anesthesia Depth Monitor API", version="1.0")

if _FASTAPI_AVAILABLE:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ------------------------------------------------------------------ #
# Global state (loaded once at startup)                                #
# ------------------------------------------------------------------ #

_model: Optional[AnesthesiaNet] = None
_pipeline: Optional[EEGPipeline] = None
_cfg: Optional[dict] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Session state: session_id → {"hx": tensor, "last_seen": float}
_sessions: Dict[str, dict] = {}
_SESSION_TIMEOUT = 60.0  # seconds


def _gc_sessions():
    now = time.time()
    expired = [sid for sid, s in _sessions.items()
               if now - s["last_seen"] > _SESSION_TIMEOUT]
    for sid in expired:
        del _sessions[sid]


def load_model_and_pipeline(
    config_path: str = "configs/pipeline_v1.yaml",
    checkpoint_path: str = "outputs/checkpoints/best_model.pt",
):
    global _model, _pipeline, _cfg
    with open(config_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    _pipeline = EEGPipeline.from_config(_cfg)

    _model = AnesthesiaNet.from_config(_cfg)
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Service] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[Service] WARNING: No checkpoint at {checkpoint_path}. "
              f"Using untrained model.")
    _model.to(_device)
    _model.eval()


# ------------------------------------------------------------------ #
# Request / Response schemas                                           #
# ------------------------------------------------------------------ #

class InferRequest(BaseModel):
    eeg: List[List[float]]   # shape: (n_channels, n_samples) — 4 s × 128 Hz = 512
    fs: float = 128.0


class InferResponse(BaseModel):
    bis: float               # 0-100
    sqi: List[float]         # per channel, 0-1
    features: Dict[str, float]
    lazy_mode: bool          # True if SQI too low → output held


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    return {"status": "ok", "model": "AnesthesiaNet",
            "device": str(_device), "model_loaded": _model is not None}


@app.post("/infer/stream", response_model=InferResponse)
def infer_stream(
    req: InferRequest,
    x_session_id: str = Header(default="default"),
):
    if _model is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _gc_sessions()

    # Parse and validate EEG
    eeg = np.array(req.eeg, dtype=np.float32)  # (n_ch, n_samples)
    if eeg.ndim != 2 or eeg.shape[0] not in (1, 2):
        raise HTTPException(status_code=400, detail="EEG must be shape (n_channels, n_samples)")

    # Pad to 2 channels if only 1 is provided
    if eeg.shape[0] == 1:
        eeg = np.repeat(eeg, 2, axis=0)

    # Run preprocessing pipeline
    ctx = EEGContext(data=eeg, fs=req.fs)
    try:
        ctx = _pipeline.run(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # SQI gating
    mean_sqi = float(np.mean(ctx.sqi)) if ctx.sqi is not None else 1.0
    sqi_threshold = _cfg["sqi"]["min_score"]
    lazy_mode = mean_sqi < sqi_threshold

    # Retrieve session hidden state
    sid = x_session_id
    hx = None
    if sid in _sessions:
        hx = _sessions[sid]["hx"]
        _sessions[sid]["last_seen"] = time.time()

    if not lazy_mode:
        wave_t = torch.from_numpy(ctx.data).to(_device)
        feat_t = torch.from_numpy(ctx.features).to(_device)
        sqi_t = torch.from_numpy(ctx.sqi).to(_device)

        bis_val, new_hx = _model.predict_single(wave_t, feat_t, sqi_t, hx)

        _sessions[sid] = {"hx": new_hx, "last_seen": time.time()}
    else:
        # Lazy mode: return last known value from session, don't update hx
        bis_val = _sessions[sid].get("last_bis", 50.0) if sid in _sessions else 50.0

    if not lazy_mode:
        _sessions[sid]["last_bis"] = bis_val

    # Build feature summary for frontend (DSA-friendly).
    # Layout must exactly match FeatureExtractor._channel_features():
    #   per channel (11): delta,theta,alpha,beta,gamma, PE, SEF95, LZC, BSR_2,BSR_5,BSR_10
    #   inter-channel (2): alpha_asymmetry, mean_sqi
    # Total: 2×11 + 2 = 24
    feat_names = [
        "delta_ch1", "theta_ch1", "alpha_ch1", "beta_ch1", "gamma_ch1",
        "pe_ch1", "sef95_ch1", "lzc_ch1",
        "bsr2_ch1", "bsr5_ch1", "bsr10_ch1",
        "delta_ch2", "theta_ch2", "alpha_ch2", "beta_ch2", "gamma_ch2",
        "pe_ch2", "sef95_ch2", "lzc_ch2",
        "bsr2_ch2", "bsr5_ch2", "bsr10_ch2",
        "alpha_asymmetry", "mean_sqi",
    ]
    features_dict = {}
    if ctx.features is not None:
        for i, v in enumerate(ctx.features):
            name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
            features_dict[name] = float(v)

    return InferResponse(
        bis=round(float(bis_val), 2),
        sqi=[float(s) for s in ctx.sqi] if ctx.sqi is not None else [1.0, 1.0],
        features=features_dict,
        lazy_mode=lazy_mode,
    )


# ------------------------------------------------------------------ #
# Entry point for direct run                                           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn
    load_model_and_pipeline()
    uvicorn.run(app, host="0.0.0.0", port=8000)
