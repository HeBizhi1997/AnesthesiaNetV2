"""
BIS predictor with online AnesthesiaNetV3 (MERIDIAN) streaming inference.
Falls back to a spectral heuristic if the model or dependencies are unavailable.

Streaming design (mirrors training pipeline exactly):
  Each /process call delivers one 1-second EEG chunk at input_fs Hz.
  We maintain:
    • a 4-second rolling buffer at 128 Hz (model target rate)
    • GRU hidden state hx carried across calls
  Per call:
    1. Resample chunk from input_fs → 128 Hz
    2. Append resampled samples to rolling buffer
    3. Once buffer has 512 samples: filter → SQI → features → model T=1 step
    4. Return BIS scaled [0, 100]
"""
from __future__ import annotations

import sys
from collections import deque
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly

_MODEL_ROOT = Path(__file__).resolve().parents[3]   # tianjin/
sys.path.insert(0, str(_MODEL_ROOT))

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available – using heuristic BIS estimator")


# ── Window-level filters (matching training WindowFilter) ─────────────────────

def _window_filter(data: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """
    data: (n_channels, n_samples) float64.
    Applies highpass → notch(s) → lowpass in-place.
    Uses sosfiltfilt (zero-phase IIR) — same as training WindowFilter.
    """
    nyq = fs / 2.0
    out = data.astype(np.float64)

    hp = cfg.get("highpass_hz", 0.5)
    lp = cfg.get("lowpass_hz", 47.0)
    notches = cfg.get("notch_hz", [60.0])
    notch_q = cfg.get("notch_q", 30.0)

    if hp > 0:
        sos = butter(4, hp / nyq, btype="high", output="sos")
        for ch in range(out.shape[0]):
            out[ch] = sosfiltfilt(sos, out[ch])

    for freq in notches:
        if 0 < freq < nyq:
            b, a = iirnotch(freq / nyq, notch_q)
            for ch in range(out.shape[0]):
                out[ch] = filtfilt(b, a, out[ch])

    if lp > 0 and lp < nyq:
        sos = butter(4, lp / nyq, btype="low", output="sos")
        for ch in range(out.shape[0]):
            out[ch] = sosfiltfilt(sos, out[ch])

    return out


# ── Main predictor ────────────────────────────────────────────────────────────

class BISPredictor:
    _TARGET_FS  = 128       # model was trained at 128 Hz
    _WIN_SEC    = 4         # 4-second context window
    _WIN_SAMP   = _TARGET_FS * _WIN_SEC   # 512 samples
    _N_CHANNELS = 2

    def __init__(self, model_path: str | None = None, sample_rate: int = 256):
        self.input_fs   = sample_rate
        self._model     = None
        self._device    = "cpu"
        self._hx        = None          # GRU hidden state across calls
        self._cfg       = {}
        self._feat_ext  = None
        self._sqi_comp  = None

        # Per-channel rolling buffer at 128 Hz
        self._buf: list[deque] = [
            deque(maxlen=self._WIN_SAMP) for _ in range(self._N_CHANNELS)
        ]

        if _TORCH_AVAILABLE:
            self._try_load_model(model_path)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _try_load_model(self, model_path: str | None):
        import torch
        candidates = [
            model_path,
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "v11" / "best_model_v3.pt"),
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "best_model.pt"),
            str(_MODEL_ROOT / "checkpoints" / "best_model.pth"),
        ]
        for path in candidates:
            if not (path and Path(path).exists()):
                continue
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                self._cfg = ck.get("cfg") or ck.get("config") or {}

                from src.models.anesthesia_net_v3 import AnesthesiaNetV3
                model = AnesthesiaNetV3.from_config(self._cfg)
                model.load_state_dict(ck.get("model_state_dict", ck), strict=True)
                model.eval()
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(self._device)
                self._model = model

                self._init_pipeline()
                mae = ck.get("val_mae", "?")
                logger.info(f"Loaded AnesthesiaNetV3 from {path}  val_mae={mae}")
                return
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")

        logger.warning("No model checkpoint found – using heuristic BIS")

    def _init_pipeline(self):
        try:
            from src.pipeline.steps.features import FeatureExtractor
            from src.pipeline.steps.sqi import SQIComputer
            feat_cfg = self._cfg.get("features", {})
            sqi_cfg  = self._cfg.get("sqi", {})
            self._feat_ext = FeatureExtractor(feat_cfg, fs=float(self._TARGET_FS))
            self._sqi_comp = SQIComputer(sqi_cfg)
            dim = self._feat_ext.total_feature_dim_for(self._N_CHANNELS)
            logger.info(f"Feature pipeline ready: dim={dim}")
        except Exception as e:
            logger.warning(f"Feature pipeline init failed: {e} – falling back to heuristic")
            self._model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset_state(self):
        """Reset rolling buffer and GRU state (call at session start)."""
        for buf in self._buf:
            buf.clear()
        self._hx = None

    def predict(self, eeg_epoch: np.ndarray, band_powers: dict) -> float:
        """
        eeg_epoch : (n_samples, n_channels) at self.input_fs Hz.
        Returns BIS in [0, 100].
        """
        if self._model is not None:
            val = self._streaming_predict(eeg_epoch)
            if not np.isnan(val):
                return val
        return self._heuristic_bis(band_powers)

    # ── Streaming model inference ─────────────────────────────────────────────

    def _streaming_predict(self, eeg_epoch: np.ndarray) -> float:
        import torch

        # 1. Ensure we have 2 channels
        n_ch = eeg_epoch.shape[1]
        if n_ch < 2:
            eeg2 = np.column_stack([eeg_epoch, eeg_epoch])
        else:
            eeg2 = eeg_epoch[:, :2]

        # 2. Resample chunk to 128 Hz
        g    = gcd(self._TARGET_FS, self.input_fs)
        up   = self._TARGET_FS // g
        down = self.input_fs // g
        try:
            resampled = np.stack(
                [resample_poly(eeg2[:, ch].astype(np.float64), up, down)
                 for ch in range(self._N_CHANNELS)],
                axis=0,
            ).astype(np.float32)   # (2, new_len)
        except Exception as e:
            logger.warning(f"Resample error: {e}")
            return float("nan")

        # 3. Append to rolling buffer
        for i in range(resampled.shape[1]):
            for ch in range(self._N_CHANNELS):
                self._buf[ch].append(resampled[ch, i])

        if len(self._buf[0]) < self._WIN_SAMP:
            return float("nan")     # still warming up

        # 4. Extract 4-second window (2, 512)
        window = np.array([list(b) for b in self._buf], dtype=np.float64)

        # 5. Apply window filters (matching training preprocessing)
        filter_cfg = self._cfg.get("filters", {
            "highpass_hz": 0.5, "lowpass_hz": 47.0, "notch_hz": [60.0], "notch_q": 30.0
        })
        try:
            window = _window_filter(window, self._TARGET_FS, filter_cfg)
        except Exception as e:
            logger.warning(f"Filter error: {e}")

        # 6. SQI + feature extraction
        try:
            from src.pipeline.context import EEGContext
            ctx = EEGContext(data=window, fs=float(self._TARGET_FS))
            ctx = self._sqi_comp.process(ctx)
            ctx = self._feat_ext.process(ctx)
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return float("nan")

        sqi_arr  = ctx.sqi.astype(np.float32)      # (2,)
        feat_arr = ctx.features.astype(np.float32)  # (28,)

        # 7. Model forward T=1
        try:
            wave_t = torch.tensor(
                window.astype(np.float32), dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(self._device)   # (1,1,2,512)

            feat_t = torch.tensor(feat_arr).unsqueeze(0).unsqueeze(0).to(self._device)   # (1,1,28)
            sqi_t  = torch.tensor(sqi_arr).unsqueeze(0).unsqueeze(0).to(self._device)    # (1,1,2)

            with torch.no_grad():
                out = self._model(wave_t, feat_t, sqi_t, hx=self._hx)

            self._hx = out["h"]   # carry GRU state to next call
            bis_norm = float(out["pred_bis"].squeeze().cpu().item())   # [0, 1]
            return float(np.clip(bis_norm * 100.0, 0.0, 100.0))

        except Exception as e:
            logger.error(f"Model inference error: {e}")
            self._hx = None   # reset state on error
            return float("nan")

    # ── Heuristic fallback ────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_bis(band_powers: dict) -> float:
        delta = band_powers.get("delta", 0.0)
        alpha = band_powers.get("alpha", 0.0)
        beta  = band_powers.get("beta",  0.0)
        gamma = band_powers.get("gamma", 0.0)
        arousal_score = 0.4 * beta + 0.3 * gamma + 0.15 * alpha - 0.35 * delta
        return float(np.clip(50.0 + arousal_score * 80.0, 0.0, 100.0))
