"""
BIS predictor with online AnesthesiaNetV3 (MERIDIAN v13) streaming inference.
Falls back to a spectral heuristic if the model or dependencies are unavailable.

v13 feature set (38-dim): 5 bands + PE + SEF95 + LZC + 3×BSR + slope + gemg
                          + sigma + slow_osc + zcr + hjorth_mob + pac  (18/ch)

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

# tianjin/ must be on sys.path BEFORE importing src.pipeline modules
_MODEL_ROOT = Path(__file__).resolve().parents[3]   # tianjin/
sys.path.insert(0, str(_MODEL_ROOT))

try:
    from src.pipeline.context import EEGContext
except ImportError:
    EEGContext = None

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

    # Calibration: accumulate 60 s at 128 Hz before computing the MAD scale
    _CALIB_SEC  = 60
    _CALIB_SAMP = _TARGET_FS * _CALIB_SEC   # 7 680 samples

    def __init__(self, model_path: str | None = None, sample_rate: int = 256):
        self.input_fs   = sample_rate
        self._model     = None
        self._device    = "cpu"
        self._hx        = None          # GRU hidden state across calls
        self._cfg       = {}
        self._feat_ext  = None
        self._sqi_comp  = None
        # Channel count the model expects — derived from the checkpoint config in
        # _try_load_model (eeg.channels). Single-channel board → 1ch model just works.
        self._n_channels = self._N_CHANNELS

        if _TORCH_AVAILABLE:
            self._try_load_model(model_path)   # sets self._cfg + self._n_channels

        # Per-channel rolling buffer at 128 Hz — sized to the model's channel count.
        self._buf: list[deque] = [
            deque(maxlen=self._WIN_SAMP) for _ in range(self._n_channels)
        ]

        # Per-session amplitude normalisation (matches training VitalLoader.process_file)
        # Scale = MAD / 0.6745 ≈ robust σ, computed from the first _CALIB_SEC seconds.
        # Training divided every window by this scale → model expects unitless input ~O(1).
        self._calib_buf: list[list[float]] = [[] for _ in range(self._n_channels)]
        self._norm_scale: np.ndarray = np.ones(self._n_channels, dtype=np.float32)
        self._calibrated: bool = False
        self._dead_channel: int | None = None    # auto-detected dead channel
        self._mirror_source: int | None = None   # which channel to mirror from

    # ── Model loading ─────────────────────────────────────────────────────────

    def _try_load_model(self, model_path: str | None):
        import torch
        candidates = [
            model_path,
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "v13" / "best_model_v3.pt"),
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "v11" / "best_model_v3.pt"),
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "best_model.pt"),
        ]
        for path in candidates:
            if not (path and Path(path).exists()):
                continue
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                self._cfg = ck.get("cfg") or ck.get("config") or {}

                # Channel count from checkpoint (1ch single-electrode board vs 2ch).
                ch_list = (self._cfg.get("eeg", {}) or {}).get("channels")
                if ch_list:
                    self._n_channels = len(ch_list)

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
            self._feature_dim = self._feat_ext.total_feature_dim_for(self._n_channels)
            cfg_dim = self._cfg.get("model", {}).get("feature_dim")
            if cfg_dim and cfg_dim != self._feature_dim:
                logger.warning(
                    f"Config feature_dim={cfg_dim} vs computed={self._feature_dim} — "
                    f"using computed value. Check config consistency."
                )
            logger.info(f"Feature pipeline ready: dim={self._feature_dim} (v13={self._feature_dim})")
        except Exception as e:
            logger.warning(f"Feature pipeline init failed: {e} – falling back to heuristic")
            self._model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset_state(self):
        """Reset rolling buffer, GRU state, and amplitude calibration (call at session start)."""
        for buf in self._buf:
            buf.clear()
        self._hx = None
        self._calib_buf = [[] for _ in range(self._n_channels)]
        self._norm_scale = np.ones(self._n_channels, dtype=np.float32)
        self._calibrated = False
        self._dead_channel = None
        self._mirror_source = None
        logger.debug("BISPredictor state + amplitude calibration reset")

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

    def _get_filter_cfg(self) -> dict:
        """Window-filter config: checkpoint filters + notch BOTH 50 and 60 Hz (China + Korea
        grids). Shared by amplitude calibration and per-window filtering so they stay
        consistent."""
        filter_cfg = dict(self._cfg.get("filters", {
            "highpass_hz": 0.1, "lowpass_hz": 47.0, "notch_hz": [60.0], "notch_q": 30.0
        }))
        notches = list(filter_cfg.get("notch_hz", [60.0]) or [])
        for f in (50.0, 60.0):
            if f not in notches:
                notches.append(f)
        filter_cfg["notch_hz"] = notches
        return filter_cfg

    # ── Streaming model inference ─────────────────────────────────────────────

    def _streaming_predict(self, eeg_epoch: np.ndarray) -> float:
        import torch

        # 1. Adapt input channels → model's expected channel count (self._n_channels).
        #    Fewer input channels (single-electrode board) → mirror cyclically.
        #    More → take the first N.
        n_ch_in = eeg_epoch.shape[1]
        want = self._n_channels
        if n_ch_in >= want:
            eegN = eeg_epoch[:, :want]
        else:
            eegN = np.column_stack([eeg_epoch[:, i % n_ch_in] for i in range(want)])

        # 2. Resample chunk to 128 Hz
        g    = gcd(self._TARGET_FS, self.input_fs)
        up   = self._TARGET_FS // g
        down = self.input_fs // g
        try:
            resampled = np.stack(
                [resample_poly(eegN[:, ch].astype(np.float64), up, down)
                 for ch in range(self._n_channels)],
                axis=0,
            ).astype(np.float32)   # (n_channels, new_len)
        except Exception as e:
            logger.warning(f"Resample error: {e}")
            return float("nan")

        # 3. Append to rolling buffer + calibration accumulator
        for i in range(resampled.shape[1]):
            for ch in range(self._n_channels):
                v = float(resampled[ch, i])
                # Auto-mirror: if channel is dead, use good channel's data
                if getattr(self, '_dead_channel', None) is not None and ch == self._dead_channel:
                    v = float(resampled[self._mirror_source, i])
                self._buf[ch].append(v)
                if not self._calibrated:
                    self._calib_buf[ch].append(v)

        # Compute per-session amplitude scale once we have _CALIB_SEC of data.
        # v13: band-ratio aware normalization (matches training VitalLoader._compute_baseline_scale).
        #
        # Problem: pure MAD normalization divides all frequencies by the same scale.
        # In deep anesthesia, delta amplitude is 5-10× alpha/beta, so MAD is dominated
        # by low-frequency power. Dividing by this large value compresses high-frequency
        # components (alpha, beta, gamma) to near-zero, destroying spectral information
        # that the model relies on for BIS discrimination.
        #
        # Fix: when delta exceeds 90% of total spectral power, compute alpha+beta RMS
        # via time-domain bandpass filter and use it as the normalization reference.
        # The 2.5× factor maps narrow-band alpha+beta RMS to full-bandwidth equivalent σ
        # (empirically calibrated against awake EEG where delta ≈ 2.5× alpha RMS).
        if not self._calibrated and len(self._calib_buf[0]) >= self._CALIB_SAMP:
            scale = np.ones(self._n_channels, dtype=np.float32)
            _fcfg = self._get_filter_cfg()
            for ch in range(self._n_channels):
                arr = np.array(self._calib_buf[ch], dtype=np.float64)
                # Remove mains BEFORE measuring amplitude — otherwise the 50 Hz that the model
                # input has notched out dominates the MAD, leaving the real EEG ~16x under-scaled
                # (near-flat) → the model collapses to a constant BIS. (Matches training intent:
                # VitalDB was clean, so filter-then-MAD ≈ MAD-then-filter there.)
                arr = _window_filter(arr[None, :], self._TARGET_FS, _fcfg)[0]

                # Standard MAD fallback
                mad = float(np.median(np.abs(arr - np.median(arr))))
                mad_sigma = max(mad / 0.6745, 0.1)

                # Band-ratio detection: compute delta / total power ratio
                freqs = np.fft.rfftfreq(len(arr), d=1.0 / self._TARGET_FS)
                psd_full = np.abs(np.fft.rfft(arr - arr.mean())) ** 2
                total_p = psd_full.sum() + 1e-12
                delta_ratio = psd_full[(freqs >= 0.5) & (freqs < 4.0)].sum() / total_p

                if delta_ratio > 0.90:
                    # Delta-dominant → use alpha+beta RMS as reference (time-domain)
                    sos_ab = butter(4, [8.0 / (self._TARGET_FS / 2), 30.0 / (self._TARGET_FS / 2)],
                                    btype='bandpass', output='sos')
                    ab_signal = sosfiltfilt(sos_ab, arr)
                    ab_rms = float(np.std(ab_signal))
                    if ab_rms > 0.01:
                        scale[ch] = max(ab_rms * 2.5, 0.1)
                    else:
                        scale[ch] = mad_sigma
                else:
                    scale[ch] = mad_sigma

            # Dead-channel detection: if any channel has calibration σ < 2 uV,
            # it's likely a broken electrode/amplifier. Mirror the good channel.
            # Only possible with ≥2 channels — a single-electrode board has no fallback.
            dead_threshold = 2.0  # uV — below this, channel is considered dead
            if self._n_channels >= 2:
                for ch in range(self._n_channels):
                    arr = np.array(self._calib_buf[ch], dtype=np.float64)
                    rms = float(np.std(arr))
                    if rms < dead_threshold:
                        good_ch = 1 - ch  # assume the other channel is good
                        good_arr = np.array(self._calib_buf[good_ch], dtype=np.float64)
                        good_rms = float(np.std(good_arr))
                        if good_rms > dead_threshold:
                            logger.warning(
                                f"ch{ch} appears DEAD (RMS={rms:.2f} uV) — "
                                f"mirroring ch{good_ch} (RMS={good_rms:.1f} uV)"
                            )
                            scale[ch] = scale[good_ch]
                            self._dead_channel = ch
                            self._mirror_source = good_ch
                            break

            self._norm_scale = scale
            self._calibrated = True
            logger.info(
                "Amplitude calibration: " +
                " ".join(f"ch{c}={scale[c]:.3f}" for c in range(self._n_channels)) +
                f" (MAD-σ: {mad_sigma:.3f}, delta_ratio: {delta_ratio:.1%})"
            )
            self._calib_buf = [[] for _ in range(self._n_channels)]  # free memory

        if len(self._buf[0]) < self._WIN_SAMP:
            return float("nan")     # still warming up

        # 4. Extract 4-second window (n_channels, 512)
        window = np.array([list(b) for b in self._buf], dtype=np.float64)

        # 5. Apply window filters FIRST (highpass + 50/60 Hz notch + lowpass), matching v13
        #    training preprocessing. Checkpoint config notches 60 Hz (VitalDB / Korea grid);
        #    live hardware is China 50 Hz, so we notch BOTH (the extra band carries no signal
        #    on either grid and sits near the 47 Hz lowpass edge → harmless to training align).
        #
        #    ORDER MATTERS: filter must precede normalisation. The raw China-grid signal is
        #    ~99% 50 Hz mains; normalising first sets the MAD scale by the mains, then the notch
        #    leaves the real EEG ~16x under-scaled (near-flat) → model collapses to a constant
        #    BIS (~93, the bug we debugged). So: filter → then normalise.
        filter_cfg = self._get_filter_cfg()
        try:
            window = _window_filter(window, self._TARGET_FS, filter_cfg)
        except Exception as e:
            logger.warning(f"Filter error: {e}")

        # 6. Per-session amplitude normalisation (matching training), now on the mains-free
        #    signal. During the first _CALIB_SEC we use a per-window fallback (MAD of the
        #    current filtered 4-second slice) so inference isn't blocked during calibration.
        if self._calibrated:
            norm = self._norm_scale.astype(np.float64)
        else:
            norm = np.array([
                max(np.median(np.abs(window[ch] - np.median(window[ch]))) / 0.6745, 0.1)
                for ch in range(self._n_channels)
            ], dtype=np.float64)
        window = window / norm[:, np.newaxis]

        # 6. SQI + feature extraction
        try:
            if EEGContext is None:
                raise RuntimeError("EEGContext is None — import failed at module load")
            ctx = EEGContext(data=window, fs=float(self._TARGET_FS))
            if self._sqi_comp is None:
                raise RuntimeError("_sqi_comp is None")
            ctx = self._sqi_comp.process(ctx)
            if self._feat_ext is None:
                raise RuntimeError("_feat_ext is None")
            ctx = self._feat_ext.process(ctx)
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return float("nan")

        sqi_arr  = ctx.sqi.astype(np.float32)      # (2,)
        feat_arr = ctx.features.astype(np.float32)  # (feature_dim,)

        # 7. Model forward T=1
        try:
            wave_t = torch.tensor(
                window.astype(np.float32), dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(self._device)   # (1,1,2,512)

            feat_dim = getattr(self, '_feature_dim', len(feat_arr))
            feat_t = torch.tensor(feat_arr).unsqueeze(0).unsqueeze(0).to(self._device)   # (1,1,feature_dim)
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
        """
        Spectral heuristic — fallback when model unavailable.
        Calibrated: awake (δ≈0.2 α≈0.3 β≈0.2) → BIS ~90
                    deep anesthesia (δ≈0.7 α/β≈0.05) → BIS ~25
        """
        delta = band_powers.get("delta", 0.0)
        theta = band_powers.get("theta", 0.0)
        alpha = band_powers.get("alpha", 0.0)
        beta  = band_powers.get("beta",  0.0)
        gamma = band_powers.get("gamma", 0.0)
        total = delta + theta + alpha + beta + gamma + 1e-12

        # Beta ratio is the strongest single-band predictor of wakefulness
        beta_ratio = beta / total
        # Delta ratio drives BIS down
        delta_ratio = delta / total
        # Alpha/theta balance
        alpha_ratio = alpha / total

        # Base 50 + beta drives up, delta drives down
        bis = 50.0 + 120.0 * beta_ratio - 60.0 * delta_ratio + 30.0 * alpha_ratio
        return float(np.clip(bis, 0.0, 100.0))
