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


def _deblink(x: np.ndarray, fs: float, k: float = 4.0, lp_hz: float = 5.0,
             pad_s: float = 0.15, max_frac: float = 0.5) -> np.ndarray:
    """
    去除大幅慢眼电瞬变(眨眼/眼动),仅用于幅度标定的去伪迹。
    在 <5Hz 分量上用稳健阈值(MAD)检出眨眼核,膨胀后线性插值掉。无眨眼或检出过多时原样返回。
    与 eeg_preprocessor._deblink_for_power 同法,前额单通道无独立 EOG 时的标准做法。
    """
    n = len(x)
    if n < int(fs // 2):
        return x
    nyq = fs / 2.0
    lp = butter(2, min(lp_hz, nyq * 0.9) / nyq, btype="low", output="sos")
    lf = sosfiltfilt(lp, x)
    med = np.median(lf)
    rstd = 1.4826 * (np.median(np.abs(lf - med)) + 1e-9)
    core = np.abs(lf - med) > k * rstd
    frac = float(core.mean())
    if frac == 0.0 or frac > max_frac:
        return x
    win = int(pad_s * fs) * 2 + 1
    mask = np.convolve(core.astype(np.float64), np.ones(win), mode="same") > 0.5
    good = ~mask
    if mask.all() or good.sum() < max(8, int(0.3 * n)):
        return x
    idx = np.arange(n)
    y = x.copy()
    y[mask] = np.interp(idx[mask], idx[good], x[good])
    return y


# ── Main predictor ────────────────────────────────────────────────────────────

class BISPredictor:
    _TARGET_FS  = 128       # model was trained at 128 Hz
    _WIN_SEC    = 4         # 4-second context window
    _WIN_SAMP   = _TARGET_FS * _WIN_SEC   # 512 samples
    _N_CHANNELS = 2
    # 幅度归一化:每窗 _deblink(P1-3)去眨眼;标定基线用"抗污染重建"——广带 MAD ≫ α+β 皮层幅度
    # (>2×=被 EOG/漂移/EMG 污染)或 δ 主导时改用 α+β RMS×2.5,逼近训练干净 MAD(域适配,见标定段)。

    # Calibration: accumulate 60 s at 128 Hz before computing the MAD scale
    _CALIB_SEC  = 60
    _CALIB_SAMP = _TARGET_FS * _CALIB_SEC   # 7 680 samples

    # ── 运动伪迹保持 + 输出平滑(消除体动导致的推理值骤降)─────────────────────────
    # 体动 = 突发大幅瞬变。判据用"相对突跳"(当前窗 RMS ≫ 近期基线中值)——天然区分
    # 体动(突发)与麻醉加深(渐变,基线会跟随)。命中则冻结 GRU、保持上次输出,不让坏窗污染。
    _MOTION_JUMP     = 3.0    # 窗 RMS > 此倍数×基线中值 ⇒ 运动伪迹
    _MOTION_HIST     = 30     # 基线统计窗口(≈30 个 epoch);median 抗离群,偶发体动不污染基线
    _MOTION_MIN_HIST = 10     # 需累积的最少历史
    _SMOOTH_N        = 5      # 输出中值平滑长度(≈5s);中值比 EMA 更能压单帧尖刺

    def __init__(self, model_path: str | None = None, sample_rate: int = 256):
        self.input_fs   = sample_rate
        self._model     = None
        self._device    = "cpu"
        self._hx        = None          # GRU hidden state across calls
        self._cfg       = {}
        self._feat_ext  = None
        self._sqi_comp  = None

        # ── awake-anchor 偏置校准（P0-2：默认关闭）────────────────────────────
        # 设计初衷：个体 PK/PD 差异(CE50 ±30%)→同等 EEG 形态下 BIS 系统性偏移；
        # 用开机清醒段把预测锚定到 ~95，求一个钳制常数偏置加到后续所有预测。
        #
        # 为何默认关闭（深度评估 P0-2）：
        #   1. BIS 量程非线性——"清醒处 +15"在维持/深麻醉处并不等于 +15，常数平移不成立；
        #   2. 方向不安全——整体抬高维持期读数 → 看起来更浅 → 可能诱导过量；
        #   3. 无商用监护用"开机 1 分钟学一个全程常数偏移"；个体差异应由模型/药物先验处理。
        # 仅当 checkpoint 的 cfg.inference.calibrate_awake 显式为 true 时才启用。
        self._bias_cal_enabled  = False
        self._bias_target_awake = 95.0   # 清醒锚点 BIS
        self._bias_window_n     = 60     # 需累积的有效预测数（≈60s）
        self._bias_clamp        = 15.0   # 偏置上限（防离谱校准）
        self._bias_min_awake    = 75.0   # 模型自身需≥此值才认为开机清醒，否则不校准
        self._bias               = 0.0
        self._bias_locked        = False
        self._cal_preds: list[float] = []
        self.last_bis_uncertainty: float = float("nan")  # P2：最近一次预测的 BIS 不确定度
        # 实际加载的模型信息（用于诚实上报，避免"代码声称 v17 实跑 v13"）
        self.model_path: str | None    = None   # 实际加载的 checkpoint 绝对路径
        self.model_tag: str            = "none" # 从路径推断的版本标签（v17/v14/v13/…）
        self.model_caps: dict          = {}     # {bis_head, bis_uncertainty, channels, val_mae}
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

        # 运动伪迹保持 + 输出平滑状态
        self._rms_hist: deque    = deque(maxlen=self._MOTION_HIST)   # 近期窗 RMS(µV),求基线中值
        self._bis_hist: deque    = deque(maxlen=self._SMOOTH_N)      # 近期输出 BIS,中值平滑
        self._last_emitted: float | None = None                     # 上次对外输出(伪迹时保持它)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _try_load_model(self, model_path: str | None):
        import torch
        # 部署目标 = v17（shared BIS 头修过渡区 2× 误差 + 异方差不确定度 + 药物派生相位）。
        # v17 的 .pt 需在 GPU 机器上训练产出后放入 outputs/checkpoints/v17/；放入即自动启用。
        # 回退链按"已知可用 + val_mae"排序：v17 → v13(MAE 4.57，最优可用) → v14(shared 头,MAE 5.8)
        # → best_model。回退时会 WARNING，避免静默跑到非目标模型。
        candidates = [
            model_path,
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "v17" / "best_model_v3.pt"),
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "v13" / "best_model_v3.pt"),
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "v14" / "best_model_v3.pt"),
            str(_MODEL_ROOT / "outputs" / "checkpoints" / "best_model.pt"),
        ]
        for path in candidates:
            if not (path and Path(path).exists()):
                continue
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                self._cfg = ck.get("cfg") or ck.get("config") or {}

                # 从 checkpoint 的 cfg.inference 读取校准"参数"(可调)。
                inf = self._cfg.get("inference", {}) or {}
                self._bias_target_awake = float(inf.get("awake_target_bis", self._bias_target_awake))
                self._bias_window_n     = int(inf.get("calib_window_sec", self._bias_window_n))
                self._bias_clamp        = float(inf.get("calib_clamp", self._bias_clamp))
                self._bias_min_awake    = float(inf.get("calib_min_awake_pred", self._bias_min_awake))
                # P0-2：awake 偏置的"开关"是部署级临床安全决策(默认关),**不允许被 checkpoint
                # 内嵌的训练期 cfg.inference.calibrate_awake 静默打开**——否则用旧 yaml 训出的 .pt
                # 会把已关掉的偏置又开起来。只接受显式运行时开关 EEG_CALIBRATE_AWAKE=1。
                import os
                env = os.environ.get("EEG_CALIBRATE_AWAKE")
                if env is not None:
                    self._bias_cal_enabled = env.strip().lower() in ("1", "true", "yes", "on")
                # 否则保持 __init__ 的 False（不被 checkpoint 覆盖）。

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

                # ── 诚实上报：记录实际加载的模型 + 能力，避免"声称 v17 实跑 v13"──────
                self.model_path = str(path)
                self.model_tag = next((t for t in ("v17", "v14", "v13", "v11")
                                       if f"/{t}/" in str(path).replace("\\", "/")), "legacy")
                mcfg = self._cfg.get("model", {}) or {}
                has_uncertainty = bool(mcfg.get("bis_uncertainty", False))
                self.model_caps = {
                    "tag": self.model_tag,
                    "bis_head": mcfg.get("bis_head", "gated"),
                    "bis_uncertainty": has_uncertainty,
                    "channels": self._n_channels,
                    "val_mae": mae,
                }
                logger.info(f"Loaded AnesthesiaNetV3 [{self.model_tag}] from {path}  "
                            f"val_mae={mae}  head={self.model_caps['bis_head']}  "
                            f"uncertainty={has_uncertainty}  ch={self._n_channels}")
                if self.model_tag != "v17":
                    logger.warning(
                        f"部署目标是 v17，但实际加载的是 [{self.model_tag}]。"
                        f"v17 的改进（shared 头修过渡区 2× 误差、异方差不确定度、药物派生相位）"
                        f"未生效。请在 GPU 机训练产出 outputs/checkpoints/v17/best_model_v3.pt 后放入。"
                    )
                if not has_uncertainty:
                    logger.warning(
                        f"[{self.model_tag}] 无异方差不确定度头 → pred_logvar 不会输出 → "
                        f"UI 的 BIS 可信区间(last_bis_uncertainty)将恒为 N/A。仅 v17 提供该能力。"
                    )
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

    def reset_state(self, keep_calibration: bool = False):
        """
        重置流式状态。

        keep_calibration=False(默认,新会话/换病人):清空一切——GRU、缓冲、幅度标定、平滑、偏置。
        keep_calibration=True(仅因采样率变更):只清 GRU + 滚动窗(时序断了、旧 fs 样本作废),
            **保留每病例幅度标定 + 平滑/基线**。幅度尺度以 µV 计、与 fs 无关,复用即可稳定出值;
            否则每次(常由体动致掉帧→测速抖动触发的)reset 都要 60s 重标定,期间用每窗回退归一化
            → 输出骤降(实测:每 90s 一次 reset 制造 30 次骤降)。
        """
        for buf in self._buf:
            buf.clear()
        self._hx = None
        if keep_calibration and self._calibrated:
            logger.info("reset_state(keep_calibration): 保留幅度标定/平滑,仅清 GRU+缓冲(采样率变更)")
            return
        self._calib_buf = [[] for _ in range(self._n_channels)]
        self._norm_scale = np.ones(self._n_channels, dtype=np.float32)
        self._calibrated = False
        self._dead_channel = None
        self._mirror_source = None
        # 重置运动伪迹保持 + 输出平滑
        self._rms_hist.clear()
        self._bis_hist.clear()
        self._last_emitted = None
        # P3：重置 awake-anchor 偏置校准
        self._bias = 0.0
        self._bias_locked = False
        self._cal_preds = []
        self.last_bis_uncertainty = float("nan")
        logger.debug("BISPredictor full reset (GRU + amplitude + bias)")

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
                # 去眨眼:前额电极眨眼/眼动会把 MAD 顶大 → 标定尺度偏大 → 皮层电被压没。
                arr = _deblink(arr, self._TARGET_FS)

                # Standard MAD fallback
                mad = float(np.median(np.abs(arr - np.median(arr))))
                mad_sigma = max(mad / 0.6745, 0.1)

                # Band-ratio detection: compute delta / total power ratio
                freqs = np.fft.rfftfreq(len(arr), d=1.0 / self._TARGET_FS)
                psd_full = np.abs(np.fft.rfft(arr - arr.mean())) ** 2
                total_p = psd_full.sum() + 1e-12
                delta_ratio = psd_full[(freqs >= 0.5) & (freqs < 4.0)].sum() / total_p

                # α+β 中频带 RMS:对 EOG(<4Hz)/漂移/EMG(>30Hz) 都更干净的幅度基准。
                sos_ab = butter(4, [8.0 / (self._TARGET_FS / 2), 30.0 / (self._TARGET_FS / 2)],
                                btype='bandpass', output='sos')
                ab_rms = float(np.std(sosfiltfilt(sos_ab, arr)))

                # 归一化基准 = 皮层 EEG 幅度(训练时干净 MAD 测的就是它)。前额脏信号的广带 MAD
                # 会被 EOG(<4Hz)/漂移/EMG(>30Hz)顶大(实测达 284µV，真皮层仅 10–30µV)→ 把皮层电压没。
                # α+β(8–30Hz) RMS 是抗污染的皮层幅度代理(避开 EOG/EMG)；×2.5 映射到全带等效 σ
                # (清醒 δ≈2.5×α RMS)。
                # P1-2(修订):当广带 MAD 远大于 α+β 重建值(>2×=基线被污染)或 δ 主导时改用 α+β 重建，
                #   使脏信号的归一化逼近训练时的"干净 MAD"分布——这是【域适配】(把脏信号拉回训练
                #   分布)，非偏斜；判据用相对比值(与量纲/montage 无关),清醒干净信号 ratio≈1 不触发。
                #   (上一版误删此分支 → scale 由 64µV 退回 284µV;回放证据促成本次修订。)
                ab_recon = ab_rms * 2.5
                delta_dominant = delta_ratio > 0.90
                contaminated   = (not delta_dominant) and (mad_sigma > 2.0 * ab_recon)
                if (delta_dominant or contaminated) and ab_rms > 0.01:
                    scale[ch] = max(ab_recon, 0.1)
                    if contaminated:
                        logger.info(f"ch{ch} 基线被污染(广带σ={mad_sigma:.0f}µV ≫ α+β重建{ab_recon:.0f}µV) "
                                    f"→ 用 α+β 皮层幅度基准 scale={scale[ch]:.1f}")
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

        # P1-4：爆发抑制(BS)安全权威 —— 必须在任何伪迹插值之前、在归一化前的 µV 窗(ch0=真实信号)
        #        上量抑制占比。抑制段临床定义 <5µV(Burst suppression: PMC8648516 / Wikipedia)。
        #        这是独立于 NN 的硬安全网：NN 可能不给 BSR 特征足够权重 → 漏判过深；此处强制压低 BIS。
        #        win_max>20µV 门控排除"平线/电极脱落"被误当抑制(脱落时无爆发)。
        supp_frac = float(np.mean(np.abs(window[0]) < 5.0))
        win_max   = float(np.max(np.abs(window[0])))
        win_rms   = float(np.std(window[0]))   # 用于运动伪迹检测(下方)

        # P1-3：每窗去伪迹(前端去 EOG/EMG)。前额采集先天受眼电/肌电污染；训练数据(VitalDB)本就干净。
        #        对脏信号去伪迹 = 把它拉回训练分布(对干净信号 frac≈0 原样返回)，优于"靠归一化补偿"
        #        (P1-2 已移除的偏斜支路)。**抑制主导(supp_frac≥0.5)时跳过** —— 此时大瞬变是真实
        #        爆发(burst)而非眨眼，去伪迹会误删爆发；deblink 仅用于清醒/浅麻醉的眼电。
        if supp_frac < 0.5:
            for ch in range(self._n_channels):
                window[ch] = _deblink(window[ch], self._TARGET_FS)

        # 运动伪迹保持:窗 RMS 相对近期基线突跳(>_MOTION_JUMP×)⇒ 体动 → 冻结 GRU、保持上次输出，
        # 不让坏窗污染递归状态/画出骤降。基线用 median(抗离群);渐变(麻醉加深)会被基线跟随,不误判。
        # (抑制主导窗不参与:那是真实 BS,不是体动。)
        motion = False
        if self._calibrated and supp_frac < 0.5 and len(self._rms_hist) >= self._MOTION_MIN_HIST:
            base = float(np.median(self._rms_hist))
            if base > 1e-6 and win_rms > self._MOTION_JUMP * base:
                motion = True
        self._rms_hist.append(win_rms)
        if motion and self._last_emitted is not None:
            logger.info(f"运动伪迹:窗RMS={win_rms:.0f}µV ≫ 基线{base:.0f}µV(×{win_rms/base:.1f}) "
                        f"→ 冻结GRU/保持BIS={self._last_emitted:.0f}")
            self._bis_hist.append(self._last_emitted)   # 保持平滑缓冲连续
            return self._last_emitted

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
            raw_bis  = float(np.clip(bis_norm * 100.0, 0.0, 100.0))

            # P1-4：BSR 安全权威。>50% 窗 <5µV 且存在爆发(win_max>20µV) ⇒ 爆发抑制(过深)，
            #        强制 BIS 进入低区间(50%→25, 80%→10, 100%→0)，独立于 NN 输出。
            if supp_frac > 0.5 and win_max > 20.0:
                ceiling = float(np.clip(50.0 * (1.0 - supp_frac), 0.0, 30.0))
                if raw_bis > ceiling:
                    logger.info(f"BSR 安全权威：抑制占比={supp_frac:.0%}(爆发≈{win_max:.0f}µV) "
                                f"→ BIS {raw_bis:.0f}→{ceiling:.0f}")
                    raw_bis = ceiling

            # P2：异方差不确定度（Laplace 尺度 b → BIS 点数），供 UI 显示可信区间
            if "pred_logvar" in out:
                logb = float(out["pred_logvar"].squeeze().cpu().item())
                self.last_bis_uncertainty = float(np.exp(logb) * 100.0)

            # P3：每病例 awake-anchor 偏置校准 → 再经中值平滑对外输出
            return self._emit(self._apply_bias_calibration(raw_bis))

        except Exception as e:
            logger.error(f"Model inference error: {e}")
            self._hx = None   # reset state on error
            return float("nan")

    def _apply_bias_calibration(self, raw_bis: float) -> float:
        """
        P3：每病例常数偏置校准（awake-anchor）。

        开机后累积前 _bias_window_n 个有效预测；若其中位数 ≥ _bias_min_awake
        （模型自身判为清醒 → 录制确在诱导前），则把该中位数锚定到 _bias_target_awake，
        得到一个钳制在 ±_bias_clamp 内的常数偏置，加到此后所有预测上。
        若开机即非清醒（接电极时已麻醉），无可靠锚点 → 偏置保持 0。
        """
        if not self._bias_cal_enabled:
            return raw_bis
        if not self._bias_locked:
            self._cal_preds.append(raw_bis)
            if len(self._cal_preds) >= self._bias_window_n:
                med = float(np.median(self._cal_preds))
                if med >= self._bias_min_awake:
                    self._bias = float(np.clip(self._bias_target_awake - med,
                                               -self._bias_clamp, self._bias_clamp))
                    logger.info(f"Awake-anchor calibration: median_pred={med:.1f} "
                                f"→ bias={self._bias:+.1f}")
                else:
                    self._bias = 0.0
                    logger.info(f"Awake-anchor skipped (start not awake: "
                                f"median_pred={med:.1f} < {self._bias_min_awake:.0f})")
                self._bias_locked = True
        return float(np.clip(raw_bis + self._bias, 0.0, 100.0))

    def _emit(self, val: float) -> float:
        """对外输出前做中值平滑(≈_SMOOTH_N 秒),压掉残余的单帧尖刺;并记录为'上次有效值'供伪迹保持。"""
        self._bis_hist.append(val)
        out = float(np.median(self._bis_hist))
        self._last_emitted = out
        return out

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
