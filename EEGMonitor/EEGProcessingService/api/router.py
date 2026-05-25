import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import ProcessRequest, ProcessResponse
from preprocessing.eeg_preprocessor import EEGPreprocessor
from preprocessing.hrv_processor import HRVProcessor
from preprocessing.entropy_processor import EntropyProcessor
from preprocessing.fnox_processor import FNoxProcessor
from preprocessing.spindle_detector import SpindleDetector
from models.bis_predictor import BISPredictor

# Thread pool for parallel CPU-bound processing (numpy/scipy release GIL)
_executor = ThreadPoolExecutor(max_workers=3)

router = APIRouter()

# Singleton instances – warm up once on startup
_preprocessor: EEGPreprocessor | None = None
_hrv: HRVProcessor | None = None
_bis_predictor: BISPredictor | None = None
_entropy: EntropyProcessor | None = None
_fnox: FNoxProcessor | None = None
_spindle: SpindleDetector | None = None

# Propofol alpha correction — EXPERIMENTAL, NOT clinically validated.
# When propofol induces frontal alpha (8-13 Hz) concurrently with delta,
# the bimodal spectrum can paradoxically raise Shannon entropy even though
# the patient is adequately anesthetized.  This correction scales SE/RE
# down by clip(delta_power * alpha_power * factor, 0, cap).
# Set to 0.0 (default) to disable.  The GE Entropy Module does NOT apply
# any such correction; use only if independently validated against clinical BIS.
_propofol_alpha_factor: float = 0.0   # was 1.8 — disabled pending clinical validation
_propofol_alpha_cap:    float = 0.30


def init_services(model_path: str | None = None):
    global _preprocessor, _hrv, _bis_predictor, _entropy, _fnox, _spindle
    _preprocessor = EEGPreprocessor(sample_rate=128)
    _hrv = HRVProcessor(sample_rate=128)
    _bis_predictor = BISPredictor(model_path=model_path, sample_rate=128)
    _entropy = EntropyProcessor(sample_rate=128)
    _fnox = FNoxProcessor()
    _spindle = SpindleDetector(fs=100.0)  # NSM native rate; updated per-request if different
    logger.info("Processing services initialized")


@router.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _bis_predictor is not None and _bis_predictor._model is not None}


@router.post("/reset")
async def reset_state():
    """Reset stateful processors (call at the start of each monitoring session)."""
    if _preprocessor is not None:
        _preprocessor.reset()
    if _entropy is not None:
        _entropy.reset()
    if _bis_predictor is not None:
        _bis_predictor.reset_state()
    if _fnox is not None:
        _fnox.reset()
    if _spindle is not None:
        _spindle.reset()
    logger.info("Processor state reset")
    return {"status": "reset"}


@router.post("/process", response_model=ProcessResponse)
async def process_eeg(req: ProcessRequest):
    if _preprocessor is None:
        raise HTTPException(503, "Service not initialized")

    eeg = np.array(req.eeg_data, dtype=np.float64)  # (n_samples, n_channels)
    if eeg.ndim != 2 or eeg.shape[0] < 32:
        raise HTTPException(400, f"Invalid EEG shape: {eeg.shape}")

    # Update sample rate if changed
    if _preprocessor.fs != req.sample_rate:
        _preprocessor.fs = req.sample_rate
    if _bis_predictor is not None and _bis_predictor.input_fs != req.sample_rate:
        _bis_predictor.input_fs = req.sample_rate
        _bis_predictor.reset_state()
    if _spindle is not None and _spindle.fs != req.sample_rate:
        _spindle.fs = req.sample_rate


    # 1. EEG preprocessing (with optional device band powers for cross-validation)
    try:
        device_db = req.device_band_powers_db or None
        eeg_result = _preprocessor.preprocess(eeg, device_band_db=device_db)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # 2-5. Parallel processing: BIS, Entropy, HRV are independent; fNox depends on BIS+Entropy
    band_powers = {
        "delta": eeg_result["delta_power"],
        "theta": eeg_result["theta_power"],
        "alpha": eeg_result["alpha_power"],
        "beta":  eeg_result["beta_power"],
        "gamma": eeg_result["gamma_power"],
    }

    loop = asyncio.get_event_loop()

    # BIS prediction (GPU/CPU)
    async def compute_bis():
        try:
            return await loop.run_in_executor(
                _executor, lambda: _bis_predictor.predict(eeg, band_powers))
        except Exception as e:
            logger.warning(f"BIS prediction error: {e}")
            return None

    # Spectral Entropy (CPU)
    async def compute_entropy():
        if _entropy is None or not eeg_result.get("filtered_eeg"):
            return {}
        eeg_ch0 = np.array(eeg_result["filtered_eeg"], dtype=np.float64)
        return await loop.run_in_executor(_executor, lambda: _entropy.compute(eeg_ch0))

    # HRV from pulse wave (CPU, independent)
    async def compute_hrv():
        pulse_arr = np.array(req.pulse_wave, dtype=np.float64) if req.pulse_wave else np.array([])
        if pulse_arr.size == 0:
            return {}
        return await loop.run_in_executor(_executor, lambda: _hrv.compute(pulse_arr))

    # Launch BIS, Entropy, HRV in parallel
    bis_raw, entropy_result, hrv_result = await asyncio.gather(
        compute_bis(), compute_entropy(), compute_hrv()
    )

    bis = bis_raw

    # Propofol alpha correction (EXPERIMENTAL — disabled by default)
    se_raw = entropy_result.get("se")
    re_raw = entropy_result.get("re")
    if se_raw is not None and _propofol_alpha_factor > 0:
        propofol_sig = band_powers["delta"] * band_powers["alpha"]
        correction   = float(np.clip(propofol_sig * _propofol_alpha_factor,
                                     0.0, _propofol_alpha_cap))
        if correction > 0:
            entropy_result = {
                "se": se_raw * (1.0 - correction),
                "re": re_raw * (1.0 - correction) if re_raw is not None else None,
            }

    # fNox — depends on BIS + Entropy results (runs after they complete)
    fnox_result: dict = {}
    if _fnox is not None:
        fnox_result = _fnox.compute(
            bis=float(bis) if bis is not None else None,
            se=entropy_result.get("se"),
            re=entropy_result.get("re"),
        )

    # 6. Sleep spindle detection (sleep vs anesthesia discrimination)
    spindle_density = 0.0
    is_likely_sleep = False
    total_spindle_count = 0
    if _spindle is not None and len(eeg_result.get("filtered_eeg", [])) > 0:
        filtered = np.array(eeg_result["filtered_eeg"], dtype=np.float64)
        _spindle.detect(filtered)
        spindle_density = _spindle.density
        is_likely_sleep = _spindle.is_likely_sleep
        total_spindle_count = _spindle.total_spindle_count

    response = ProcessResponse(
        eeg_amplitude_uv=eeg_result.get("eeg_amplitude_uv", 0.0),
        eeg_dominant_hz=eeg_result.get("eeg_dominant_hz", 0.0),
        eeg_tonal_ratio=eeg_result.get("eeg_tonal_ratio", 0.0),
        raw_eeg=eeg_result["raw_eeg"],
        filtered_eeg=eeg_result["filtered_eeg"],
        delta_wave=eeg_result["delta_wave"],
        theta_wave=eeg_result["theta_wave"],
        alpha_wave=eeg_result["alpha_wave"],
        beta_wave=eeg_result["beta_wave"],
        gamma_wave=eeg_result["gamma_wave"],
        delta_power=eeg_result["delta_power"],
        theta_power=eeg_result["theta_power"],
        alpha_power=eeg_result["alpha_power"],
        beta_power=eeg_result["beta_power"],
        gamma_power=eeg_result["gamma_power"],
        dsa_matrix=eeg_result["dsa_matrix"],
        dsa_frequencies=eeg_result["dsa_frequencies"],
        dsa_times=eeg_result["dsa_times"],
        sqi=eeg_result["sqi"],
        bis=float(bis) if bis is not None else None,
        se=entropy_result.get("se"),
        re=entropy_result.get("re"),
        fnox=fnox_result.get("fnox"),
        fnox_mr_mode=fnox_result.get("mr_mode"),
        spindle_density=spindle_density,
        is_likely_sleep=is_likely_sleep,
        total_spindle_count=total_spindle_count,
        heart_rate=hrv_result.get("hr") or req.heart_rate,
        hrv_rmssd=hrv_result.get("hrv_rmssd"),
        pulse_wave=hrv_result.get("pulse_wave", req.pulse_wave),
        spo2=req.spo2,
        hw_clipping_pct=eeg_result.get("hw_clipping_pct", 0.0),
        hw_dc_offset_uv=eeg_result.get("hw_dc_offset_uv", 0.0),
        hw_is_saturated=eeg_result.get("hw_is_saturated", False),
        hw_adc_range_uv=eeg_result.get("hw_adc_range_uv", 0.0),
        device_delta_ratio=eeg_result.get("device_delta_ratio"),
        device_delta_discrepancy=eeg_result.get("device_delta_discrepancy"),
    )
    se_val = entropy_result.get("se")
    re_val = entropy_result.get("re")
    logger.info(
        f"BIS={f'{bis:.1f}' if bis is not None else 'N/A'} "
        f"SE={f'{se_val:.1f}' if se_val is not None else 'N/A'} "
        f"RE={f'{re_val:.1f}' if re_val is not None else 'N/A'} "
        f"SQI={eeg_result['sqi']:.0f} "
        f"Δ={eeg_result.get('delta_power', 0):.2f} α={eeg_result.get('alpha_power', 0):.2f}"
    )
    return response
