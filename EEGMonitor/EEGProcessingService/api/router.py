import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import ProcessRequest, ProcessResponse
from preprocessing.eeg_preprocessor import EEGPreprocessor
from preprocessing.hrv_processor import HRVProcessor
from preprocessing.entropy_processor import EntropyProcessor
from models.bis_predictor import BISPredictor

router = APIRouter()

# Singleton instances – warm up once on startup
_preprocessor: EEGPreprocessor | None = None
_hrv: HRVProcessor | None = None
_bis_predictor: BISPredictor | None = None
_entropy: EntropyProcessor | None = None


def init_services(model_path: str | None = None):
    global _preprocessor, _hrv, _bis_predictor, _entropy
    _preprocessor = EEGPreprocessor(sample_rate=256)
    _hrv = HRVProcessor(sample_rate=256)
    _bis_predictor = BISPredictor(model_path=model_path, sample_rate=256)
    _entropy = EntropyProcessor(sample_rate=256)
    logger.info("Processing services initialized")


@router.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _bis_predictor is not None and _bis_predictor._model is not None}


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


    # 1. EEG preprocessing
    try:
        eeg_result = _preprocessor.preprocess(eeg)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # 2. BIS prediction
    band_powers = {
        "delta": eeg_result["delta_power"],
        "theta": eeg_result["theta_power"],
        "alpha": eeg_result["alpha_power"],
        "beta":  eeg_result["beta_power"],
        "gamma": eeg_result["gamma_power"],
    }
    try:
        bis_raw = _bis_predictor.predict(eeg, band_powers)  # type: ignore
    except Exception as e:
        logger.warning(f"BIS prediction error: {e}")
        bis_raw = None

    bis = bis_raw  # raw model output, no post-processing smoothing

    # 3. Spectral Entropy (SE / RE)
    entropy_result: dict = {}
    if _entropy is not None:
        eeg_ch0 = eeg[:, 0]
        entropy_result = _entropy.compute(eeg_ch0)

    # 4. HRV from pulse wave
    pulse_arr = np.array(req.pulse_wave, dtype=np.float64) if req.pulse_wave else np.array([])
    hrv_result = _hrv.compute(pulse_arr) if pulse_arr.size > 0 else {}  # type: ignore

    response = ProcessResponse(
        raw_eeg=eeg_result["raw_eeg"],
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
        heart_rate=hrv_result.get("hr") or req.heart_rate,
        hrv_rmssd=hrv_result.get("hrv_rmssd"),
        pulse_wave=hrv_result.get("pulse_wave", req.pulse_wave),
        spo2=req.spo2,
    )
    se_val = entropy_result.get("se")
    re_val = entropy_result.get("re")
    logger.debug(
        f"Processed epoch: BIS={bis:.1f if bis else 'N/A'}, "
        f"SE={se_val:.1f if se_val else 'N/A'}, "
        f"RE={re_val:.1f if re_val else 'N/A'}, "
        f"SQI={eeg_result['sqi']:.0f}"
    )
    return response
