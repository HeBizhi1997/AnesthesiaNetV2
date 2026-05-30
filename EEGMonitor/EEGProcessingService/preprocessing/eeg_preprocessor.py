import numpy as np
from scipy import signal
from scipy.signal import (butter, sosfiltfilt, iirnotch, filtfilt, welch, detrend,
                          sosfilt, sosfilt_zi, tf2sos)
from scipy.ndimage import uniform_filter1d
from loguru import logger


class _StreamSOS:
    """
    Stateful causal SOS filter for gap-free streaming. Carries the filter delay state (zi)
    across successive chunks so concatenated output is continuous (no per-chunk edge
    transients). Initialised to the steady state for the first sample to avoid a startup
    step from a large DC offset.
    """

    def __init__(self, sos):
        self._sos = sos
        self._zi = None

    def reset(self):
        self._zi = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return x
        if self._zi is None:
            self._zi = sosfilt_zi(self._sos) * x[0]
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y


class EEGPreprocessor:
    """
    EEG preprocessing for real-time display.

    Band definitions (per doc/eeg成分.md):
      δ  0.5 – 4   Hz   deep sleep / deep anesthesia
      θ  4   – 7   Hz   light sedation
      α  8   – 13  Hz   awake/relaxed
      β  13  – 30  Hz   active cognition
      γ  30  – 47  Hz   high-frequency; capped at 47 Hz to avoid EMG contamination

    Pipeline:
      1. Broadband bandpass  0.5–47 Hz  (SOS, zero-phase)
      2. 50 Hz notch
      3. Band wave extraction via per-band SOS bandpass
      4. Band power ratios   (Welch PSD on rolling 4s buffer, trapz, relative to 5-band sum)
      5. DSA spectrogram
      6. SQI heuristic
    """

    # Band boundaries aligned with v13 training feature extractor
    BANDS = {
        "delta": (0.5,  4.0),
        "theta": (4.0,  8.0),   # v13: 4-8 Hz (matches training)
        "alpha": (8.0,  13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 47.0),  # cap at 47 Hz – above is EMG territory
    }

    # Seconds of filtered EEG to buffer for stable Welch PSD (4s matches BIS buffer)
    PSD_BUFFER_SEC = 4.0

    # Eye-blink rejection — applied to ENERGY estimates only (band-power ratios + DSA).
    # Blinks are large, slow (<~5 Hz) ocular transients that dump huge energy into the
    # delta band and dominate the band-power ratio (δ can read 80%+ while the patient is
    # plainly awake). We detect them on the low-frequency component via a robust MAD
    # threshold and interpolate them out before the PSD. Raw + per-band display waveforms
    # are NOT touched (the user wants blinks visible there).
    BLINK_LP_HZ    = 5.0    # detect on the <5 Hz component (where blinks live)
    BLINK_K        = 4.0    # core = |lf| > K · robust-σ (1.4826·MAD)
    BLINK_PAD_S    = 0.15   # dilate each detected core ±150 ms (blink waveform is wider than its peak)
    BLINK_MAX_FRAC = 0.50   # if >50% flagged it's a sustained slow-wave state, not blinks → keep as-is

    def __init__(self, sample_rate: int = 128):
        self._fs = sample_rate
        self._psd_buffer: list[float] = []   # rolling buffer for stable PSD
        self._psd_buffer_max = int(sample_rate * self.PSD_BUFFER_SEC)
        # Stateful CAUSAL streaming filters for the display waveforms. The client streams
        # 1-second chunks; filtering each chunk independently (zero-phase filtfilt) produces
        # edge transients + visible discontinuities at every boundary. A causal filter that
        # carries its state across chunks is perfectly continuous (textbook real-time EEG).
        self._build_stream_filters()

    def _build_stream_filters(self):
        """(Re)build the causal streaming filter chain for the current sample rate."""
        nyq = self._fs / 2.0
        hp = butter(4, max(0.5, 0.01) / nyq, btype="high", output="sos")
        lp = butter(4, min(47.0, nyq * 0.99) / nyq, btype="low", output="sos")
        nb, na = iirnotch(50.0 / nyq, 30.0) if 50.0 < nyq else (None, None)
        sections = [hp]
        if nb is not None:
            # Cascade the 50 Hz notch 3× : a single causal Q=30 biquad only knocks ~9 dB off
            # the mains, leaving ~8% residual on the ADS1299 raw stream (which is ~99% 50 Hz).
            # Three causal passes ≈ the depth of a zero-phase filtfilt notch (−21 dB / 0.8%
            # residual) without introducing non-causal latency. Matches the vendor app's
            # "EEG 滤波器" cleanliness on the same hardware.
            nsos = tf2sos(nb, na)
            sections.extend([nsos, nsos, nsos])
        sections.append(lp)
        self._sf_broad = _StreamSOS(np.vstack(sections))
        # Per-band streaming bandpass (on the broadband output). γ tightened to 30–45 Hz to
        # avoid the 47 Hz broadband edge + 50 Hz mains shoulder that otherwise show as noise.
        disp_bands = dict(self.BANDS)
        disp_bands["gamma"] = (30.0, 45.0)
        self._sf_bands = {}
        for band, (lo, hi) in disp_bands.items():
            hic = min(hi, nyq * 0.99)
            if lo >= hic:
                self._sf_bands[band] = None
                continue
            self._sf_bands[band] = _StreamSOS(
                butter(4, [lo / nyq, hic / nyq], btype="bandpass", output="sos"))

    @property
    def fs(self) -> int:
        return self._fs

    @fs.setter
    def fs(self, value: int):
        if value != self._fs:
            self._fs = value
            self._psd_buffer_max = int(value * self.PSD_BUFFER_SEC)
            self._psd_buffer.clear()  # buffer was accumulated at old rate
            self._build_stream_filters()

    # ── Public entry point ────────────────────────────────────────────────────

    def preprocess(self, eeg_data: np.ndarray,
                   device_band_db: dict | None = None) -> dict:
        """
        eeg_data: (n_samples, n_channels) – uses channel 0 for all analysis.
        device_band_db: optional NSM device-reported band powers in dB
                        {"delta": dB, "theta": dB, "alpha": dB, "beta": dB, "gamma": dB}
        Returns a dict ready to be JSON-serialised for the C# frontend.
        """
        n_samples, n_channels = eeg_data.shape
        if n_samples < 32:
            return self._empty_result()

        chunk = eeg_data[:, 0].copy().astype(np.float64)
        chunk_len = len(chunk)

        # 0. Hardware diagnostics BEFORE any processing – detect issues at the source
        hw_diag = self._hardware_diagnostics(chunk)

        # 1. Causal streaming filter: broadband (0.5–47 Hz + 50 Hz notch) with state carried
        #    across chunks → continuous, no per-chunk edge transients/discontinuities.
        filtered = self._sf_broad(chunk)            # (chunk_len,) continuous broadband EEG

        # 2. Per-band display waveforms from the broadband stream (each band keeps its state).
        waves = {}
        for band in self.BANDS:
            sf = self._sf_bands.get(band)
            waves[band] = sf(filtered) if sf is not None else np.zeros_like(filtered)

        raw_ch = chunk - chunk.mean()   # raw_eeg output: this chunk, DC-removed for display

        # Blink-rejected copy for ENERGY estimates ONLY (band-power ratios + DSA). The raw
        # waveform and the per-band display waveforms above keep their blinks untouched.
        filtered_for_power, blink_frac = self._deblink_for_power(filtered)

        # 5. Band power ratios (Welch PSD on the rolling buffer — already stable)
        powers = self._band_powers(filtered_for_power)

        # 6. DSA spectrogram (newest chunk)
        freqs, times, dsa_db = self._dsa(filtered_for_power)

        # 7. Signal quality index (now aware of NSM ±127 µV clipping)
        sqi = self._sqi(raw_ch, filtered, hw_diag)

        amplitude_uv, dominant_hz, tonal_ratio = self._signal_diagnostics(filtered)

        # 7. Cross-validate with device band powers if available.
        #    When the NSM signed-byte ADC is clipping (>5% of samples at ±127 µV),
        #    the pipeline's Welch PSD is computed from a distorted waveform and its
        #    band power ratios are unreliable.  The NSM device computes its band
        #    powers internally from the full-resolution signal BEFORE the byte
        #    truncation, so they remain accurate.  Use them instead.
        device_validation = {}
        powers_source = "pipeline"  # for logging
        if device_band_db is not None:
            device_validation = self._validate_against_device(powers, device_band_db)
            clip_pct = hw_diag["clipping_pct"]

            if clip_pct > 5.0:
                # Severe clipping — pipeline PSD is garbage, use device band powers
                device_ratios = {
                    b: device_validation[f"device_{b}_ratio"]
                    for b in ["delta", "theta", "alpha", "beta", "gamma"]
                }
                powers = device_ratios
                powers_source = "device (clipping)"
                logger.warning(
                    f"Pipeline band powers replaced by device: clip={clip_pct:.0f}% "
                    f"pipeline δ={device_validation.get('delta_discrepancy', 0)*100:.0f}% off "
                    f"→ using device δ={device_ratios['delta']:.2f}"
                )
            elif clip_pct > 2.0:
                # Mild clipping — blend pipeline and device
                device_ratios = {
                    b: device_validation[f"device_{b}_ratio"]
                    for b in ["delta", "theta", "alpha", "beta", "gamma"]
                }
                w = (clip_pct - 2.0) / 3.0  # 0 at 2%, 1 at 5%
                for b in device_ratios:
                    powers[b] = powers[b] * (1 - w) + device_ratios[b] * w
                powers_source = f"blend (clip={clip_pct:.0f}% w={w:.2f})"
            elif device_validation.get("delta_discrepancy", 0) > 0.30:
                logger.warning(
                    f"Device/pipeline delta mismatch (no clipping): "
                    f"pipeline δ={powers['delta']:.2f} vs device δ={device_validation.get('device_delta_ratio', 0):.2f} "
                    f"(diff={device_validation['delta_discrepancy']:.2f}) — "
                    f"hw: clip={clip_pct:.0f}% dc={hw_diag['dc_offset_uv']:.1f}µV"
                )

        return {
            "raw_eeg":          raw_ch.tolist(),
            "filtered_eeg":     filtered.tolist(),
            "delta_wave":       waves["delta"].tolist(),
            "theta_wave":       waves["theta"].tolist(),
            "alpha_wave":       waves["alpha"].tolist(),
            "beta_wave":        waves["beta"].tolist(),
            "gamma_wave":       waves["gamma"].tolist(),
            "delta_power":      powers["delta"],
            "theta_power":      powers["theta"],
            "alpha_power":      powers["alpha"],
            "beta_power":       powers["beta"],
            "gamma_power":      powers["gamma"],
            "dsa_matrix":       dsa_db.tolist(),
            "dsa_frequencies":  freqs.tolist(),
            "dsa_times":        times.tolist(),
            "sqi":              sqi,
            "eeg_amplitude_uv": amplitude_uv,
            "eeg_dominant_hz":  dominant_hz,
            "eeg_tonal_ratio":  tonal_ratio,
            # Hardware diagnostics for UI display / debugging
            "hw_clipping_pct":      hw_diag["clipping_pct"],
            "hw_dc_offset_uv":      hw_diag["dc_offset_uv"],
            "hw_is_saturated":      hw_diag["is_saturated"],
            "hw_adc_range_uv":      hw_diag["adc_range_uv"],
            # Device cross-validation (only when device data available)
            "device_delta_ratio":   device_validation.get("device_delta_ratio"),
            "device_delta_discrepancy": device_validation.get("delta_discrepancy"),
        }

    # ── Filters ──────────────────────────────────────────────────────────────

    def _bandpass(self, data: np.ndarray, lo: float, hi: float,
                  order: int = 4) -> np.ndarray:
        """
        Zero-phase Butterworth bandpass using SOS form.
        SOS is numerically stable even for narrow low-frequency bands (delta 0.5-4 Hz).
        Clamps hi to 0.99×Nyquist to prevent boundary failures at any sample rate.
        Uses padlen proportional to the lowest-frequency period to suppress edge
        transients that would otherwise leak into band power estimates.
        """
        nyq = self.fs / 2.0
        lo_c = max(lo, 0.01)
        hi_c = min(hi, nyq * 0.99)
        if lo_c >= hi_c:
            return np.zeros_like(data)
        sos = butter(order, [lo_c / nyq, hi_c / nyq], btype="bandpass", output="sos")
        # pad at least one period of the lowest frequency to reduce edge transients
        # but must be strictly less than the signal length for sosfiltfilt
        padlen = max(3 * order, min(int(self.fs / lo_c), len(data) - 1))
        return sosfiltfilt(sos, data, padlen=padlen)

    def _notch(self, data: np.ndarray, freq: float, q: float = 30.0) -> np.ndarray:
        nyq = self.fs / 2.0
        if freq >= nyq:
            return data
        b, a = iirnotch(freq / nyq, q)
        return filtfilt(b, a, data)

    # ── Spectral analysis ─────────────────────────────────────────────────────

    def _deblink_for_power(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect eye-blink transients and interpolate them out — for band-power / DSA
        energy estimates ONLY (never for the raw or per-band display waveforms).

        Method (single frontal channel, no separate EOG):
          1. Isolate the <5 Hz component (zero-phase LP) — blinks are slow & large there.
          2. Robust threshold: core = |lf − median| > K·(1.4826·MAD). MAD is dominated by
             the sparse non-blink background, so a real blink stands out while a uniformly
             high-amplitude slow-wave state does NOT (its MAD is large → threshold high).
          3. Dilate each core ±BLINK_PAD_S to cover the full blink waveform.
          4. Linearly interpolate the flagged samples from the clean neighbours.

        Returns (cleaned_signal, blink_fraction). Falls back to the original signal when
        there is nothing to do or when >BLINK_MAX_FRAC is flagged (sustained slow waves).
        """
        n = len(x)
        if n < self.fs // 2:
            return x, 0.0
        nyq = self.fs / 2.0
        lp = butter(2, min(self.BLINK_LP_HZ, nyq * 0.9) / nyq, btype="low", output="sos")
        lf = sosfiltfilt(lp, x)
        med = np.median(lf)
        rstd = 1.4826 * (np.median(np.abs(lf - med)) + 1e-9)
        core = np.abs(lf - med) > self.BLINK_K * rstd
        frac = float(core.mean())
        if frac == 0.0 or frac > self.BLINK_MAX_FRAC:
            return x, frac
        win = int(self.BLINK_PAD_S * self.fs) * 2 + 1
        mask = np.convolve(core.astype(np.float64), np.ones(win), mode="same") > 0.5
        good = ~mask
        if mask.all() or good.sum() < max(8, int(0.3 * n)):
            return x, frac
        idx = np.arange(n)
        y = x.copy()
        y[mask] = np.interp(idx[mask], idx[good], x[good])
        return y, float(mask.mean())

    def _band_powers(self, eeg: np.ndarray) -> dict:
        """
        Relative band powers summing to 1.0 across the 5 defined bands.

        Uses a rolling 4-second buffer of filtered EEG for stable Welch PSD.
        A single 1-second chunk at 200 Hz has only 1 Welch segment (nperseg=200
        with 256 samples → 1.28 effective segments), producing high-variance
        estimates that inflate delta.  By pooling the last 4 seconds we get
        4-16× more segments, matching the stability of the NSM device's internal
        PSD computation.
        """
        # Maintain rolling buffer
        self._psd_buffer.extend(float(x) for x in eeg)
        max_len = self._psd_buffer_max
        if len(self._psd_buffer) > max_len:
            self._psd_buffer = self._psd_buffer[-max_len:]

        buf = np.array(self._psd_buffer, dtype=np.float64)

        # Use 1 Hz frequency resolution (nperseg = fs) to ensure every
        # clinical band gets at least 3-4 frequency bins.  With a 4-second
        # buffer this gives 4-8× more Welch segments than a single 1-second
        # chunk, dramatically reducing estimate variance.
        #   nperseg = fs  →  1 Hz bins  → 7+ segments for 4s at 200 Hz
        nperseg = self.fs
        nperseg = max(64, min(nperseg, len(buf) // 2))
        noverlap = nperseg // 2

        freqs, psd = welch(buf, fs=self.fs, nperseg=nperseg, noverlap=noverlap)

        raw_powers = {}
        for band, (lo, hi) in self.BANDS.items():
            idx = (freqs >= lo) & (freqs < hi)
            raw_powers[band] = float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0

        # Relative normalisation: each value = fraction of 5-band total
        total = sum(raw_powers.values()) or 1.0
        return {k: v / total for k, v in raw_powers.items()}

    def reset(self):
        """Clear rolling buffers + streaming filter state (call at session start)."""
        self._psd_buffer.clear()
        self._sf_broad.reset()
        for sf in self._sf_bands.values():
            if sf is not None:
                sf.reset()

    def _dsa(self, eeg: np.ndarray,
             nperseg: int = 128,
             noverlap: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Density Spectral Array — short-time power spectrogram.
        noverlap=96 (75%) gives smooth time resolution at 1s updates.
        Frequency axis limited to 0–40 Hz (clinical BIS/DOA range).
        """
        f, t, Sxx = signal.spectrogram(
            eeg, fs=self.fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )
        mask = f <= 40.0
        Sxx_db = 10.0 * np.log10(Sxx[mask] + 1e-12)
        return f[mask], t, Sxx_db

    # ── Signal Quality Index ──────────────────────────────────────────────────

    def _sqi(self, raw: np.ndarray, filtered: np.ndarray,
             hw_diag: dict | None = None) -> float:
        """
        Four-component SQI [0, 100]:
          1. Flatline guard    — std < 0.1 µV → electrode off
          2. Amplitude guard   — saturation (threshold depends on ADC range) or weak < 1 µV
          3. HF artefact       — power >47 Hz vs total (EMG/ESU noise)
          4. Tonal interference — single narrow-band peak dominates (e.g. 25 Hz EMI)
             Real EEG has a broadband 1/f spectrum; if ≥40% of power sits in
             any ±2 Hz window, this is almost certainly electrical interference.

        When hw_diag is provided (NSM device), saturation is detected from
        the raw ADC data rather than a fixed 1000 µV threshold, because NSM's
        signed-byte ADC saturates at ±127 µV — well below normal clinical limits.
        """
        std = float(np.std(filtered))
        if std < 0.1:
            return 0.0

        # Saturation detection: use hardware diagnostics if available,
        # otherwise fall back to the 1000 µV heuristic (for ADS1299 / simulation).
        if hw_diag is not None and hw_diag.get("is_saturated"):
            # Signal clipped at ADC rail – spectrum is distorted
            clip_pct = hw_diag.get("clipping_pct", 100)
            # Penalty proportional to clipping severity: 10%→score=0.5, 50%→score=0.1
            sat_score = float(np.clip(1.0 - clip_pct / 20.0, 0.05, 1.0))
            amp_score = sat_score
        else:
            max_amp = float(np.max(np.abs(filtered)))
            if max_amp > 1000.0:
                return 5.0  # saturated electrode (ADS1299 / simulation)
            # Weak signal penalty (std < 1 µV → electrode barely touching)
            amp_score = float(np.clip(std / 1.0, 0.0, 1.0)) if std < 1.0 else 1.0

        nperseg = min(self.fs, len(raw))
        f, pxx = welch(raw, fs=self.fs, nperseg=nperseg)

        # Exclude power-line noise (50 Hz mains + 100 Hz harmonic, ±2 Hz) from the quality
        # reference. The pipeline notches the mains out, so it must NOT count as 'HF noise'
        # or skew the tonal/total reference. Without this, the raw signal — typically ~95%+
        # 50 Hz on this hardware — pins the HF score at its 0.25 floor and caps SQI near 25%
        # no matter how clean the actual EEG is. Genuine EMG (52–98, 102–125 Hz) still counts.
        mains_mask = np.zeros_like(f, dtype=bool)
        for mf in (50.0, 100.0):
            mains_mask |= (f >= mf - 2.0) & (f <= mf + 2.0)
        total_power = float(pxx[~mains_mask].sum()) + 1e-12

        # HF noise: power > 47 Hz (EMG/ESU), excluding the line harmonics above.
        nyq = self.fs / 2.0
        if nyq > 50:
            hf_ratio = float(pxx[(f > 47.0) & ~mains_mask].sum() / total_power)
            hf_score = float(np.clip(1.0 - hf_ratio * 1.5, 0.25, 1.0))
        else:
            hf_score = 1.0

        # Tonal interference: check if any ±2 Hz window holds >40% of total power.
        # Real EEG is broadband (1/f); a dominant narrow-band peak = EMI / bad contact.
        # Only check in the clinical range 4–45 Hz to avoid delta burst false positives.
        tonal_score = 1.0
        clinical_mask = (f >= 4.0) & (f <= 45.0)
        if clinical_mask.sum() > 4:
            pxx_clin = pxx.copy()
            pxx_clin[~clinical_mask] = 0.0
            freq_step = float(f[1] - f[0]) if len(f) > 1 else 1.0
            window_bins = max(1, int(2.0 / freq_step))   # ±2 Hz
            running_sum = uniform_filter1d(pxx_clin, size=window_bins * 2 + 1) * (window_bins * 2 + 1)
            peak_band_power = float(running_sum.max())
            peak_ratio = peak_band_power / total_power
            if peak_ratio > 0.40:
                # Linear penalty: 0.40→score=1.0, 0.80→score=0.0 (hard limit at 5)
                tonal_score = float(np.clip(1.0 - (peak_ratio - 0.40) * 2.5, 0.05, 1.0))

        return float(np.clip(100.0 * amp_score * hf_score * tonal_score, 0.0, 100.0))

    def _signal_diagnostics(self, filtered: np.ndarray) -> tuple[float, float, float]:
        """
        Returns (amplitude_uv, dominant_hz, tonal_ratio).
        amplitude_uv : std of filtered signal [µV] — expect 15-80 µV for real scalp EEG
        dominant_hz  : frequency with peak power in the 4-45 Hz clinical band
        tonal_ratio  : fraction of total power within ±2 Hz of that peak (>0.4 = interference)
        """
        amplitude_uv = float(np.std(filtered))
        nperseg = min(self.fs, len(filtered))
        f, pxx = welch(filtered, fs=self.fs, nperseg=nperseg)
        total = float(pxx.sum()) + 1e-12

        clin = (f >= 4.0) & (f <= 45.0)
        if clin.sum() == 0:
            return amplitude_uv, 0.0, 0.0

        peak_i = int(np.argmax(pxx[clin]))
        peak_f = float(f[clin][peak_i])

        band = (f >= peak_f - 2.0) & (f <= peak_f + 2.0)
        tonal_ratio = float(pxx[band].sum() / total)

        return amplitude_uv, peak_f, tonal_ratio

    @staticmethod
    def _empty_result() -> dict:
        return {
            "raw_eeg": [], "filtered_eeg": [], "delta_wave": [], "theta_wave": [],
            "alpha_wave": [], "beta_wave": [], "gamma_wave": [],
            "delta_power": 0.0, "theta_power": 0.0, "alpha_power": 0.0,
            "beta_power": 0.0, "gamma_power": 0.0,
            "dsa_matrix": [], "dsa_frequencies": [], "dsa_times": [],
            "sqi": 0.0,
            "eeg_amplitude_uv": 0.0, "eeg_dominant_hz": 0.0, "eeg_tonal_ratio": 0.0,
            "hw_clipping_pct": 0.0, "hw_dc_offset_uv": 0.0,
            "hw_is_saturated": False, "hw_adc_range_uv": 0.0,
            "device_delta_ratio": None, "device_delta_discrepancy": None,
        }

    # ── Hardware diagnostics ──────────────────────────────────────────────────

    def _hardware_diagnostics(self, raw: np.ndarray) -> dict:
        """
        Detect hardware-level signal issues from raw ADC data BEFORE any filtering.

        NSM-specific considerations:
          - ADC range is ±127 µV (signed byte); deep-anesthesia delta can reach
            200-500 µV at the scalp → clipping is expected and MUST be detected.
          - Persistent DC offset suggests ADC zero-calibration drift.
          - Very low amplitude (< 2 µV std) suggests electrode disconnect or
            ultra-high impedance.

        Returns diagnostic flags for downstream SQI and band-power validation.
        """
        n = len(raw)
        raw_min = float(raw.min())
        raw_max = float(raw.max())
        raw_std = float(np.std(raw))
        dc_offset = float(raw.mean())  # before detrend

        # Detect implied ADC rail from the data range.
        # For NSM signed byte: rails at ±127.  For other devices the data
        # may span a different range; we infer it from the absolute max.
        abs_max = max(abs(raw_min), abs(raw_max))
        # Assume the ADC rail is the next power-of-2 boundary above abs_max,
        # or 127 for NSM-class devices (signed byte), or 32767 for 16-bit.
        if abs_max <= 130:
            adc_rail = 127.0   # NSM signed byte
        elif abs_max <= 33000:
            adc_rail = 32767.0 # 16-bit ADC (e.g. ADS1299)
        else:
            adc_rail = abs_max * 1.1  # fallback

        # Clipping: samples within 3% of the rail
        clip_threshold = adc_rail * 0.97
        n_clipped = int(np.sum(np.abs(raw) >= clip_threshold))
        clipping_pct = (n_clipped / n) * 100.0 if n > 0 else 0.0

        # Saturation: >10% of samples at rail → signal is severely distorted
        is_saturated = clipping_pct > 10.0

        # Only warn on real ADC clipping. A large DC offset by itself is NOT a fault on
        # this hardware (the electrode half-cell potential is routinely thousands of µV and
        # is removed by the 0.5 Hz highpass) — it only matters if it pushes the input into
        # the rail, which clipping already captures. The DC value is still returned below
        # for diagnostics, just not logged every chunk.
        if clipping_pct > 5.0:
            logger.warning(
                f"HW: signal clipping detected — {clipping_pct:.0f}% of samples near "
                f"±{adc_rail:.0f} µV rail (min={raw_min:.1f} max={raw_max:.1f} std={raw_std:.1f}, "
                f"dc={dc_offset:.0f} µV). Band power estimates will be UNRELIABLE."
            )

        return {
            "clipping_pct":  clipping_pct,
            "dc_offset_uv":  dc_offset,
            "is_saturated":  is_saturated,
            "adc_range_uv":  adc_rail,
            "raw_min_uv":    raw_min,
            "raw_max_uv":    raw_max,
            "raw_std_uv":    raw_std,
        }

    # ── Device band-power cross-validation ────────────────────────────────────

    @staticmethod
    def _validate_against_device(pipeline_powers: dict, device_db: dict) -> dict:
        """
        Convert NSM device band powers from dB (signed byte) to relative ratios
        and compare with pipeline-computed values.

        device_db: {"delta": dB_int, "theta": dB_int, "alpha": dB_int, "beta": dB_int, "gamma": dB_int}

        Large discrepancies (>0.30 in relative power) indicate:
          - Signal clipping (delta power underestimated by device due to saturation)
          - DC offset / filter mismatch (delta overestimated by pipeline)
          - Sample rate mismatch (all bands shifted)
        """
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
        device_linear = {}
        for b in bands:
            db_val = device_db.get(b, -120)
            device_linear[b] = 10.0 ** (db_val / 10.0)

        device_total = sum(device_linear.values()) or 1.0
        device_ratios = {b: device_linear[b] / device_total for b in bands}

        discrepancies = {}
        for b in bands:
            pipeline_val = pipeline_powers.get(b, 0.0)
            device_val = device_ratios[b]
            discrepancies[f"{b}_discrepancy"] = abs(pipeline_val - device_val)

        return {
            "device_delta_ratio":  device_ratios.get("delta"),
            "device_theta_ratio":  device_ratios.get("theta"),
            "device_alpha_ratio":  device_ratios.get("alpha"),
            "device_beta_ratio":   device_ratios.get("beta"),
            "device_gamma_ratio":  device_ratios.get("gamma"),
            "delta_discrepancy":   discrepancies.get("delta_discrepancy", 0),
            "theta_discrepancy":   discrepancies.get("theta_discrepancy", 0),
            "alpha_discrepancy":   discrepancies.get("alpha_discrepancy", 0),
            "beta_discrepancy":    discrepancies.get("beta_discrepancy", 0),
            "gamma_discrepancy":   discrepancies.get("gamma_discrepancy", 0),
        }
