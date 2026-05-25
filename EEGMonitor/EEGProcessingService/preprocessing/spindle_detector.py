"""
Sleep spindle detector for EEG-based sleep vs anesthesia discrimination.

Spindles are the strongest single-marker discriminator:
  - Sleep:  present (thalamic reticular nucleus pacemaking via GABA_A synapses)
  - Anesthesia: ABSENT (propofol saturates GABA_A → TRN can't oscillate at 12-14 Hz)

Algorithm (Wamsley et al., 2012; modified for 100-128 Hz real-time):
  1. Bandpass 12-14 Hz (sigma band)
  2. Hilbert envelope
  3. Threshold crossing with minimum duration (0.4 s)
  4. Scoring: spindle density per epoch (count / minute)

Returns per-epoch metrics:
  - spindle_count : number of spindles detected in this epoch
  - spindle_density : rolling 60-second spindle count
  - is_likely_sleep : bool threshold (spindle_density >= 2/min → sleep)
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from collections import deque
from loguru import logger


class SpindleDetector:
    """Real-time sleep spindle detector for EEG anesthesia monitors."""

    SIGMA_LO   = 12.0   # Hz
    SIGMA_HI   = 14.0   # Hz
    MIN_DURATION = 0.4  # seconds (shorter than standard 0.5 to catch fragmented spindles)
    AMPL_THRESHOLD = 2.0  # sigma envelope > 2.0 × median → candidate (Wamsley et al.)
    DENSITY_WINDOW = 60   # seconds rolling window
    SLEEP_THRESHOLD = 1.0  # spindles/min → likely sleep (low because 1s epochs fragment spindles)

    def __init__(self, fs: float = 100.0):
        self._event_times: deque[float] = deque()
        self._elapsed: float = 0.0
        self._spindle_count: int = 0
        self._fs = fs
        self._build_filter()

    @property
    def fs(self) -> float:
        return self._fs

    @fs.setter
    def fs(self, value: float):
        if value != self._fs:
            self._fs = value
            self._build_filter()
            logger.info(f"SpindleDetector: fs updated to {value} Hz, sigma filter rebuilt")

    def _build_filter(self):
        nyq = self._fs / 2.0
        self._sos = butter(4, [self.SIGMA_LO / nyq, self.SIGMA_HI / nyq],
                           btype="bandpass", output="sos")

    def detect(self, eeg: np.ndarray) -> int:
        """
        Detect spindles in an EEG epoch.

        eeg: 1-D float array (one epoch, typically 1-4 seconds at self.fs Hz)

        Returns number of spindles detected in this epoch (int).
        """
        n = len(eeg)
        if n < int(self.fs * 0.5):
            return 0

        try:
            # 1. Sigma bandpass
            sigma = sosfiltfilt(self._sos, eeg)

            # 2. Hilbert envelope
            envelope = np.abs(hilbert(sigma))
            median_env = float(np.median(envelope)) + 1e-9

            # 3. Threshold crossing with minimum duration
            above = envelope > (self.AMPL_THRESHOLD * median_env)
            min_samples = int(self.MIN_DURATION * self.fs)

            count = 0
            run_start = -1
            for i in range(len(above)):
                if above[i] and run_start < 0:
                    run_start = i
                elif not above[i] and run_start >= 0:
                    if i - run_start >= min_samples:
                        count += 1
                    run_start = -1
            # Check if still above at end of buffer
            if run_start >= 0 and len(above) - run_start >= min_samples:
                count += 1

            # 4. Update rolling density
            epoch_dur = n / self.fs
            self._elapsed += epoch_dur
            for _ in range(count):
                self._event_times.append(self._elapsed)
            # Remove events older than window
            cutoff = self._elapsed - self.DENSITY_WINDOW
            while self._event_times and self._event_times[0] < cutoff:
                self._event_times.popleft()
            self._spindle_count += count

            if count > 0:
                logger.debug(f"Spindle: {count} detected in epoch, "
                             f"density={self.density:.1f}/min")

            return count

        except Exception as e:
            logger.warning(f"Spindle detection error: {e}")
            return 0

    @property
    def density(self) -> float:
        """Spindle density: count per minute over rolling window."""
        if not self._event_times:
            return 0.0
        window_dur = self._elapsed - min(self._event_times[0],
                                          self._elapsed - self.DENSITY_WINDOW)
        window_dur = max(window_dur, self.DENSITY_WINDOW)
        return len(self._event_times) / (window_dur / 60.0)

    @property
    def is_likely_sleep(self) -> bool:
        """True if spindle density suggests natural sleep (not anesthesia)."""
        return self.density >= self.SLEEP_THRESHOLD and self._spindle_count >= 5

    @property
    def total_spindle_count(self) -> int:
        return self._spindle_count

    def reset(self):
        self._event_times.clear()
        self._elapsed = 0.0
        self._spindle_count = 0
