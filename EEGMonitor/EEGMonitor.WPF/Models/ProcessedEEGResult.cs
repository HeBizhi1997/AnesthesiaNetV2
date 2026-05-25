using Newtonsoft.Json;

namespace EEGMonitor.Models;

public class ProcessedEEGResult
{
    [JsonProperty("timestamp")]
    public DateTime Timestamp { get; set; }

    // ── Depth of Anesthesia ──
    [JsonProperty("bis")] public double BIS { get; set; } = double.NaN;
    [JsonProperty("q_nox")] public double? QNox { get; set; }
    [JsonProperty("spi")] public double? SPI { get; set; }
    [JsonProperty("sqi")] public double SQI { get; set; }
    [JsonProperty("se")] public double? StateEntropy { get; set; }
    [JsonProperty("re")] public double? ResponseEntropy { get; set; }
    [JsonProperty("fnox")] public double? FNox { get; set; }
    [JsonProperty("fnox_mr_mode")] public bool? FNoxMRMode { get; set; }

    // ── EEG Component Waves ──
    [JsonIgnore]                 public double[] RawEEG { get; set; } = Array.Empty<double>();  // persisted to raw_eeg.bin only
    [JsonProperty("filtered_eeg")] public double[] FilteredEEG { get; set; } = Array.Empty<double>();
    [JsonProperty("delta_wave")] public double[] DeltaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("theta_wave")] public double[] ThetaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("alpha_wave")] public double[] AlphaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("beta_wave")]  public double[] BetaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("gamma_wave")] public double[] GammaWave { get; set; } = Array.Empty<double>();

    // ── Band Power Ratios ──
    [JsonProperty("delta_power")] public double DeltaPower { get; set; }
    [JsonProperty("theta_power")] public double ThetaPower { get; set; }
    [JsonProperty("alpha_power")] public double AlphaPower { get; set; }
    [JsonProperty("beta_power")]  public double BetaPower { get; set; }
    [JsonProperty("gamma_power")] public double GammaPower { get; set; }

    // ── DSA ──
    [JsonProperty("dsa_matrix")]     public double[,] DSAMatrix { get; set; } = new double[0, 0];
    [JsonProperty("dsa_frequencies")] public double[] DSAFrequencies { get; set; } = Array.Empty<double>();
    [JsonProperty("dsa_times")]      public double[] DSATimes { get; set; } = Array.Empty<double>();

    // ── Vitals ──
    [JsonProperty("heart_rate")] public double? HeartRate { get; set; }
    [JsonProperty("hrv_rmssd")]  public double? HRV_RMSSD { get; set; }
    [JsonProperty("pulse_wave")] public double[] PulseWave { get; set; } = Array.Empty<double>();
    [JsonProperty("spo2")]       public double? SpO2 { get; set; }

    // ── Sleep vs anesthesia discrimination ──
    [JsonProperty("spindle_density")]     public double SpindleDensity { get; set; }
    [JsonProperty("is_likely_sleep")]     public bool IsLikelySleep { get; set; }
    [JsonProperty("total_spindle_count")] public int TotalSpindleCount { get; set; }

    // ── Signal diagnostics ──
    [JsonProperty("eeg_amplitude_uv")] public double EegAmplitudeUv { get; set; }
    [JsonProperty("eeg_dominant_hz")]  public double EegDominantHz  { get; set; }
    [JsonProperty("eeg_tonal_ratio")]  public double EegTonalRatio  { get; set; }

    // ── Hardware diagnostics (NSM device) ──
    [JsonProperty("hw_clipping_pct")]      public double HwClippingPct { get; set; }
    [JsonProperty("hw_dc_offset_uv")]      public double HwDcOffsetUv { get; set; }
    [JsonProperty("hw_is_saturated")]      public bool   HwIsSaturated { get; set; }
    [JsonProperty("hw_adc_range_uv")]      public double HwAdcRangeUv { get; set; }
    [JsonProperty("device_delta_ratio")]   public double? DeviceDeltaRatio { get; set; }
    [JsonProperty("device_delta_discrepancy")] public double? DeviceDeltaDiscrepancy { get; set; }
}
