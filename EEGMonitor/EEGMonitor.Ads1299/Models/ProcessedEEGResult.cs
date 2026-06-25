using Newtonsoft.Json;

namespace Ads1299Monitor.Models;

/// <summary>Inference result returned by the Python /process endpoint (one per epoch).</summary>
public sealed class ProcessedEEGResult
{
    [JsonProperty("timestamp")] public DateTime Timestamp { get; set; }

    // ── Depth / nociception ──
    [JsonProperty("bis")]  public double BIS { get; set; } = double.NaN;   // → qCON 催眠指数
    [JsonProperty("fnox")] public double? FNox { get; set; }               // → qNOX 伤害感受
    [JsonProperty("sqi")]  public double SQI { get; set; }
    [JsonProperty("electrode_status")] public string ElectrodeStatus { get; set; } = "ok";  // ok|lead_off|flat|saturated
    [JsonProperty("signal_valid")]     public bool SignalValid { get; set; } = true;
    [JsonProperty("se")]   public double? StateEntropy { get; set; }
    [JsonProperty("re")]   public double? ResponseEntropy { get; set; }

    // ── EEG component waves ──
    [JsonIgnore]                   public double[] RawEEG { get; set; } = Array.Empty<double>();
    [JsonProperty("filtered_eeg")] public double[] FilteredEEG { get; set; } = Array.Empty<double>();
    [JsonProperty("delta_wave")]   public double[] DeltaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("theta_wave")]   public double[] ThetaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("alpha_wave")]   public double[] AlphaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("beta_wave")]    public double[] BetaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("gamma_wave")]   public double[] GammaWave { get; set; } = Array.Empty<double>();
    [JsonProperty("emg_wave")]     public double[] EmgWave { get; set; } = Array.Empty<double>();   // 宽带肌电(≥30Hz)显示通道
    [JsonProperty("emg_amplitude_uv")] public double EmgAmplitudeUv { get; set; }                  // 肌电 RMS(µV)

    // ── Band power ratios (0-1) ──
    [JsonProperty("delta_power")] public double DeltaPower { get; set; }
    [JsonProperty("theta_power")] public double ThetaPower { get; set; }
    [JsonProperty("alpha_power")] public double AlphaPower { get; set; }
    [JsonProperty("beta_power")]  public double BetaPower { get; set; }
    [JsonProperty("gamma_power")] public double GammaPower { get; set; }

    // ── Vitals ──
    [JsonProperty("heart_rate")] public double? HeartRate { get; set; }
    [JsonProperty("hrv_rmssd")]  public double? HRV_RMSSD { get; set; }
    [JsonProperty("spo2")]       public double? SpO2 { get; set; }

    // ── Signal diagnostics ──
    [JsonProperty("eeg_amplitude_uv")] public double EegAmplitudeUv { get; set; }
    [JsonProperty("eeg_dominant_hz")]  public double EegDominantHz { get; set; }
    [JsonProperty("eeg_tonal_ratio")]  public double EegTonalRatio { get; set; }
    [JsonProperty("hw_is_saturated")]  public bool HwIsSaturated { get; set; }
}
