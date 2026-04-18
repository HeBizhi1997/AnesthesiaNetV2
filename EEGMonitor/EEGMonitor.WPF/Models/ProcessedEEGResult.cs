namespace EEGMonitor.Models;

public class ProcessedEEGResult
{
    public DateTime Timestamp { get; set; }

    // ── Depth of Anesthesia ──
    public double BIS { get; set; } = double.NaN;
    public double? QNox { get; set; }   // Placeholder – not yet implemented
    public double? SPI { get; set; }    // Placeholder – not yet implemented
    public double SQI { get; set; }     // Signal Quality Index 0-100
    public double? StateEntropy { get; set; }     // SE  0–91   (spectral entropy 0.8-32 Hz)
    public double? ResponseEntropy { get; set; }  // RE  0–100  (spectral entropy 0.8-47 Hz)

    // ── EEG Component Waves (time-series, same length as epoch) ──
    public double[] RawEEG { get; set; } = Array.Empty<double>();
    public double[] DeltaWave { get; set; } = Array.Empty<double>();  // 0.5-4 Hz
    public double[] ThetaWave { get; set; } = Array.Empty<double>();  // 4-8 Hz
    public double[] AlphaWave { get; set; } = Array.Empty<double>();  // 8-13 Hz
    public double[] BetaWave { get; set; } = Array.Empty<double>();   // 13-30 Hz
    public double[] GammaWave { get; set; } = Array.Empty<double>();  // 30-70 Hz

    // ── Band Power Ratios (0-1, sum ≈ 1) ──
    public double DeltaPower { get; set; }
    public double ThetaPower { get; set; }
    public double AlphaPower { get; set; }
    public double BetaPower { get; set; }
    public double GammaPower { get; set; }

    // ── DSA (Density Spectral Array) ──
    // Matrix columns = time bins, rows = frequency bins
    public double[,] DSAMatrix { get; set; } = new double[0, 0];
    public double[] DSAFrequencies { get; set; } = Array.Empty<double>();
    public double[] DSATimes { get; set; } = Array.Empty<double>();

    // ── Vitals ──
    public double? HeartRate { get; set; }
    public double? HRV_RMSSD { get; set; }
    public double[] PulseWave { get; set; } = Array.Empty<double>();
    public double? SpO2 { get; set; }
}
