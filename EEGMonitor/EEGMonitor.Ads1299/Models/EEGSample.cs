namespace Ads1299Monitor.Models;

/// <summary>One acquisition sample. Channels holds the EEG µV value(s); the optional
/// vital fields carry the latest pulse-sensor readings when present.</summary>
public record EEGSample(
    DateTime Timestamp,
    double[] Channels,
    double? SpO2 = null,
    double? HeartRate = null,
    double? PulseWaveValue = null
);
