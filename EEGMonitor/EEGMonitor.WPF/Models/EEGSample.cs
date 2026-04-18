namespace EEGMonitor.Models;

public record EEGSample(
    DateTime Timestamp,
    double[] Channels,
    double? SpO2 = null,
    double? HeartRate = null,
    double? PulseWaveValue = null
);
