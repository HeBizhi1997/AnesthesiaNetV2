namespace EEGMonitor.AnesthesiaSleep.Models;

public class SleepStatistics
{
    public double TSTMinutes { get; set; }
    public double SEPercent { get; set; }
    public double SOLMinutes { get; set; }
    public double WASOMinutes { get; set; }

    public double N1Minutes { get; set; }
    public double N1Percent { get; set; }
    public double N2Minutes { get; set; }
    public double N2Percent { get; set; }
    public double N3Minutes { get; set; }
    public double N3Percent { get; set; }
    public double REMMinutes { get; set; }
    public double REMPercent { get; set; }
    public double REMLatencyMinutes { get; set; }

    public int ArousalCount { get; set; }
    public double ArousalIndex { get; set; }

    public double RecordingDurationMinutes { get; set; }
    public int TotalEpochs { get; set; }
    public int ArtifactEpochs { get; set; }

    public DateTime CalculatedAt { get; set; }
}
