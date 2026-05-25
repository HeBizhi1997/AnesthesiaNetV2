namespace EEGMonitor.AnesthesiaSleep.Models;

public class SleepEpoch
{
    public int EpochIndex { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public SleepStage Stage { get; set; } = SleepStage.Unknown;
    public double Confidence { get; set; }

    public double DeltaPowerAvg { get; set; }
    public double ThetaPowerAvg { get; set; }
    public double AlphaPowerAvg { get; set; }
    public double BetaPowerAvg { get; set; }
    public double GammaPowerAvg { get; set; }
    public double SpindleDensityAvg { get; set; }
    public double HeartRateAvg { get; set; }
    public double SpO2Avg { get; set; }
    public double SeValueAvg { get; set; }
    public double ReSeDiffAvg { get; set; }

    public bool IsArtifact { get; set; }
}
