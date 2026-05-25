namespace EEGMonitor.AnesthesiaSleep.Models;

public enum SedationScaleType
{
    Ramsay,
    RASS,
    MOAAS
}

public class SedationRecord
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public DateTime Timestamp { get; set; } = DateTime.Now;
    public TimeSpan? SessionOffset { get; set; }
    public SedationScaleType ScaleType { get; set; }
    public int Score { get; set; }
    public string ScoreLabel { get; set; } = string.Empty;
    public string? Observer { get; set; }
    public string? Notes { get; set; }

    public double? BISAtAssessment { get; set; }
    public double? SpO2AtAssessment { get; set; }
}
