namespace EEGMonitor.AnesthesiaSleep.Models;

public class SleepReport
{
    public string ReportId { get; set; } = Guid.NewGuid().ToString("N")[..8];
    public DateTime GeneratedAt { get; set; } = DateTime.Now;
    public string PatientId { get; set; } = string.Empty;
    public string? SurgeryType { get; set; }
    public DateTime RecordingStart { get; set; }
    public DateTime RecordingEnd { get; set; }

    public SleepStatistics Statistics { get; set; } = new();
    public List<SleepEpoch> Hypnogram { get; set; } = new();
    public List<DrugRecord> DrugAdministrations { get; set; } = new();
    public List<SedationRecord> SedationAssessments { get; set; } = new();
    public List<PositionRecord> PositionChanges { get; set; } = new();
    public List<DesaturationEvent> DesaturationEvents { get; set; } = new();

    public double ODI3 { get; set; }
    public double ODI4 { get; set; }
    public double MeanSpO2 { get; set; }
    public double MinSpO2 { get; set; }
    public double SpO2Below90Percent { get; set; }

    public string? OperatorNotes { get; set; }
}
