namespace EEGMonitor.Models;

public class RecordingSession
{
    public Guid SessionId { get; set; } = Guid.NewGuid();
    public string PatientId { get; set; } = string.Empty;
    public string SurgeryType { get; set; } = string.Empty;
    public string Operator { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public int SampleRate { get; set; } = 256;
    public int ChannelCount { get; set; } = 4;
    public string RecordingDirectory { get; set; } = string.Empty;
    public List<ClinicalEvent> Events { get; set; } = new();
    public bool IsPlayback { get; set; }

    public TimeSpan Duration => EndTime.HasValue
        ? EndTime.Value - StartTime
        : DateTime.Now - StartTime;
}
