namespace EEGMonitor.Models;

public class EEGDataChunk
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public List<EEGSample> Samples { get; set; } = new();
    public int SampleRate { get; set; } = 256;
    public int ChannelCount { get; set; } = 4;
    public double DurationSeconds => (EndTime - StartTime).TotalSeconds;
}
