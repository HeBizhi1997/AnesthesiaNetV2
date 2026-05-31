namespace Ads1299Monitor.Models;

/// <summary>A ~1-second epoch of samples sent to the Python service for inference.</summary>
public sealed class EEGDataChunk
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public List<EEGSample> Samples { get; set; } = new();
    public int SampleRate { get; set; } = 250;
    public int ChannelCount { get; set; } = 1;
    public double DurationSeconds => (EndTime - StartTime).TotalSeconds;
}
