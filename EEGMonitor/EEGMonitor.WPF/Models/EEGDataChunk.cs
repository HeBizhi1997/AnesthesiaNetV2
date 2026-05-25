namespace EEGMonitor.Models;

public class EEGDataChunk
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public List<EEGSample> Samples { get; set; } = new();
    public int SampleRate { get; set; } = 128;
    public int ChannelCount { get; set; } = 2;
    public double DurationSeconds => (EndTime - StartTime).TotalSeconds;

    /// <summary>
    /// NSM device-reported band powers in dB (averaged over this chunk's time window).
    /// null when not connected to an NSM device.
    /// Keys: delta, theta, alpha, beta, gamma.  Values: dB as signed byte ints.
    /// </summary>
    public Dictionary<string, int>? DeviceBandPowersDb { get; set; }
}
