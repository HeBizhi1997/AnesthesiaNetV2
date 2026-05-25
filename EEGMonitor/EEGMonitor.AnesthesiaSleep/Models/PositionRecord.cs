namespace EEGMonitor.AnesthesiaSleep.Models;

public enum BodyPosition
{
    Supine,
    Prone,
    LeftLateral,
    RightLateral,
    Sitting,
    Standing
}

public class PositionRecord
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public DateTime Timestamp { get; set; } = DateTime.Now;
    public TimeSpan? SessionOffset { get; set; }
    public BodyPosition Position { get; set; }
    public string? Notes { get; set; }
}
