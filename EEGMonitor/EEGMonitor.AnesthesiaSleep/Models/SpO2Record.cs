namespace EEGMonitor.AnesthesiaSleep.Models;

public class DesaturationEvent
{
    public DateTime Timestamp { get; set; }
    public double SpO2Nadir { get; set; }
    public double BaselineSpO2 { get; set; }
    public double DesaturationPercent { get; set; }
    public bool IsODI3 { get; set; }
    public bool IsODI4 { get; set; }
    public double DurationSeconds { get; set; }
}
