namespace EEGMonitor.AnesthesiaSleep.Models;

public class VitalSignsSnapshot
{
    public DateTime Timestamp { get; set; }
    public int HeartRate { get; set; }
    public int SpO2 { get; set; }
    public int SystolicBP { get; set; }
    public int DiastolicBP { get; set; }
    public int MAP => (SystolicBP + 2 * DiastolicBP) / 3;
    public int RespiratoryRate { get; set; }
    public int EtCO2 { get; set; }
    public double Temperature { get; set; }
}
