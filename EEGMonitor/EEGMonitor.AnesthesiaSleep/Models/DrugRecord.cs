namespace EEGMonitor.AnesthesiaSleep.Models;

public class DrugRecord
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public DateTime AdministrationTime { get; set; } = DateTime.Now;
    public TimeSpan? SessionOffset { get; set; }
    public string DrugName { get; set; } = string.Empty;
    public double Dose { get; set; }
    public string DoseUnit { get; set; } = "mg";
    public string Route { get; set; } = "IV";
    public double? InfusionRate { get; set; }
    public string? InfusionRateUnit { get; set; }
    public string? Notes { get; set; }
    public string? Operator { get; set; }

    public double? BISAtDrug { get; set; }
    public double? SpO2AtDrug { get; set; }
    public double? HRAtDrug { get; set; }
}
