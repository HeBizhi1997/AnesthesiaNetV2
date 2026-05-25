using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EEGMonitor.AnesthesiaSleep.Models;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class DrugAdministrationViewModel : ObservableObject
{
    [ObservableProperty] private ObservableCollection<DrugRecord> _drugRecords = new();

    [ObservableProperty] private string _drugName = "";
    [ObservableProperty] private double _dose = 1.0;
    [ObservableProperty] private string _doseUnit = "mg";
    [ObservableProperty] private string _route = "IV";
    [ObservableProperty] private double? _infusionRate;
    [ObservableProperty] private string _infusionRateUnit = "mL/hr";
    [ObservableProperty] private string _drugNotes = "";
    [ObservableProperty] private string _operatorName = "";

    public List<string> DoseUnits { get; } = new() { "mg", "mcg", "g", "mL", "mg/kg", "mcg/kg" };
    public List<string> Routes { get; } = new() { "IV", "PO", "IM", "SC", "IN", "PR", "IH" };
    public List<string> InfusionRateUnits { get; } = new() { "mL/hr", "mcg/kg/min", "mg/hr" };

    public event Action<DrugRecord>? DrugAdded;

    [RelayCommand]
    private void AddDrug()
    {
        if (string.IsNullOrWhiteSpace(DrugName)) return;

        var record = new DrugRecord
        {
            DrugName = DrugName.Trim(),
            Dose = Dose,
            DoseUnit = DoseUnit,
            Route = Route,
            InfusionRate = InfusionRate,
            InfusionRateUnit = InfusionRate > 0 ? InfusionRateUnit : null,
            Notes = string.IsNullOrWhiteSpace(DrugNotes) ? null : DrugNotes.Trim(),
            Operator = string.IsNullOrWhiteSpace(OperatorName) ? null : OperatorName.Trim()
        };
        DrugRecords.Add(record);
        DrugAdded?.Invoke(record);

        DrugName = ""; Dose = 1.0; InfusionRate = null; DrugNotes = "";
    }

    [RelayCommand]
    private void RemoveDrug(DrugRecord? record)
    {
        if (record != null) DrugRecords.Remove(record);
    }

    public void Reset()
    {
        DrugRecords.Clear();
    }

    public List<DrugRecord> GetRecords() => DrugRecords.ToList();
}
