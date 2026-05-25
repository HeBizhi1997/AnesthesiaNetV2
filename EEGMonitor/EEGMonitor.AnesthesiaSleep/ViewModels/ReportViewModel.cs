using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EEGMonitor.AnesthesiaSleep.Infrastructure.ReportGeneration;
using EEGMonitor.AnesthesiaSleep.Models;
using Microsoft.Win32;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class ReportViewModel : ObservableObject
{
    private readonly ReportGenerator _generator;
    private readonly SleepStatisticsViewModel _statisticsVM;
    private readonly SleepStagingViewModel _stagingVM;
    private readonly DrugAdministrationViewModel _drugVM;
    private readonly SedationScaleViewModel _sedationVM;
    private readonly PositionTrackingViewModel _positionVM;
    private readonly SpO2TrendViewModel _spo2VM;

    [ObservableProperty] private bool _includeRawEEG;
    [ObservableProperty] private bool _includeHypnogram = true;
    [ObservableProperty] private string _reportStatus = "";
    [ObservableProperty] private string _lastReportPath = "";
    [ObservableProperty] private string _previewText = "";
    [ObservableProperty] private string _operatorNotes = "";

    public event Action<string>? ReportGenerated;

    public ReportViewModel(
        ReportGenerator generator,
        SleepStatisticsViewModel statisticsVM,
        SleepStagingViewModel stagingVM,
        DrugAdministrationViewModel drugVM,
        SedationScaleViewModel sedationVM,
        PositionTrackingViewModel positionVM,
        SpO2TrendViewModel spo2VM)
    {
        _generator = generator;
        _statisticsVM = statisticsVM;
        _stagingVM = stagingVM;
        _drugVM = drugVM;
        _sedationVM = sedationVM;
        _positionVM = positionVM;
        _spo2VM = spo2VM;
    }

    public void SetPatientInfo(string patientId, string? surgeryType)
    {
        _patientId = patientId;
        _surgeryType = surgeryType;
    }

    private string _patientId = "";
    private string? _surgeryType;

    [RelayCommand]
    private void PreviewReport()
    {
        var report = BuildReport();
        var json = _generator.GenerateJsonString(report);
        PreviewText = json;
        ReportStatus = "Preview generated";
    }

    [RelayCommand]
    private async Task SaveReportJson()
    {
        var dlg = new SaveFileDialog
        {
            Filter = "JSON files (*.json)|*.json|All files (*.*)|*.*",
            DefaultExt = ".json",
            FileName = $"SleepReport_{DateTime.Now:yyyyMMdd_HHmmss}.json"
        };
        if (dlg.ShowDialog() == true)
        {
            try
            {
                var report = BuildReport();
                var path = await _generator.GenerateJsonAsync(report, dlg.FileName);
                LastReportPath = path;
                ReportStatus = $"Report saved: {path}";
                ReportGenerated?.Invoke(path);
            }
            catch (Exception ex)
            {
                ReportStatus = $"Error: {ex.Message}";
            }
        }
    }

    [RelayCommand]
    private async Task SaveReportCsv()
    {
        var dlg = new SaveFileDialog
        {
            Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*",
            DefaultExt = ".csv",
            FileName = $"SleepReport_{DateTime.Now:yyyyMMdd_HHmmss}.csv"
        };
        if (dlg.ShowDialog() == true)
        {
            try
            {
                var report = BuildReport();
                var csv = _generator.GenerateCsvSummary(report);
                await System.IO.File.WriteAllTextAsync(dlg.FileName, csv);
                LastReportPath = dlg.FileName;
                ReportStatus = $"Report saved: {dlg.FileName}";
                ReportGenerated?.Invoke(dlg.FileName);
            }
            catch (Exception ex)
            {
                ReportStatus = $"Error: {ex.Message}";
            }
        }
    }

    private SleepReport BuildReport()
    {
        return new SleepReport
        {
            PatientId = _patientId,
            SurgeryType = _surgeryType,
            Statistics = _statisticsVM.GetSnapshot(),
            Hypnogram = _stagingVM.RecentEpochs.ToList(),
            DrugAdministrations = _drugVM.GetRecords(),
            SedationAssessments = _sedationVM.GetRecords(),
            PositionChanges = _positionVM.GetRecords(),
            DesaturationEvents = _spo2VM.GetDesaturationEvents(),
            ODI3 = double.IsNaN(_spo2VM.Odi3) ? 0 : _spo2VM.Odi3,
            ODI4 = double.IsNaN(_spo2VM.Odi4) ? 0 : _spo2VM.Odi4,
            MeanSpO2 = double.IsNaN(_spo2VM.MeanSpO2) ? 0 : _spo2VM.MeanSpO2,
            MinSpO2 = double.IsNaN(_spo2VM.MinSpO2) ? 0 : _spo2VM.MinSpO2,
            SpO2Below90Percent = _spo2VM.T90Percent,
            OperatorNotes = string.IsNullOrWhiteSpace(OperatorNotes) ? null : OperatorNotes.Trim()
        };
    }
}
