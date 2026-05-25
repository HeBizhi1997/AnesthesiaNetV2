using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EEGMonitor.AnesthesiaSleep.Infrastructure.Messaging;
using EEGMonitor.AnesthesiaSleep.Infrastructure.Simulation;
using EEGMonitor.AnesthesiaSleep.Models;
using EEGMonitor.Models;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class SimulationViewModel : ObservableObject
{
    private readonly SleepSimulator _simulator;
    private readonly ResultBroker _broker;

    [ObservableProperty] private bool _isSimRunning;
    [ObservableProperty] private string _simStatus = "就绪";
    [ObservableProperty] private string _elapsedDisplay = "00:00:00";
    [ObservableProperty] private double _elapsedSeconds;
    [ObservableProperty] private double _simSpeed = 10.0;
    [ObservableProperty] private SleepScenarioType _selectedScenario = SleepScenarioType.NaturalSleep;

    // Latest vitals snapshot for display
    [ObservableProperty] private VitalSignsSnapshot? _latestVitals;

    public event Action? ResetRequested;

    public SleepScenarioType[] Scenarios { get; } = Enum.GetValues<SleepScenarioType>();
    public double[] SpeedOptions { get; } = { 1, 2, 5, 10, 20, 60 };

    public SimulationViewModel(SleepSimulator simulator, ResultBroker broker)
    {
        _simulator = simulator;
        _broker = broker;

        _simulator.ResultGenerated += OnResultGenerated;
        _simulator.VitalSignsUpdated += v => LatestVitals = v;
        _simulator.ClinicalEventGenerated += OnClinicalEvent;
        _simulator.StatusChanged += msg => SimStatus = msg;
        _simulator.SimulationStarted += () =>
        {
            LatestVitals = null;
            ElapsedSeconds = 0;
            ElapsedDisplay = "00:00:00";
            ResetRequested?.Invoke();
        };
        _simulator.SimulationCompleted += () =>
        {
            IsSimRunning = false;
            SimStatus = "模拟完毕";
        };
    }

    private void OnResultGenerated(ProcessedEEGResult result)
    {
        ElapsedSeconds = _simulator.Elapsed.TotalSeconds;
        ElapsedDisplay = _simulator.Elapsed.ToString(@"hh\:mm\:ss");
        Application.Current.Dispatcher.Invoke(() => _broker.Publish(result));
    }

    private void OnClinicalEvent(ClinicalEvent evt)
    {
        SimStatus = $"[{evt.Timestamp:HH:mm:ss}] {evt.Label}";
    }

    [RelayCommand]
    private void StartSimulation()
    {
        _simulator.Start(SelectedScenario);
        IsSimRunning = true;
    }

    [RelayCommand]
    private void StopSimulation()
    {
        _simulator.Stop();
        IsSimRunning = false;
    }

    [RelayCommand]
    private void SetSpeed(string speedStr)
    {
        if (double.TryParse(speedStr, out double speed))
        {
            SimSpeed = speed;
            _simulator.SetSpeed(speed);
        }
    }
}
