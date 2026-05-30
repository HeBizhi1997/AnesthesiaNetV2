using System.Collections.ObjectModel;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EEGMonitor.AnesthesiaSleep.Infrastructure.Messaging;
using EEGMonitor.AnesthesiaSleep.Models;
using EEGMonitor.Infrastructure.Pipeline;
using EEGMonitor.Models;
using EEGMonitor.Services.Events;
using EEGMonitor.Services.Playback;
using EEGMonitor.Services.Processing;
using EEGMonitor.Services.Recording;
using EEGMonitor.Services.SerialPort;
using EEGMonitor.Services.Simulation;
using Microsoft.Extensions.Logging;
using OxyPlot;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class ShellViewModel : ObservableObject
{
    private readonly ISerialPortService _serial;
    private readonly NSMSerialService _nsmSerial;
    private readonly IPulseSerialService _pulseSerial;
    private readonly IEEGProcessingClient _processing;
    private readonly IRecordingService _recording;
    private readonly IPlaybackService _playback;
    private readonly IEventAnnotationService _events;
    private readonly DataPipeline _pipeline;
    private readonly IVitalSimulatorService _simulator;
    private readonly ResultBroker _broker;
    private readonly ILogger _logger;

    private DateTime? _sessionStart;
    private System.Timers.Timer? _durationTimer;
    private System.Timers.Timer? _clockTimer;
    private CancellationTokenSource? _healthCheckCts;
    private double _sessionElapsedSec;

    // ── Sub-ViewModels (exposed for XAML binding) ──────────────────────────
    public SleepStagingViewModel Staging { get; }
    public SleepStatisticsViewModel Statistics { get; }
    public SpO2TrendViewModel SpO2Trend { get; }
    public DrugAdministrationViewModel DrugAdmin { get; }
    public SedationScaleViewModel Sedation { get; }
    public PositionTrackingViewModel Position { get; }
    public ReportViewModel Report { get; }
    public SimulationViewModel Simulation { get; }

    // Rolling EEG waveform buffer (most recent 256 samples for chart)
    private readonly double[] _eegWaveBuffer = new double[256];
    private int _eegWaveWritePos;
    private DateTime _lastEegRefresh = DateTime.MinValue;
    private static readonly TimeSpan EegRefreshInterval = TimeSpan.FromMilliseconds(100);
    public double[] EegWaveBuffer => _eegWaveBuffer;
    public int EegWaveLen => Math.Min(_eegWaveWritePos, 256);

    private OxyPlot.PlotModel? _eegPlotModel;
    public OxyPlot.PlotModel EegPlotModel => _eegPlotModel ??= BuildEegPlotModel();

    private OxyPlot.PlotModel BuildEegPlotModel()
    {
        var model = new OxyPlot.PlotModel();
        model.Background = OxyPlot.OxyColor.FromRgb(0x09, 0x0E, 0x1C);
        model.PlotAreaBackground = OxyPlot.OxyColor.FromRgb(0x09, 0x0E, 0x1C);
        model.TextColor = OxyPlot.OxyColor.FromRgb(0x3D, 0x5A, 0x7A);

        model.Axes.Add(new OxyPlot.Axes.LinearAxis
        {
            Position = OxyPlot.Axes.AxisPosition.Left,
            Minimum = -80, Maximum = 80,
            Title = "μV", TitleColor = OxyPlot.OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            TextColor = OxyPlot.OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            AxislineColor = OxyPlot.OxyColor.FromRgb(0x18, 0x2C, 0x46),
            TicklineColor = OxyPlot.OxyColor.FromRgb(0x18, 0x2C, 0x46),
            MajorGridlineColor = OxyPlot.OxyColor.FromRgb(0x0C, 0x14, 0x22),
            StringFormat = "F0",
        });
        model.Axes.Add(new OxyPlot.Axes.LinearAxis
        {
            Position = OxyPlot.Axes.AxisPosition.Bottom,
            Minimum = 0, Maximum = 256,
            Title = "Sample", TitleColor = OxyPlot.OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            TextColor = OxyPlot.OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            AxislineColor = OxyPlot.OxyColor.FromRgb(0x18, 0x2C, 0x46),
            TicklineColor = OxyPlot.OxyColor.FromRgb(0x18, 0x2C, 0x46),
            MajorGridlineColor = OxyPlot.OxyColor.FromRgb(0x0C, 0x14, 0x22),
            StringFormat = "F0",
        });

        var series = new OxyPlot.Series.LineSeries
        {
            Color = OxyPlot.OxyColor.FromRgb(0x00, 0xC8, 0xFF),
            StrokeThickness = 1,
        };
        model.Series.Add(series);
        return model;
    }

    // ── Clock ──────────────────────────────────────────────────────────────
    [ObservableProperty] private string _currentTimeDisplay = "";
    [ObservableProperty] private string _currentDateDisplay = "";

    // ── Connection ─────────────────────────────────────────────────────────
    [ObservableProperty] private bool _isConnected;
    [ObservableProperty] private bool _isServiceOnline;
    [ObservableProperty] private string _statusMessage = "就绪";
    [ObservableProperty] private ObservableCollection<string> _availablePorts = new();
    [ObservableProperty] private string _selectedPort = "COM6";   // ADS1299 2026-05 board (CP210x)
    [ObservableProperty] private int _selectedBaudRate = 230400;   // ADS1299 2026-05 board default
    [ObservableProperty] private List<int> _baudRates = new() { 9600, 57600, 115200, 230400, 460800, 921600 };
    [ObservableProperty] private int _channelCount = 1;            // 2026-05 board is single-channel

    [ObservableProperty] private DeviceType _selectedDeviceType = DeviceType.ADS1299;
    partial void OnSelectedDeviceTypeChanged(DeviceType value) => OnPropertyChanged(nameof(NsmSampleRateVisible));
    public DeviceType[] DeviceTypes { get; } = { DeviceType.ADS1299, DeviceType.NSM };
    [ObservableProperty] private int _nsmSampleRate = 100;
    public List<int> NsmSampleRates { get; } = new() { 100, 200, 250, 500 };
    public Visibility NsmSampleRateVisible => SelectedDeviceType == DeviceType.NSM ? Visibility.Visible : Visibility.Collapsed;

    // ── Pulse ──────────────────────────────────────────────────────────────
    [ObservableProperty] private ObservableCollection<string> _availablePulsePorts = new();
    [ObservableProperty] private string _selectedPulsePort = "COM4";
    [ObservableProperty] private bool _isPulseConnected;
    [ObservableProperty] private string _pulseStatusMessage = "未连接";

    // ── Session ────────────────────────────────────────────────────────────
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _patientId = "";
    [ObservableProperty] private string _surgeryType = "";
    [ObservableProperty] private string _recordingDuration = "00:00:00";

    // ── Theme ──────────────────────────────────────────────────────────────
    [ObservableProperty] private bool _isNightMode;

    // ── Simulation ─────────────────────────────────────────────────────────
    [ObservableProperty] private bool _isSimulating;
    [ObservableProperty] private string _simulationStatus = "";

    public ShellViewModel(
        ISerialPortService serial,
        NSMSerialService nsmSerial,
        IPulseSerialService pulseSerial,
        IEEGProcessingClient processing,
        IRecordingService recording,
        IPlaybackService playback,
        IEventAnnotationService events,
        DataPipeline pipeline,
        IVitalSimulatorService simulator,
        ResultBroker broker,
        SleepStagingViewModel staging,
        SleepStatisticsViewModel statistics,
        SpO2TrendViewModel spo2Trend,
        DrugAdministrationViewModel drugAdmin,
        SedationScaleViewModel sedation,
        PositionTrackingViewModel position,
        ReportViewModel report,
        SimulationViewModel simulation,
        ILogger<ShellViewModel> logger)
    {
        _serial = serial; _nsmSerial = nsmSerial; _pulseSerial = pulseSerial;
        _processing = processing; _recording = recording; _playback = playback;
        _events = events; _pipeline = pipeline; _simulator = simulator;
        _broker = broker; _logger = logger;

        Staging = staging; Statistics = statistics; SpO2Trend = spo2Trend;
        DrugAdmin = drugAdmin; Sedation = sedation; Position = position; Report = report;
        Simulation = simulation;

        // Wire pipeline → broker
        _pipeline.ResultAvailable += result =>
        {
            Application.Current.Dispatcher.Invoke(() => _broker.Publish(result));

            if (_isRecording && _recording.CurrentSession != null)
                _ = _recording.RecordResultAsync(result);
        };

        // Wire broker → sub-VMs
        _broker.Subscribe(result =>
        {
            Staging.OnResultReceived(result);
            SpO2Trend.OnResultReceived(result);

            // Append RawEEG to rolling buffer + throttled OxyPlot update
            if (result.RawEEG is { Length: > 0 } raw)
            {
                foreach (var sample in raw)
                {
                    _eegWaveBuffer[_eegWaveWritePos % 256] = sample;
                    _eegWaveWritePos++;
                }

                var now = DateTime.Now;
                if (now - _lastEegRefresh >= EegRefreshInterval)
                {
                    _lastEegRefresh = now;
                    if (_eegPlotModel?.Series.Count > 0 && _eegPlotModel.Series[0] is OxyPlot.Series.LineSeries series)
                    {
                        series.Points.Clear();
                        int len = Math.Min(_eegWaveWritePos, 256);
                        for (int i = 0; i < len; i++)
                        {
                            int idx = (_eegWaveWritePos - len + i) % 256;
                            series.Points.Add(new OxyPlot.DataPoint(i, _eegWaveBuffer[idx]));
                        }
                        _eegPlotModel.InvalidatePlot(true);
                    }
                }
            }
        });


        // Forward epoch completion from staging to statistics
        Staging.EpochCompleted += epoch =>
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                Statistics.OnEpochCompleted(epoch);
                _broker.PublishEpoch(epoch);
            });
        };

        // NSM device events
        _nsmSerial.ConnectionStatusChanged += msg =>
            Application.Current.Dispatcher.Invoke(() => StatusMessage = msg);
        _nsmSerial.ErrorOccurred += ex =>
            Application.Current.Dispatcher.Invoke(() => StatusMessage = $"NSM 错误: {ex.Message}");
        _nsmSerial.SampleReceived += sample =>
        {
            if (IsConnected && SelectedDeviceType == DeviceType.NSM)
                _pipeline.InjectSample(sample);
        };

        _serial.ConnectionStatusChanged += msg =>
            Application.Current.Dispatcher.Invoke(() => StatusMessage = msg);
        _serial.ErrorOccurred += ex =>
            Application.Current.Dispatcher.Invoke(() => StatusMessage = $"错误: {ex.Message}");

        _pulseSerial.ConnectionStatusChanged += msg =>
            Application.Current.Dispatcher.Invoke(() => PulseStatusMessage = msg);
        _pulseSerial.ErrorOccurred += ex =>
            Application.Current.Dispatcher.Invoke(() => PulseStatusMessage = $"脉搏错误: {ex.Message}");
        _pulseSerial.BpmReceived += OnBpmReceived;

        _simulator.PositionChanged += pos =>
            Application.Current.Dispatcher.Invoke(() => { IsSimulating = true; SimulationStatus = $"模拟中  {pos:hh\\:mm\\:ss}"; });
        _simulator.SimulationCompleted += () =>
            Application.Current.Dispatcher.Invoke(() => { IsSimulating = false; SimulationStatus = "模拟完毕"; });
        _simulator.SimulationStopped += () =>
            Application.Current.Dispatcher.Invoke(() => { IsSimulating = false; SimulationStatus = ""; });

        // Clock
        _clockTimer = new System.Timers.Timer(1000) { AutoReset = true };
        _clockTimer.Elapsed += (_, _) =>
            Application.Current.Dispatcher.Invoke(() =>
            {
                var now = DateTime.Now;
                CurrentTimeDisplay = now.ToString("HH:mm:ss");
                CurrentDateDisplay = now.ToString("yyyy-MM-dd");
            });
        _clockTimer.Start();
        var t0 = DateTime.Now;
        CurrentTimeDisplay = t0.ToString("HH:mm:ss");
        CurrentDateDisplay = t0.ToString("yyyy-MM-dd");

        Simulation.ResetRequested += () =>
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                Staging.Reset();
                Statistics.Reset();
                SpO2Trend.Reset();
                DrugAdmin.Reset();
                Sedation.Reset();
                Position.Reset();
                Array.Clear(_eegWaveBuffer, 0, _eegWaveBuffer.Length);
                _eegWaveWritePos = 0;
                if (_eegPlotModel?.Series.Count > 0 && _eegPlotModel.Series[0] is OxyPlot.Series.LineSeries series)
                {
                    series.Points.Clear();
                    _eegPlotModel.InvalidatePlot(true);
                }
            });
        };

        _eegPlotModel = BuildEegPlotModel();
        RefreshPorts();
        _healthCheckCts = new CancellationTokenSource();
        _ = CheckServiceAsync(_healthCheckCts.Token);
    }

    private void OnBpmReceived(int bpm)
    {
        _pipeline.CurrentHeartRate = bpm > 0 ? (double?)bpm : null;
    }

    // ── Commands ──────────────────────────────────────────────────────────

    [RelayCommand]
    private void RefreshPorts()
    {
        var ports = _serial.GetAvailablePorts().ToList();
        AvailablePorts.Clear();
        foreach (var p in ports) AvailablePorts.Add(p);
        if (AvailablePorts.Count > 0 && !AvailablePorts.Contains(SelectedPort))
            SelectedPort = AvailablePorts[0];

        AvailablePulsePorts.Clear();
        foreach (var p in ports) AvailablePulsePorts.Add(p);
        if (AvailablePulsePorts.Count > 0 && !AvailablePulsePorts.Contains(SelectedPulsePort))
            SelectedPulsePort = AvailablePulsePorts[0];
    }

    [RelayCommand]
    private void Connect()
    {
        if (IsConnected) { DisconnectDevice(); return; }

        if (SelectedDeviceType == DeviceType.NSM)
        {
            var ok = _nsmSerial.Connect(SelectedPort, SelectedBaudRate, 1, NsmSampleRate);
            if (ok)
            {
                IsConnected = true;
                _pipeline.DeviceSampleRate = _nsmSerial.DetectedSampleRate;
                _pipeline.Start();
                StatusMessage = $"NSM 已连接: {SelectedPort} @ {_nsmSerial.DetectedSampleRate}Hz";
            }
            else StatusMessage = "NSM 连接失败";
        }
        else
        {
            // ADS1299 (2026-05 board): ~250 Hz/ch fixed;取 CH1（板子实际 8ch 交织，已 de-interleave）。
            var ok = _serial.Connect(SelectedPort, SelectedBaudRate, channelCount: 1, sampleRate: 250);
            if (ok)
            {
                IsConnected = true;
                _pipeline.DeviceSampleRate = _serial.SampleRate;   // was left at 256 (bug)
                _pipeline.Start();
                StatusMessage = $"已连接: {SelectedPort} @ {_serial.SampleRate}Hz 差分";
            }
            else StatusMessage = "连接失败";
        }
    }

    private void DisconnectDevice()
    {
        _serial.Disconnect(); _nsmSerial.Disconnect();
        IsConnected = false; _pipeline.Stop();
    }

    [RelayCommand]
    private void PulseConnect()
    {
        if (IsPulseConnected) { _pulseSerial.Disconnect(); IsPulseConnected = false; return; }
        var ok = _pulseSerial.Connect(SelectedPulsePort);
        if (ok) IsPulseConnected = true;
    }

    [RelayCommand]
    private async Task ToggleRecording()
    {
        if (IsRecording)
            await StopRecording();
        else
            await StartRecording();
    }

    private async Task StartRecording()
    {
        if (string.IsNullOrWhiteSpace(PatientId))
        {
            MessageBox.Show("请先输入患者ID再开始记录。", "必填",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        _ = _processing.ResetSessionAsync();

        var session = _recording.StartSession(PatientId, SurgeryType, sampleRate: _pipeline.DeviceSampleRate);
        _sessionStart = session.StartTime;
        _sessionElapsedSec = 0;
        IsRecording = true;

        Report.SetPatientInfo(PatientId, SurgeryType);

        _durationTimer = new System.Timers.Timer(1000);
        _durationTimer.Elapsed += (_, _) =>
        {
            _sessionElapsedSec++;
            var ts = TimeSpan.FromSeconds(_sessionElapsedSec);
            Application.Current.Dispatcher.Invoke(() => RecordingDuration = ts.ToString(@"hh\:mm\:ss"));
        };
        _durationTimer.Start();
        _logger.LogInformation("患者 {PatientId} 睡眠记录已开始", PatientId);
    }

    [RelayCommand]
    private async Task StopRecording()
    {
        _durationTimer?.Stop(); _durationTimer?.Dispose(); _durationTimer = null;
        await _recording.StopSessionAsync();
        IsRecording = false; _sessionStart = null;
        _logger.LogInformation("睡眠记录已停止");
    }

    [RelayCommand]
    private void ToggleNightMode()
    {
        IsNightMode = !IsNightMode;
        ApplyTheme(IsNightMode);
    }

    [RelayCommand]
    private void OpenSimulation()
    {
        var dlg = new Views.Dialogs.SimulationDialog(Simulation)
        {
            Owner = Application.Current.MainWindow
        };
        dlg.Show();
    }

    [RelayCommand]
    private void ResetSession()
    {
        Staging.Reset(); Statistics.Reset(); SpO2Trend.Reset();
        DrugAdmin.Reset(); Sedation.Reset(); Position.Reset();
        _sessionElapsedSec = 0; RecordingDuration = "00:00:00";
    }

    [RelayCommand]
    private void OpenPlayback()
    {
        // Sleep-specific playback can reuse the existing PlaybackService
        var sessions = _recording.GetRecordedSessions().ToList();
        if (sessions.Count == 0)
        {
            MessageBox.Show("未找到已记录的会话。", "回放", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        StatusMessage = $"可用会话: {sessions.Count}";
    }

    private async Task CheckServiceAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var ok = await _processing.PingAsync();
                Application.Current.Dispatcher.Invoke(() => IsServiceOnline = ok);
            }
            catch { Application.Current.Dispatcher.Invoke(() => IsServiceOnline = false); }
            await Task.Delay(5000, ct);
        }
    }

    public void ApplyTheme(bool isNightMode)
    {
        var app = Application.Current;
        app.Resources.MergedDictionaries.Clear();

        if (isNightMode)
        {
            var nightDict = new ResourceDictionary
            {
                Source = new Uri("pack://application:,,,/EEGMonitor.AnesthesiaSleep;component/Themes/NightModeColors.xaml")
            };
            app.Resources.MergedDictionaries.Add(nightDict);
        }

        Staging.ApplyTheme(isNightMode);
        SpO2Trend.ApplyTheme(isNightMode);
    }

    public void Cleanup()
    {
        _clockTimer?.Stop(); _clockTimer?.Dispose();
        _healthCheckCts?.Cancel();
    }
}
