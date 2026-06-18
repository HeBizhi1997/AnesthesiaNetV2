using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
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
using OxyPlot.Axes;
using OxyPlot.Series;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Threading;

namespace EEGMonitor.ViewModels;

public partial class MainViewModel : BaseViewModel
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
    private readonly ILogger<MainViewModel> _logger;

    private DateTime? _sessionStart;
    private System.Timers.Timer? _durationTimer;
    private System.Timers.Timer? _clockTimer;
    private CancellationTokenSource? _healthCheckCts;
    private Views.Dialogs.SimulationDialog? _simulationDialog;

    // ── Clock ────────────────────────────────────────────────────────────────
    [ObservableProperty] private string _currentTimeDisplay = "";
    [ObservableProperty] private string _currentDateDisplay = "";

    // ── Connection ──────────────────────────────────────────────────────────
    [ObservableProperty] private bool _isConnected;
    [ObservableProperty] private bool _isServiceOnline;
    [ObservableProperty] private string _statusMessage = "Ready";
    [ObservableProperty] private string _eegDiagnostics = "";
    [ObservableProperty] private ObservableCollection<string> _availablePorts = new();
    [ObservableProperty] private string _selectedPort = "COM6";   // ADS1299 2026-05 board (CP210x)
    [ObservableProperty] private int _selectedBaudRate = 230400;   // ADS1299 2026-05 board default
    [ObservableProperty] private List<int> _baudRates = new() { 9600, 57600, 115200, 230400, 460800, 921600 };
    [ObservableProperty] private int _channelCount = 2;

    // ── Device type selection ──────────────────────────────────────────────────
    [ObservableProperty]
    private DeviceType _selectedDeviceType = DeviceType.ADS1299;

    partial void OnSelectedDeviceTypeChanged(DeviceType value)
    {
        OnPropertyChanged(nameof(NsmSampleRateVisible));
    }
    public DeviceType[] DeviceTypes { get; } = { DeviceType.ADS1299, DeviceType.NSM };
    // NSM(NSA-2000)固定 ~100 Hz(100 样本/包、约 1 包/s)。只保留 100:错误地选 200/250/500
    // 会把频段整体平移 2× 等(历史 bug 根因);连接后仍以 NSMSerialService 的实测包率自动锁定/纠偏。
    [ObservableProperty] private int _nsmSampleRate = 100;
    public List<int> NsmSampleRates { get; } = new() { 100 };
    public Visibility NsmSampleRateVisible => SelectedDeviceType == DeviceType.NSM ? Visibility.Visible : Visibility.Collapsed;

    // ── Pulse sensor connection ──────────────────────────────────────────────
    [ObservableProperty] private ObservableCollection<string> _availablePulsePorts = new();
    [ObservableProperty] private string _selectedPulsePort = "COM4";
    [ObservableProperty] private bool _isPulseConnected;
    [ObservableProperty] private string _pulseStatusMessage = "未连接";

    // ── Session ──────────────────────────────────────────────────────────────
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _patientId = "";
    [ObservableProperty] private string _surgeryType = "";
    [ObservableProperty] private string _recordingDuration = "00:00:00";

    // ── Depth of Anesthesia ──────────────────────────────────────────────────
    [ObservableProperty] private double _bisValue = double.NaN;
    [ObservableProperty] private double _sqiValue = double.NaN;
    [ObservableProperty] private string _bisDisplay = "---";
    [ObservableProperty] private string _sqiDisplay = "---";
    [ObservableProperty] private string _qNoxDisplay = "---";  // placeholder
    [ObservableProperty] private string _spiDisplay = "---";   // placeholder
    [ObservableProperty] private string _bisZoneDisplay = "";
    [ObservableProperty] private string _bisSource = "ML";     // "ML" or "---"

    // ── NSM native indices (device-reported, not ML-inferred) ─────────────────
    [ObservableProperty] private double _nsmCsiValue = double.NaN;
    [ObservableProperty] private string _nsmCsiDisplay = "---";
    [ObservableProperty] private string _nsmCsiZone = "";
    [ObservableProperty] private double _nsmBsValue = double.NaN;
    [ObservableProperty] private string _nsmBsDisplay = "---";
    [ObservableProperty] private double _nsmSqiValue = double.NaN;
    [ObservableProperty] private string _nsmSqiDisplay = "---";
    [ObservableProperty] private double _nsmEmgValue = double.NaN;
    [ObservableProperty] private string _nsmEmgDisplay = "---";
    [ObservableProperty] private double _nsmNoxValue = double.NaN;
    [ObservableProperty] private string _nsmNoxDisplay = "---";
    [ObservableProperty] private string _nsmNoxZone = "";
    [ObservableProperty] private double _nsmDeltaPower;
    [ObservableProperty] private double _nsmThetaPower;
    [ObservableProperty] private double _nsmAlphaPower;
    [ObservableProperty] private double _nsmBetaPower;
    [ObservableProperty] private double _nsmGammaPower;
    [ObservableProperty] private string _nsmStatusDisplay = "---";

    // ── Spectral Entropy ─────────────────────────────────────────────────────
    [ObservableProperty] private double _seValue = double.NaN;   // State Entropy 0-91
    [ObservableProperty] private double _reValue = double.NaN;   // Response Entropy 0-100
    [ObservableProperty] private string _seDisplay = "---";
    [ObservableProperty] private string _reDisplay = "---";
    [ObservableProperty] private string _reSeDiffDisplay = "---"; // RE-SE: EMG indicator

    // ── fNox ─────────────────────────────────────────────────────────────────
    [ObservableProperty] private double _fnoxValue = double.NaN;  // 0-100
    [ObservableProperty] private string _fnoxDisplay = "---";
    [ObservableProperty] private string _fnoxZoneDisplay = "";
    [ObservableProperty] private double _fnoxBarValue = 0.0;
    [ObservableProperty] private bool   _fnoxMRMode = false;

    // ── Vitals ──────────────────────────────────────────────────────────────
    [ObservableProperty] private string _heartRateDisplay = "---";
    [ObservableProperty] private string _spO2Display = "---";
    [ObservableProperty] private string _hrvDisplay = "---";

    // ── Sleep vs anesthesia discrimination ───────────────────────────────────
    [ObservableProperty] private string _saIndicator = "---";
    [ObservableProperty] private string _saDisplay = "";
    [ObservableProperty] private double _spindleDensityValue;
    [ObservableProperty] private bool   _isSleepDetected;

    // ── Signal quality diagnostics ───────────────────────────────────────────
    [ObservableProperty] private string _signalWarning = "";
    [ObservableProperty] private bool   _hasSignalWarning;

    // ── Band Powers ──────────────────────────────────────────────────────────
    [ObservableProperty] private double _deltaPower;
    [ObservableProperty] private double _thetaPower;
    [ObservableProperty] private double _alphaPower;
    [ObservableProperty] private double _betaPower;
    [ObservableProperty] private double _gammaPower;

    // ── Events ──────────────────────────────────────────────────────────────
    [ObservableProperty] private ObservableCollection<ClinicalEvent> _clinicalEvents = new();
    [ObservableProperty] private string _newEventLabel = "";
    [ObservableProperty] private string _selectedEventType = "Induction";
    [ObservableProperty] private List<string> _eventTypeOptions = Enum
        .GetNames<ClinicalEventType>()
        .Where(n => !n.StartsWith("Auto"))
        .ToList();

    // ── OxyPlot Models ───────────────────────────────────────────────────────
    [ObservableProperty] private PlotModel _rawEEGModel = new();
    [ObservableProperty] private PlotModel _deltaModel = new();
    [ObservableProperty] private PlotModel _thetaModel = new();
    [ObservableProperty] private PlotModel _alphaModel = new();
    [ObservableProperty] private PlotModel _betaModel = new();
    [ObservableProperty] private PlotModel _gammaModel = new();
    [ObservableProperty] private PlotModel _dsaModel = new();
    [ObservableProperty] private PlotModel _trendModel = new();       // BIS 5-min history
    [ObservableProperty] private PlotModel _pulseWaveModel = new();
    [ObservableProperty] private PlotModel _hrvModel = new();

    // Simulation status
    [ObservableProperty] private bool _isSimulating;
    [ObservableProperty] private string _simulationStatus = "";

    // Rolling 5-minute BIS + SE + fNox trend
    private readonly Queue<(double time, double bis)>  _bisTrend  = new();
    private readonly Queue<(double time, double se)>   _seTrend   = new();
    private readonly Queue<(double time, double fnox)> _fnoxTrend = new();
    private const double TREND_WINDOW_SEC = 300.0;
    private double _sessionElapsedSec;

    // Rolling wave buffers: 10 seconds × 256 Hz = 2560 points per band
    private const int WAVE_WINDOW_SAMPLES = 256 * 10;
    private readonly Queue<double> _rawBuf   = new();
    private readonly Queue<double> _deltaBuf = new();
    private readonly Queue<double> _thetaBuf = new();
    private readonly Queue<double> _alphaBuf = new();
    private readonly Queue<double> _betaBuf  = new();
    private readonly Queue<double> _gammaBuf = new();
    private readonly Queue<double> _pulseBuf = new();

    public MainViewModel(
        ISerialPortService serial,
        NSMSerialService nsmSerial,
        IPulseSerialService pulseSerial,
        IEEGProcessingClient processing,
        IRecordingService recording,
        IPlaybackService playback,
        IEventAnnotationService events,
        DataPipeline pipeline,
        IVitalSimulatorService simulator,
        ILogger<MainViewModel> logger)
    {
        _serial = serial;
        _nsmSerial = nsmSerial;
        _pulseSerial = pulseSerial;
        _processing = processing;
        _recording = recording;
        _playback = playback;
        _events = events;
        _pipeline = pipeline;
        _simulator = simulator;
        _logger = logger;

        // NSM device events
        _nsmSerial.ConnectionStatusChanged += msg => RunOnUI(() => StatusMessage = msg);
        _nsmSerial.ErrorOccurred += ex => RunOnUI(() => StatusMessage = $"NSM Error: {ex.Message}");
        _nsmSerial.NSMDataReceived += OnNSMDataReceived;
        _nsmSerial.SampleReceived += sample =>
        {
            if (IsConnected && SelectedDeviceType == DeviceType.NSM)
                _pipeline.InjectSample(sample);
        };

        _serial.ConnectionStatusChanged += msg => RunOnUI(() =>
        {
            if (msg.StartsWith("接收中"))
                EegDiagnostics = msg;
            else
                StatusMessage = msg;
        });
        _serial.ErrorOccurred += ex => RunOnUI(() => StatusMessage = $"Error: {ex.Message}");

        _pulseSerial.ConnectionStatusChanged += msg => RunOnUI(() => PulseStatusMessage = msg);
        _pulseSerial.ErrorOccurred += ex => RunOnUI(() => PulseStatusMessage = $"脉搏错误: {ex.Message}");
        _pulseSerial.BpmReceived += OnBpmReceived;
        _pipeline.ResultAvailable += OnResultAvailable;
        _pipeline.RawChunkAvailable += OnRawChunkAvailable;
        _events.EventAdded += ev => RunOnUI(() => ClinicalEvents.Add(ev));

        _simulator.PositionChanged += pos =>
            RunOnUI(() => { IsSimulating = true; SimulationStatus = $"模拟中  {pos:hh\\:mm\\:ss}"; });
        _simulator.SimulationCompleted += () =>
            RunOnUI(() => { IsSimulating = false; SimulationStatus = "模拟完毕"; ClearAllCharts(); });
        _simulator.SimulationStopped += () =>
            RunOnUI(() => { IsSimulating = false; SimulationStatus = ""; ClearAllCharts(); });

        // Real-time clock
        _clockTimer = new System.Timers.Timer(1000) { AutoReset = true };
        _clockTimer.Elapsed += (_, _) => RunOnUI(() =>
        {
            var now = DateTime.Now;
            CurrentTimeDisplay = now.ToString("HH:mm:ss");
            CurrentDateDisplay = now.ToString("yyyy-MM-dd");
        });
        _clockTimer.Start();
        var t0 = DateTime.Now;
        CurrentTimeDisplay = t0.ToString("HH:mm:ss");
        CurrentDateDisplay = t0.ToString("yyyy-MM-dd");

        InitCharts();
        RefreshPorts();
        _healthCheckCts = new CancellationTokenSource();
        _ = CheckServiceAsync(_healthCheckCts.Token);
    }

    // ── Commands ─────────────────────────────────────────────────────────────

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
                // Use auto-detected rate from packet timing (NSM always ~100 Hz).
                // Falls back to configured rate until detection locks in after ~5 packets.
                _pipeline.DeviceSampleRate = _nsmSerial.DetectedSampleRate;
                _pipeline.Start();
                StatusMessage = $"NSM Connected: {SelectedPort} @ {_nsmSerial.DetectedSampleRate}Hz";

                // Sync detected rate once auto-detection locks (within ~5s).
                // The pipeline reads DeviceSampleRate at chunk creation, so updating
                // it after detection fixes any configuration mismatch mid-session.
                _ = Task.Run(async () =>
                {
                    for (int i = 0; i < 15; i++)  // wait up to ~15s for rate lock
                    {
                        await Task.Delay(1000);
                        var detected = _nsmSerial.DetectedSampleRate;
                        if (detected != _pipeline.DeviceSampleRate)
                        {
                            var old = _pipeline.DeviceSampleRate;
                            _pipeline.DeviceSampleRate = detected;
                            RunOnUI(() =>
                            {
                                StatusMessage = $"NSM rate corrected: {old}Hz -> {detected}Hz (auto-detected)";
                            });
                            _logger.LogWarning("NSM rate corrected: {Old}Hz -> {New}Hz", old, detected);
                            break;
                        }
                    }
                });
            }
            else
            {
                StatusMessage = "NSM Connection failed";
            }
        }
        else
        {
            // ADS1299 (2026-05 board): ~250 Hz/ch fixed; keep CH1 (board streams 8ch interleaved).
            var ok = _serial.Connect(SelectedPort, SelectedBaudRate, channelCount: 1, sampleRate: 250);
            if (ok)
            {
                IsConnected = true;
                _pipeline.DeviceSampleRate = _serial.SampleRate;   // was left at 256 (bug)
                _pipeline.Start();
                StatusMessage = $"Connected: {SelectedPort} @ {_serial.SampleRate}Hz";
            }
            else
            {
                StatusMessage = "Connection failed";
            }
        }
    }

    private void DisconnectDevice()
    {
        _serial.Disconnect();
        _nsmSerial.Disconnect();
        IsConnected = false;
        _pipeline.Stop();
    }

    [RelayCommand]
    private void PulseConnect()
    {
        if (IsPulseConnected)
        {
            _pulseSerial.Disconnect();
            IsPulseConnected = false;
            return;
        }
        var ok = _pulseSerial.Connect(SelectedPulsePort);
        if (ok)
        {
            IsPulseConnected = true;
        }
    }

    private void OnBpmReceived(int bpm)
    {
        // Update pipeline so BPM is included in every subsequent EEG chunk sent to Python
        _pipeline.CurrentHeartRate = bpm > 0 ? (double?)bpm : null;

        RunOnUI(() =>
        {
            if (bpm > 0) HeartRateDisplay = bpm.ToString();
        });
    }

    [RelayCommand]
    private void StartRecording()
    {
        if (string.IsNullOrWhiteSpace(PatientId))
        {
            MessageBox.Show("Please enter a Patient ID before recording.", "Required",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        // Reset Python stateful processors so BIS/SE/fNox start fresh for this patient
        _ = _processing.ResetSessionAsync();

        var session = _recording.StartSession(PatientId, SurgeryType, sampleRate: _pipeline.DeviceSampleRate);
        _sessionStart = session.StartTime;
        _sessionElapsedSec = 0;
        IsRecording = true;

        _durationTimer = new System.Timers.Timer(1000);
        _durationTimer.Elapsed += (_, _) =>
        {
            _sessionElapsedSec++;
            var ts = TimeSpan.FromSeconds(_sessionElapsedSec);
            RunOnUI(() => RecordingDuration = ts.ToString(@"hh\:mm\:ss"));
        };
        _durationTimer.Start();
        _logger.LogInformation("Recording started for patient {PatientId}", PatientId);
    }

    [RelayCommand]
    private async Task StopRecording()
    {
        _durationTimer?.Stop(); _durationTimer?.Dispose(); _durationTimer = null;
        await _recording.StopSessionAsync();
        IsRecording = false;
        _sessionStart = null;
        _logger.LogInformation("Recording stopped");
    }

    [RelayCommand]
    private void AddEvent()
    {
        if (string.IsNullOrWhiteSpace(NewEventLabel)) return;
        if (!Enum.TryParse<ClinicalEventType>(SelectedEventType, out var eventType))
            eventType = ClinicalEventType.Custom;

        _events.AddEvent(eventType, NewEventLabel,
            bis: double.IsNaN(BisValue) ? null : BisValue,
            spo2: null,
            sessionStart: _sessionStart ?? DateTime.Now);

        if (IsRecording && _recording.CurrentSession != null)
            _ = _recording.RecordEventAsync(ClinicalEvents[^1]);

        NewEventLabel = "";
    }

    [RelayCommand]
    private void OpenPlayback()
    {
        var dlg = new Views.Dialogs.PlaybackWindow(_playback, _recording.GetRecordedSessions().ToList());
        dlg.ShowDialog();
    }

    [RelayCommand]
    private void OpenSettings()
    {
        var dlg = new Views.Dialogs.SettingsDialog { Owner = Application.Current.MainWindow };
        dlg.ShowDialog();
    }

    [RelayCommand]
    private void OpenSimulation()
    {
        if (_simulationDialog != null && _simulationDialog.IsLoaded)
        {
            _simulationDialog.Activate();
            return;
        }
        _simulationDialog = new Views.Dialogs.SimulationDialog(_simulator)
        {
            Owner = Application.Current.MainWindow,
        };
        IsSimulating = _simulator.IsRunning;
        _simulationDialog.Show();
    }

    /// <summary>Called by MainWindow.OnClosed before host shutdown.</summary>
    public void Cleanup()
    {
        _healthCheckCts?.Cancel();
        _healthCheckCts?.Dispose();
        _healthCheckCts = null;

        _durationTimer?.Stop();
        _durationTimer?.Dispose();
        _durationTimer = null;

        _clockTimer?.Stop();
        _clockTimer?.Dispose();
        _clockTimer = null;

        _simulator.Stop();

        _simulationDialog?.Close();
        _simulationDialog = null;

        if (IsConnected)
        {
            _serial.Disconnect();
            _nsmSerial.Disconnect();
            _pipeline.Stop();
        }

        if (IsPulseConnected)
            _pulseSerial.Disconnect();
    }

    // ── Data handler ─────────────────────────────────────────────────────────

    private void OnRawChunkAvailable(double[] rawData)
    {
        // No longer plots the EEG chart — the EEG trace is driven by the Python service's
        // preprocessed FilteredEEG (see OnResultAvailable) so it matches the band charts and
        // is mains-free + continuous. Kept as a hook in case a fast raw view is wanted later.
    }

    private void OnResultAvailable(ProcessedEEGResult result)
    {
        RunOnUI(() =>
        {
            // Depth of anesthesia (ML-inferred)
            BisValue = result.BIS;
            SqiValue = result.SQI;
            BisDisplay = double.IsNaN(result.BIS) ? "---" : result.BIS.ToString("F0");
            SqiDisplay = double.IsNaN(result.SQI) ? "---" : $"{result.SQI:F0}%";
            BisSource = "ML";
            BisZoneDisplay = double.IsNaN(result.BIS) ? "" : result.BIS switch
            {
                < 40 => "过深麻醉",
                < 60 => "适宜区间",
                < 80 => "偏浅",
                _    => "清醒风险"
            };

            // Signal quality diagnostics
            {
                var warnings = new System.Text.StringBuilder();
                if (result.EegAmplitudeUv > 0 && result.EegAmplitudeUv < 5.0)
                    warnings.Append($"⚠ 信号幅度过低 {result.EegAmplitudeUv:F1}µV (应≥15µV) · 检查电极接触");
                if (result.EegTonalRatio > 0.40)
                {
                    if (warnings.Length > 0) warnings.Append("  |  ");
                    warnings.Append($"⚠ {result.EegDominantHz:F1}Hz 干扰 (占{result.EegTonalRatio*100:F0}%功率) · 非脑电信号");
                }

                // Hardware diagnostics (NSM device)
                if (result.HwIsSaturated)
                {
                    if (warnings.Length > 0) warnings.Append("  |  ");
                    warnings.Append($"⚠ ADC饱和 ({result.HwClippingPct:F0}%削顶 @ ±{result.HwAdcRangeUv:F0}µV) · 信号幅度超出硬件量程，频谱失真");
                }
                else if (result.HwClippingPct > 5.0)
                {
                    if (warnings.Length > 0) warnings.Append("  |  ");
                    warnings.Append($"⚠ 信号近饱和 ({result.HwClippingPct:F0}%近轨) · 可能有削顶失真");
                }

                // DC offset alone is NOT alarmed: on this ADS1299 board the raw electrode
                // half-cell potential routinely sits at thousands of µV, yet the 0.5 Hz
                // highpass removes it entirely and the EEG is unaffected. A DC issue only
                // matters when it drives the input into the ADC rail — which is already
                // reported by the HwIsSaturated alarm above. So no standalone DC alarm.

                // Device/pipeline cross-validation
                if (result.DeviceDeltaDiscrepancy > 0.30)
                {
                    if (warnings.Length > 0) warnings.Append("  |  ");
                    warnings.Append($"⚠ 设备/管线δ偏差 ({result.DeviceDeltaDiscrepancy*100:F0}%) · 管线δ={result.DeltaPower*100:F0}% 设备δ={result.DeviceDeltaRatio*100:F0}%");
                }

                SignalWarning = warnings.ToString();
                HasSignalWarning = warnings.Length > 0;
            }

            // Spectral Entropy
            SeValue = result.StateEntropy ?? double.NaN;
            ReValue = result.ResponseEntropy ?? double.NaN;
            SeDisplay = result.StateEntropy.HasValue ? result.StateEntropy.Value.ToString("F0") : "---";
            ReDisplay = result.ResponseEntropy.HasValue ? result.ResponseEntropy.Value.ToString("F0") : "---";
            if (result.StateEntropy.HasValue && result.ResponseEntropy.HasValue)
            {
                var diff = result.ResponseEntropy.Value - result.StateEntropy.Value;
                ReSeDiffDisplay = $"{diff:F0}";
            }
            else
            {
                ReSeDiffDisplay = "---";
            }

            // fNox
            FnoxValue    = result.FNox ?? double.NaN;
            FnoxBarValue = double.IsNaN(FnoxValue) ? 0 : FnoxValue;
            FnoxDisplay  = result.FNox.HasValue ? result.FNox.Value.ToString("F0") : "---";
            FnoxMRMode   = result.FNoxMRMode ?? false;
            FnoxZoneDisplay = result.FNox switch
            {
                null               => "",
                { } v when v < 30  => "镇痛充分",
                { } v when v <= 50 => "靶区",
                { } v when v < 65  => "关注",
                _                  => "镇痛不足!",
            };

            // Vitals
            if (result.HeartRate.HasValue) HeartRateDisplay = $"{result.HeartRate:F0}";
            if (result.SpO2.HasValue) SpO2Display = $"{result.SpO2:F1}%";
            if (result.HRV_RMSSD.HasValue) HrvDisplay = $"{result.HRV_RMSSD:F1}ms";

            // Sleep vs anesthesia
            SpindleDensityValue = result.SpindleDensity;
            IsSleepDetected = result.IsLikelySleep;
            if (result.IsLikelySleep)
            {
                SaIndicator = "睡眠";
                SaDisplay = $"{result.SpindleDensity:F1} spindles/min";
            }
            else if (result.TotalSpindleCount >= 3)
            {
                SaIndicator = "可能睡眠";
                SaDisplay = $"{result.SpindleDensity:F1}/min ({result.TotalSpindleCount} total)";
            }
            else
            {
                SaIndicator = "---";
                SaDisplay = result.TotalSpindleCount > 0
                    ? $"{result.TotalSpindleCount} spindles" : "";
            }

            // Band powers
            DeltaPower = result.DeltaPower * 100;
            ThetaPower = result.ThetaPower * 100;
            AlphaPower = result.AlphaPower * 100;
            BetaPower = result.BetaPower * 100;
            GammaPower = result.GammaPower * 100;

            // EEG waveform = Python service's preprocessed signal (causal continuous filter,
            // mains-notched), so the EEG trace matches the band charts and is artefact-free.
            AppendWaveChart(RawEEGModel, _rawBuf, result.FilteredEEG);
            AppendWaveChart(DeltaModel,  _deltaBuf, result.DeltaWave);
            AppendWaveChart(ThetaModel,  _thetaBuf, result.ThetaWave);
            AppendWaveChart(AlphaModel,  _alphaBuf, result.AlphaWave);
            AppendWaveChart(BetaModel,   _betaBuf,  result.BetaWave);
            AppendWaveChart(GammaModel,  _gammaBuf, result.GammaWave);
            UpdateDSA(result);
            UpdateTrend(result);
            UpdatePulseWave(result);

            // Auto alerts
            if (_sessionStart.HasValue)
                _events.AutoCheckThresholds(
                    result.BIS,
                    result.SpO2 ?? double.NaN,
                    result.SQI,
                    _sessionStart.Value);
        });
    }

    private void OnNSMDataReceived(NSMDataPacket pkt)
    {
        RunOnUI(() =>
        {
            // ── NSM native CSI (device-reported anesthesia depth) ──────────
            if (pkt.CSIValid && pkt.CSI <= 99)
            {
                NsmCsiValue = pkt.CSI;
                NsmCsiDisplay = pkt.CSI.ToString("F0");
                NsmCsiZone = pkt.CSI switch
                {
                    < 40 => "过深麻醉",
                    < 60 => "适宜区间",
                    < 80 => "偏浅",
                    _    => "清醒风险"
                };
            }
            else
            {
                NsmCsiDisplay = "---";
                NsmCsiZone = "";
            }

            // ── NSM burst suppression ──────────────────────────────────────
            if (pkt.BSValid && pkt.BS <= 100)
            {
                NsmBsValue = pkt.BS;
                NsmBsDisplay = $"{pkt.BS}%";
            }
            else
                NsmBsDisplay = "---";

            // ── NSM signal quality ─────────────────────────────────────────
            if (pkt.SQIValid && pkt.SQI <= 100)
            {
                NsmSqiValue = pkt.SQI;
                NsmSqiDisplay = $"{pkt.SQI}%";
            }
            else
                NsmSqiDisplay = "---";

            // ── NSM EMG ────────────────────────────────────────────────────
            if (pkt.EMGValid && pkt.EMG <= 100)
            {
                NsmEmgValue = pkt.EMG;
                NsmEmgDisplay = pkt.EMG.ToString("F0");
            }
            else
                NsmEmgDisplay = "---";

            // ── NSM NOX (nociception) ──────────────────────────────────────
            if (pkt.NOXValid && pkt.NOX <= 99)
            {
                NsmNoxValue = pkt.NOX;
                NsmNoxDisplay = pkt.NOX.ToString("F0");
                NsmNoxZone = pkt.NOX switch
                {
                    < 30  => "镇痛充分",
                    <= 50 => "靶区",
                    < 65  => "关注",
                    _     => "镇痛不足!",
                };
            }
            else
            {
                NsmNoxDisplay = "---";
                NsmNoxZone = "";
            }

            // ── NSM band powers ────────────────────────────────────────────
            // 显示用「相对%」(dB→线性→归一,口径同厂家界面);0–100 进度条/文本因此可比。
            // 原始 dB 仍单独传给 Python 做交叉校验(那里按 dB 处理)。
            NsmDeltaPower = pkt.DeltaPct;
            NsmThetaPower = pkt.ThetaPct;
            NsmAlphaPower = pkt.AlphaPct;
            NsmBetaPower  = pkt.BetaPct;
            NsmGammaPower = pkt.GammaPct;

            // Feed RAW device band powers (dB) to the pipeline for cross-validation
            if (IsRecording || IsConnected)
            {
                _pipeline.SetDeviceBandPowers(new Dictionary<string, int>
                {
                    ["delta"] = pkt.DeltaPowerDb,
                    ["theta"] = pkt.ThetaPowerDb,
                    ["alpha"] = pkt.AlphaPowerDb,
                    ["beta"]  = pkt.BetaPowerDb,
                    ["gamma"] = pkt.GammaPowerDb,
                });
            }

            // ── NSM connection status ──────────────────────────────────────
            var sb = new System.Text.StringBuilder();
            sb.Append($"CSI:{NsmCsiDisplay}");
            if (pkt.SEF95 > 0) sb.Append($"  SEF:{pkt.SEF95}Hz");
            NsmStatusDisplay = sb.ToString();

            // ── Signal quality warnings ────────────────────────────────────
            if (pkt.ElectrodeAlarm || pkt.ElectrodeInvalid)
            {
                var w = new System.Text.StringBuilder();
                if (pkt.ElectrodeAlarm)   w.Append("NSM ⚠ 电极脱落");
                if (pkt.ImpedanceHigh)    w.Append("  |  ⚠ 阻抗偏高");
                if (pkt.ElectrodeInvalid) w.Append("  |  ⚠ 电极失效");
                SignalWarning = w.ToString();
                HasSignalWarning = true;
            }
            else if (pkt.BlackImpedance >= 15 || pkt.WhiteImpedance >= 15)
            {
                SignalWarning = $"NSM ⚠ 阻抗过高 (Blk:{pkt.BlackImpedance} Wht:{pkt.WhiteImpedance})";
                HasSignalWarning = true;
            }
            else if (!HasSignalWarning)
            {
                SignalWarning = "";
            }

            // ── Clinical event from NSM ────────────────────────────────────
            if (pkt.EventNumber > 0)
            {
                var eventType = pkt.EventType switch
                {
                    NSMEventType.Induction      => ClinicalEventType.Induction,
                    NSMEventType.Intubation     => ClinicalEventType.Intubation,
                    NSMEventType.Maintenance    => ClinicalEventType.Custom,
                    NSMEventType.Surgery        => ClinicalEventType.Incision,
                    NSMEventType.Injection      => ClinicalEventType.DrugAdministration,
                    NSMEventType.EndMaintenance => ClinicalEventType.Emergence,
                    _                           => ClinicalEventType.Custom,
                };
                _events.AddEvent(eventType,
                    label: $"NSM: {pkt.EventType} (#{pkt.EventNumber})",
                    bis: pkt.CSIValid ? pkt.CSI : null,
                    sqi: pkt.SQIValid ? pkt.SQI : null);
            }
        });
    }

    // ── Chart initializers ───────────────────────────────────────────────────

    private void InitCharts()
    {
        RawEEGModel = MakeWaveModel("", OxyColor.FromRgb(0x5B, 0x8E, 0xFF), yMin: -200, yMax: 200);
        // New reference palette: γ=red β=orange α=green θ=cyan δ=indigo-blue
        // Fixed Y-axis (µV): amplitude change visible across anesthesia states
        DeltaModel = MakeWaveModel("", OxyColor.FromRgb(0x5B, 0x8E, 0xFF), yMin: -120, yMax: 120); // indigo
        ThetaModel = MakeWaveModel("", OxyColor.FromRgb(0x00, 0xC8, 0xFF), yMin: -60,  yMax: 60);  // cyan
        AlphaModel = MakeWaveModel("", OxyColor.FromRgb(0x00, 0xE6, 0x76), yMin: -60,  yMax: 60);  // green
        BetaModel  = MakeWaveModel("", OxyColor.FromRgb(0xFF, 0x95, 0x00), yMin: -40,  yMax: 40);  // orange
        GammaModel = MakeWaveModel("", OxyColor.FromRgb(0xFF, 0x45, 0x60), yMin: -25,  yMax: 25);  // coral

        InitDSAModel();
        InitTrendModel();
        InitPulseModel();
    }

    private static PlotModel MakeWaveModel(string title, OxyColor color,
        double yMin = double.NaN, double yMax = double.NaN, string yUnit = "µV")
    {
        var m = new PlotModel
        {
            Background = OxyColor.FromRgb(0x09, 0x10, 0x1C),
            PlotAreaBackground = OxyColor.FromRgb(0x05, 0x0B, 0x14),
            Title = title,
            TitleFontSize = 0,   // labels come from XAML overlays
        };
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom,
            IsAxisVisible = false,
            Minimum = 0, Maximum = WAVE_WINDOW_SAMPLES,
            IsZoomEnabled = false, IsPanEnabled = false,
        });
        var yAxis = new LinearAxis
        {
            Position = AxisPosition.Left,
            IsAxisVisible = true,
            TickStyle = TickStyle.Inside,
            AxislineColor = OxyColor.FromRgb(0x30, 0x36, 0x3D),
            TextColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
            FontSize = 7,
            Title = yUnit,
            TitleFontSize = 7,
            TitleColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
            IsZoomEnabled = false, IsPanEnabled = false,
        };
        // Initial range only — NO AbsoluteMinimum/Maximum: the Y axis auto-scales to the
        // signal at runtime (AutoScaleY in AppendWaveChart). Hard caps caused off-screen
        // traces when the device amplitude/units differ from the assumed µV range.
        if (!double.IsNaN(yMin)) yAxis.Minimum = yMin;
        if (!double.IsNaN(yMax)) yAxis.Maximum = yMax;
        m.Axes.Add(yAxis);
        m.Series.Add(new LineSeries
        {
            Color = color,
            StrokeThickness = 1.2,
            LineStyle = LineStyle.Solid,
        });
        return m;
    }

    private void InitDSAModel()
    {
        var muted = OxyColor.FromRgb(0x2A, 0x42, 0x60);
        DsaModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x09, 0x10, 0x1C),
            PlotAreaBackground = OxyColors.Black,
            TitleFontSize = 0,
        };
        DsaModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left,
            Title = "Hz", Minimum = 0, Maximum = 40,
            TitleFontSize = 8, FontSize = 8,
            TextColor = muted, AxislineColor = muted, TicklineColor = muted, TitleColor = muted,
        });
        DsaModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom,
            TitleFontSize = 0, FontSize = 7,
            TextColor = muted, AxislineColor = muted, TicklineColor = muted,
        });
        DsaModel.Axes.Add(new LinearColorAxis
        {
            Key = "color",
            Position = AxisPosition.Right,
            Palette = OxyPalettes.Jet(512),
            TitleFontSize = 0, FontSize = 0,
            IsAxisVisible = false,
        });
        DsaModel.Series.Add(new HeatMapSeries
        {
            ColorAxisKey = "color",
            RenderMethod = HeatMapRenderMethod.Bitmap,
            Interpolate = false,
            X0 = 0, X1 = 60,
            Y0 = 0, Y1 = 40,
        });
    }

    private void InitTrendModel()
    {
        var muted = OxyColor.FromRgb(0x2A, 0x42, 0x60);
        TrendModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x09, 0x10, 0x1C),
            PlotAreaBackground = OxyColor.FromRgb(0x05, 0x0B, 0x14),
            TitleFontSize = 0,
        };
        TrendModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom,
            Minimum = -TREND_WINDOW_SEC, Maximum = 0,
            TextColor = muted, AxislineColor = muted, TicklineColor = muted,
            FontSize = 8,
        });
        TrendModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left,
            Minimum = 0, Maximum = 100, MajorStep = 20,
            TextColor = muted, AxislineColor = muted, TicklineColor = muted,
            FontSize = 8,
        });
        // Safe-zone band (40–60)
        TrendModel.Annotations.Add(new OxyPlot.Annotations.RectangleAnnotation
        {
            MinimumY = 40, MaximumY = 60,
            MinimumX = -TREND_WINDOW_SEC, MaximumX = 0,
            Fill = OxyColor.FromAColor(25, OxyColor.FromRgb(0x00, 0xE6, 0x76)),
            Layer = OxyPlot.Annotations.AnnotationLayer.BelowSeries,
        });
        // BIS series (cyan)
        TrendModel.Series.Add(new LineSeries
        {
            Title = "BIS",
            Color = OxyColor.FromRgb(0x00, 0xC8, 0xFF),
            StrokeThickness = 2,
        });
        // SE series (amber dashed)
        TrendModel.Series.Add(new LineSeries
        {
            Title = "SE",
            Color = OxyColor.FromRgb(0xF0, 0xA0, 0x20),
            StrokeThickness = 1.5,
            LineStyle = LineStyle.Dash,
        });
        // fNox series (magenta dotted)
        TrendModel.Series.Add(new LineSeries
        {
            Title = "fNox",
            Color = OxyColor.FromRgb(0xFF, 0x45, 0xA0),
            StrokeThickness = 1.5,
            LineStyle = LineStyle.Dot,
        });
    }

    private void InitPulseModel()
    {
        PulseWaveModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x09, 0x10, 0x1C),
            PlotAreaBackground = OxyColor.FromRgb(0x05, 0x0B, 0x14),
            TitleFontSize = 0,
        };
        PulseWaveModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false, IsZoomEnabled = false, IsPanEnabled = false });
        PulseWaveModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left,   IsAxisVisible = false, IsZoomEnabled = false, IsPanEnabled = false });
        PulseWaveModel.Series.Add(new LineSeries
        {
            Color = OxyColor.FromRgb(0xFF, 0x45, 0x60),
            StrokeThickness = 1.5,
        });

        HrvModel = new PlotModel { TitleFontSize = 0 };
        HrvModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        HrvModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left,   IsAxisVisible = false });
        HrvModel.Series.Add(new StemSeries
        {
            Color = OxyColor.FromRgb(0x00, 0xE6, 0x76),
            StrokeThickness = 1.5,
        });
    }

    // ── Chart updaters ───────────────────────────────────────────────────────

    private static void AppendWaveChart(PlotModel model, Queue<double> buf, double[] newData)
    {
        if (newData.Length == 0) return;
        foreach (var v in newData) buf.Enqueue(v);
        while (buf.Count > WAVE_WINDOW_SAMPLES) buf.Dequeue();

        var series = (LineSeries)model.Series[0];
        // Incremental update: reuse existing points, only rebuild when buffer count changes
        var bufList = buf.ToList();
        if (series.Points.Count == bufList.Count)
        {
            // Same length — update Y values in-place (avoids allocation)
            for (int i = 0; i < bufList.Count; i++)
            {
                var pt = series.Points[i];
                series.Points[i] = new DataPoint(pt.X, bufList[i]);
            }
        }
        else
        {
            series.Points.Clear();
            for (int i = 0; i < bufList.Count; i++)
                series.Points.Add(new DataPoint(i, bufList[i]));
        }

        bool axisChanged = AutoScaleY(model, bufList);   // adapt Y range to the signal
        // When the axis range moved, force a full invalidate so OxyPlot re-resolves the
        // axis transform; otherwise a data-only invalidate can leave the trace off-screen.
        model.InvalidatePlot(axisChanged);
    }

    /// <summary>
    /// Robust, EMA-smoothed Y-axis auto-scaling for the waveform charts.
    ///
    /// Centers on the median (handles large DC offsets, e.g. raw ADS1299 counts) and
    /// scales to the 2nd–98th percentile span (rejects transient spikes/artefacts) with
    /// 30% headroom. The axis is eased toward the target (α=0.2) so it stays readable
    /// instead of jittering every update; it snaps immediately when the trace is fully
    /// off the current view (e.g. on connect or a unit/scale change).
    /// </summary>
    /// <returns>true if the Y-axis Min/Max changed materially (caller should full-invalidate).</returns>
    private static bool AutoScaleY(PlotModel model, IReadOnlyList<double> data)
    {
        if (data.Count < 8) return false;
        if (model.Axes.FirstOrDefault(a => a.Position == AxisPosition.Left) is not LinearAxis y)
            return false;

        var s = data.ToArray();
        Array.Sort(s);
        double med = s[s.Length / 2];
        // 0.5th / 99.5th percentile: keep ~99% of the waveform, reject only true
        // artefacts/spikes; 25% headroom so normal peaks never touch the frame edge.
        double lo  = s[(int)(s.Length * 0.005)];
        double hi  = s[Math.Min(s.Length - 1, (int)(s.Length * 0.995))];
        double half = Math.Max(Math.Max(hi - med, med - lo), 1e-6) * 1.25;
        double tMin = med - half, tMax = med + half;

        double curMin = y.Minimum, curMax = y.Maximum;
        bool needSnap = double.IsNaN(curMin) || double.IsNaN(curMax)
                        || hi > curMax || lo < curMin                    // trace clipped
                        || (curMax - curMin) > 6.0 * half                // view far too wide
                        || (curMax - curMin) < 0.5 * half;               // view far too tight
        double a = needSnap ? 1.0 : 0.2;
        double newMin = double.IsNaN(curMin) ? tMin : curMin + (tMin - curMin) * a;
        double newMax = double.IsNaN(curMax) ? tMax : curMax + (tMax - curMax) * a;
        double span = Math.Max(newMax - newMin, 1e-9);
        bool changed = double.IsNaN(curMin) || double.IsNaN(curMax)
                       || Math.Abs(newMin - curMin) > 0.01 * span
                       || Math.Abs(newMax - curMax) > 0.01 * span;
        y.Minimum = newMin;
        y.Maximum = newMax;
        return changed;
    }

    public void ClearAllCharts()
    {
        _rawBuf.Clear(); _deltaBuf.Clear(); _thetaBuf.Clear();
        _alphaBuf.Clear(); _betaBuf.Clear(); _gammaBuf.Clear();
        _pulseBuf.Clear();

        foreach (var m in new[] { RawEEGModel, DeltaModel, ThetaModel, AlphaModel, BetaModel, GammaModel, PulseWaveModel })
        {
            ((LineSeries)m.Series[0]).Points.Clear();
            m.InvalidatePlot(false);
        }

        _dsaHistory.Clear();
        ((HeatMapSeries)DsaModel.Series[0]).Data = new double[0, 0];
        DsaModel.InvalidatePlot(false);

        _bisTrend.Clear(); _seTrend.Clear(); _fnoxTrend.Clear();
        _sessionElapsedSec = 0;
        ((LineSeries)TrendModel.Series[0]).Points.Clear();
        ((LineSeries)TrendModel.Series[1]).Points.Clear();
        ((LineSeries)TrendModel.Series[2]).Points.Clear();
        TrendModel.InvalidatePlot(false);

        ((StemSeries)HrvModel.Series[0]).Points.Clear();
        HrvModel.InvalidatePlot(false);

        BisDisplay = "---"; BisValue = double.NaN; BisZoneDisplay = "";
        SqiDisplay = "---"; SqiValue = double.NaN;
        SeDisplay = "---"; SeValue = double.NaN;
        ReDisplay = "---"; ReValue = double.NaN;
        ReSeDiffDisplay = "---";
        FnoxDisplay = "---"; FnoxValue = double.NaN; FnoxZoneDisplay = ""; FnoxBarValue = 0; FnoxMRMode = false;
        HeartRateDisplay = "---"; SpO2Display = "---"; HrvDisplay = "---";
        DeltaPower = ThetaPower = AlphaPower = BetaPower = GammaPower = 0;
    }

    private readonly Queue<(double[] freqs, double[] times, double[,] mat)> _dsaHistory = new();
    private const int DSA_HISTORY_SEC = 60;

    private void UpdateDSA(ProcessedEEGResult result)
    {
        if (result.DSAMatrix.GetLength(0) == 0) return;
        _dsaHistory.Enqueue((result.DSAFrequencies, result.DSATimes, result.DSAMatrix));
        if (_dsaHistory.Count > DSA_HISTORY_SEC) _dsaHistory.Dequeue();

        var series = (HeatMapSeries)DsaModel.Series[0];
        // Combine DSA history into one matrix (freq × time)
        var allCols = _dsaHistory.SelectMany(h => Enumerable.Range(0, h.times.Length)).Count();
        var freqBins = result.DSAFrequencies.Length;
        var combined = new double[freqBins, allCols];
        int col = 0;
        foreach (var (_, times, mat) in _dsaHistory)
        {
            for (int tc = 0; tc < times.Length; tc++, col++)
                for (int f = 0; f < freqBins; f++)
                    combined[f, col] = mat[f, tc];
        }
        series.Data = combined;
        series.X0 = 0; series.X1 = col;
        series.Y0 = result.DSAFrequencies[0]; series.Y1 = result.DSAFrequencies[^1];
        DsaModel.InvalidatePlot(false);
    }

    private void UpdateTrend(ProcessedEEGResult result)
    {
        if (double.IsNaN(result.BIS)) return;
        _sessionElapsedSec++;

        _bisTrend.Enqueue((_sessionElapsedSec, result.BIS));
        while (_bisTrend.Count > 0 && _sessionElapsedSec - _bisTrend.Peek().time > TREND_WINDOW_SEC)
            _bisTrend.Dequeue();

        if (result.StateEntropy.HasValue)
        {
            _seTrend.Enqueue((_sessionElapsedSec, result.StateEntropy.Value));
            while (_seTrend.Count > 0 && _sessionElapsedSec - _seTrend.Peek().time > TREND_WINDOW_SEC)
                _seTrend.Dequeue();
        }

        if (result.FNox.HasValue)
        {
            _fnoxTrend.Enqueue((_sessionElapsedSec, result.FNox.Value));
            while (_fnoxTrend.Count > 0 && _sessionElapsedSec - _fnoxTrend.Peek().time > TREND_WINDOW_SEC)
                _fnoxTrend.Dequeue();
        }

        var bisSeries = (LineSeries)TrendModel.Series[0];
        bisSeries.Points.Clear();
        foreach (var (t, b) in _bisTrend)
            bisSeries.Points.Add(new DataPoint(t - _sessionElapsedSec, b));

        var seSeries = (LineSeries)TrendModel.Series[1];
        seSeries.Points.Clear();
        foreach (var (t, s) in _seTrend)
            seSeries.Points.Add(new DataPoint(t - _sessionElapsedSec, s));

        var fnoxSeries = (LineSeries)TrendModel.Series[2];
        fnoxSeries.Points.Clear();
        foreach (var (t, n) in _fnoxTrend)
            fnoxSeries.Points.Add(new DataPoint(t - _sessionElapsedSec, n));

        TrendModel.InvalidatePlot(false);
    }

    private void UpdatePulseWave(ProcessedEEGResult result)
    {
        if (result.PulseWave.Length == 0) return;
        AppendWaveChart(PulseWaveModel, _pulseBuf, result.PulseWave);
    }

    private async Task CheckServiceAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            IsServiceOnline = await _processing.PingAsync();
            try { await Task.Delay(5000, ct); }
            catch (OperationCanceledException) { break; }
        }
    }
}
