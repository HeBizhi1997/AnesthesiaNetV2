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

namespace EEGMonitor.ViewModels;

public partial class MainViewModel : BaseViewModel
{
    private readonly ISerialPortService _serial;
    private readonly IEEGProcessingClient _processing;
    private readonly IRecordingService _recording;
    private readonly IPlaybackService _playback;
    private readonly IEventAnnotationService _events;
    private readonly DataPipeline _pipeline;
    private readonly IVitalSimulatorService _simulator;
    private readonly ILogger<MainViewModel> _logger;

    private DateTime? _sessionStart;
    private System.Timers.Timer? _durationTimer;
    private Views.Dialogs.SimulationDialog? _simulationDialog;

    // ── Connection ──────────────────────────────────────────────────────────
    [ObservableProperty] private bool _isConnected;
    [ObservableProperty] private bool _isServiceOnline;
    [ObservableProperty] private string _statusMessage = "Ready";
    [ObservableProperty] private ObservableCollection<string> _availablePorts = new();
    [ObservableProperty] private string _selectedPort = "COM3";
    [ObservableProperty] private int _selectedBaudRate = 115200;
    [ObservableProperty] private List<int> _baudRates = new() { 9600, 57600, 115200, 230400, 460800, 921600 };
    [ObservableProperty] private int _channelCount = 4;

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

    // ── Vitals ──────────────────────────────────────────────────────────────
    [ObservableProperty] private string _heartRateDisplay = "---";
    [ObservableProperty] private string _spO2Display = "---";
    [ObservableProperty] private string _hrvDisplay = "---";

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

    // Rolling 5-minute BIS trend
    private readonly Queue<(double time, double bis)> _bisTrend = new();
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
        IEEGProcessingClient processing,
        IRecordingService recording,
        IPlaybackService playback,
        IEventAnnotationService events,
        DataPipeline pipeline,
        IVitalSimulatorService simulator,
        ILogger<MainViewModel> logger)
    {
        _serial = serial;
        _processing = processing;
        _recording = recording;
        _playback = playback;
        _events = events;
        _pipeline = pipeline;
        _simulator = simulator;
        _logger = logger;

        _serial.ConnectionStatusChanged += msg => RunOnUI(() => StatusMessage = msg);
        _serial.ErrorOccurred += ex => RunOnUI(() => StatusMessage = $"Error: {ex.Message}");
        _pipeline.ResultAvailable += OnResultAvailable;
        _events.EventAdded += ev => RunOnUI(() => ClinicalEvents.Add(ev));

        _simulator.PositionChanged += pos =>
            RunOnUI(() => { IsSimulating = true; SimulationStatus = $"模拟中  {pos:hh\\:mm\\:ss}"; });
        _simulator.SimulationCompleted += () =>
            RunOnUI(() => { IsSimulating = false; SimulationStatus = "模拟完毕"; ClearAllCharts(); });
        _simulator.SimulationStopped += () =>
            RunOnUI(() => { IsSimulating = false; SimulationStatus = ""; ClearAllCharts(); });

        InitCharts();
        RefreshPorts();
        _ = CheckServiceAsync();
    }

    // ── Commands ─────────────────────────────────────────────────────────────

    [RelayCommand]
    private void RefreshPorts()
    {
        AvailablePorts.Clear();
        foreach (var p in _serial.GetAvailablePorts()) AvailablePorts.Add(p);
        if (AvailablePorts.Count > 0 && !AvailablePorts.Contains(SelectedPort))
            SelectedPort = AvailablePorts[0];
    }

    [RelayCommand]
    private void Connect()
    {
        if (IsConnected) { _serial.Disconnect(); IsConnected = false; _pipeline.Stop(); return; }
        var ok = _serial.Connect(SelectedPort, SelectedBaudRate, ChannelCount);
        if (ok)
        {
            IsConnected = true;
            _pipeline.Start();
            StatusMessage = $"Connected: {SelectedPort}";
        }
        else
        {
            StatusMessage = "Connection failed";
        }
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
        var session = _recording.StartSession(PatientId, SurgeryType);
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
        _durationTimer?.Stop();
        _durationTimer?.Dispose();
        _durationTimer = null;

        _simulator.Stop();

        _simulationDialog?.Close();
        _simulationDialog = null;

        if (IsConnected)
        {
            _serial.Disconnect();
            _pipeline.Stop();
        }
    }

    // ── Data handler ─────────────────────────────────────────────────────────

    private void OnResultAvailable(ProcessedEEGResult result)
    {
        RunOnUI(() =>
        {
            // Depth of anesthesia
            BisValue = result.BIS;
            SqiValue = result.SQI;
            BisDisplay = double.IsNaN(result.BIS) ? "---" : result.BIS.ToString("F0");
            SqiDisplay = double.IsNaN(result.SQI) ? "---" : $"{result.SQI:F0}%";
            BisZoneDisplay = double.IsNaN(result.BIS) ? "" : result.BIS switch
            {
                < 40 => "过深麻醉",
                < 60 => "适宜区间",
                < 80 => "偏浅",
                _    => "清醒风险"
            };

            // Vitals
            if (result.HeartRate.HasValue) HeartRateDisplay = $"{result.HeartRate:F0}";
            if (result.SpO2.HasValue) SpO2Display = $"{result.SpO2:F1}%";
            if (result.HRV_RMSSD.HasValue) HrvDisplay = $"{result.HRV_RMSSD:F1}ms";

            // Band powers
            DeltaPower = result.DeltaPower * 100;
            ThetaPower = result.ThetaPower * 100;
            AlphaPower = result.AlphaPower * 100;
            BetaPower = result.BetaPower * 100;
            GammaPower = result.GammaPower * 100;

            // Charts (Raw EEG not displayed – skip its buffer to save CPU)
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

    // ── Chart initializers ───────────────────────────────────────────────────

    private void InitCharts()
    {
        RawEEGModel = MakeWaveModel("Raw EEG", OxyColor.FromRgb(0x8B, 0x94, 0x9E));
        // Fixed Y-axis (µV) per band: amplitude changes indicate anesthesia depth
        DeltaModel = MakeWaveModel("δ 0.5-4 Hz", OxyColor.FromRgb(0x4F, 0xC3, 0xF7), yMin: -120, yMax: 120);
        ThetaModel = MakeWaveModel("θ 4-7 Hz",   OxyColor.FromRgb(0x81, 0xC7, 0x84), yMin: -60,  yMax: 60);
        AlphaModel = MakeWaveModel("α 8-13 Hz",  OxyColor.FromRgb(0xFF, 0xB7, 0x4D), yMin: -60,  yMax: 60);
        BetaModel  = MakeWaveModel("β 13-30 Hz", OxyColor.FromRgb(0xF0, 0x62, 0x92), yMin: -40,  yMax: 40);
        GammaModel = MakeWaveModel("γ 30-47 Hz", OxyColor.FromRgb(0xCE, 0x93, 0xD8), yMin: -25,  yMax: 25);

        InitDSAModel();
        InitTrendModel();
        InitPulseModel();
    }

    private static PlotModel MakeWaveModel(string title, OxyColor color,
        double yMin = double.NaN, double yMax = double.NaN, string yUnit = "µV")
    {
        var m = new PlotModel
        {
            Background = OxyColor.FromRgb(0x21, 0x26, 0x2D),
            PlotAreaBackground = OxyColor.FromRgb(0x0D, 0x11, 0x17),
            Title = title,
            TitleFontSize = 10,
            TitleColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
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
        if (!double.IsNaN(yMin)) { yAxis.Minimum = yMin; yAxis.AbsoluteMinimum = yMin; }
        if (!double.IsNaN(yMax)) { yAxis.Maximum = yMax; yAxis.AbsoluteMaximum = yMax; }
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
        DsaModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x21, 0x26, 0x2D),
            PlotAreaBackground = OxyColors.Black,
            Title = "DSA (Density Spectral Array)",
            TitleFontSize = 10,
            TitleColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
        };
        DsaModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left,
            Title = "Freq (Hz)",
            Minimum = 0, Maximum = 40,
            TitleFontSize = 9,
            TextColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
        });
        DsaModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom,
            Title = "Time (s)",
            TitleFontSize = 9,
            TextColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
        });
        DsaModel.Axes.Add(new LinearColorAxis
        {
            Key = "color",
            Position = AxisPosition.Right,
            Palette = OxyPalettes.Jet(512),
            Title = "dB",
            TitleFontSize = 8,
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
        TrendModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x21, 0x26, 0x2D),
            PlotAreaBackground = OxyColor.FromRgb(0x0D, 0x11, 0x17),
            Title = "BIS Trend (5 min)",
            TitleFontSize = 10,
            TitleColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
        };
        TrendModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom,
            Title = "Time (s)",
            Minimum = -TREND_WINDOW_SEC, Maximum = 0,
            TitleFontSize = 9,
            TextColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
        });
        TrendModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left,
            Title = "BIS",
            Minimum = 0, Maximum = 100,
            MajorStep = 20,
            TitleFontSize = 9,
            TextColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
        });
        // BIS safe zone annotation (40-60)
        TrendModel.Annotations.Add(new OxyPlot.Annotations.RectangleAnnotation
        {
            MinimumY = 40, MaximumY = 60,
            MinimumX = -TREND_WINDOW_SEC, MaximumX = 0,
            Fill = OxyColor.FromAColor(30, OxyColors.Green),
            Layer = OxyPlot.Annotations.AnnotationLayer.BelowSeries,
        });
        TrendModel.Series.Add(new LineSeries
        {
            Title = "BIS",
            Color = OxyColor.FromRgb(0x58, 0xA6, 0xFF),
            StrokeThickness = 2,
        });
    }

    private void InitPulseModel()
    {
        PulseWaveModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x21, 0x26, 0x2D),
            PlotAreaBackground = OxyColor.FromRgb(0x0D, 0x11, 0x17),
            Title = "Pulse / PPG",
            TitleFontSize = 10,
            TitleColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
        };
        PulseWaveModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        PulseWaveModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, IsAxisVisible = false });
        PulseWaveModel.Series.Add(new LineSeries
        {
            Color = OxyColor.FromRgb(0xF8, 0x51, 0x49),
            StrokeThickness = 1.5,
        });

        HrvModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x21, 0x26, 0x2D),
            PlotAreaBackground = OxyColor.FromRgb(0x0D, 0x11, 0x17),
            Title = "HRV (RR Intervals)",
            TitleFontSize = 10,
            TitleColor = OxyColor.FromRgb(0x8B, 0x94, 0x9E),
        };
        HrvModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        HrvModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, IsAxisVisible = false });
        HrvModel.Series.Add(new StemSeries
        {
            Color = OxyColor.FromRgb(0x3F, 0xB9, 0x50),
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
        series.Points.Clear();
        int i = 0;
        foreach (var v in buf) series.Points.Add(new DataPoint(i++, v));

        var xAxis = (LinearAxis)model.Axes[0];
        xAxis.Minimum = 0;
        xAxis.Maximum = WAVE_WINDOW_SAMPLES;
        model.InvalidatePlot(true);
    }

    public void ClearAllCharts()
    {
        _rawBuf.Clear(); _deltaBuf.Clear(); _thetaBuf.Clear();
        _alphaBuf.Clear(); _betaBuf.Clear(); _gammaBuf.Clear();
        _pulseBuf.Clear();

        foreach (var m in new[] { RawEEGModel, DeltaModel, ThetaModel, AlphaModel, BetaModel, GammaModel, PulseWaveModel })
        {
            ((LineSeries)m.Series[0]).Points.Clear();
            m.InvalidatePlot(true);
        }

        _dsaHistory.Clear();
        ((HeatMapSeries)DsaModel.Series[0]).Data = new double[0, 0];
        DsaModel.InvalidatePlot(true);

        _bisTrend.Clear();
        _sessionElapsedSec = 0;
        ((LineSeries)TrendModel.Series[0]).Points.Clear();
        TrendModel.InvalidatePlot(true);

        ((StemSeries)HrvModel.Series[0]).Points.Clear();
        HrvModel.InvalidatePlot(true);

        BisDisplay = "---"; BisValue = double.NaN; BisZoneDisplay = "";
        SqiDisplay = "---"; SqiValue = double.NaN;
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
        DsaModel.InvalidatePlot(true);
    }

    private void UpdateTrend(ProcessedEEGResult result)
    {
        if (double.IsNaN(result.BIS)) return;
        _sessionElapsedSec++;
        _bisTrend.Enqueue((_sessionElapsedSec, result.BIS));
        while (_bisTrend.Count > 0 && _sessionElapsedSec - _bisTrend.Peek().time > TREND_WINDOW_SEC)
            _bisTrend.Dequeue();

        var series = (LineSeries)TrendModel.Series[0];
        series.Points.Clear();
        foreach (var (t, b) in _bisTrend)
            series.Points.Add(new DataPoint(t - _sessionElapsedSec, b));

        var xAxis = (LinearAxis)TrendModel.Axes[0];
        xAxis.Minimum = -TREND_WINDOW_SEC; xAxis.Maximum = 0;
        TrendModel.InvalidatePlot(true);
    }

    private void UpdatePulseWave(ProcessedEEGResult result)
    {
        if (result.PulseWave.Length == 0) return;
        AppendWaveChart(PulseWaveModel, _pulseBuf, result.PulseWave);
    }

    private async Task CheckServiceAsync()
    {
        while (true)
        {
            IsServiceOnline = await _processing.PingAsync();
            await Task.Delay(5000);
        }
    }
}
