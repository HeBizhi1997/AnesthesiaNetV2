using Ads1299Monitor.Configuration;
using Ads1299Monitor.Models;
using Ads1299Monitor.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System.Windows;
using System.Windows.Threading;

namespace Ads1299Monitor.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly AppConfig _cfg;
    private readonly SerialPortService _serial;
    private readonly PulseSerialService _pulse;
    private readonly DataPipeline _pipeline;
    private readonly RecordingService _recording;
    private readonly EEGProcessingClient _processing;
    private readonly ILogger<MainViewModel> _logger;
    private readonly DispatcherTimer _clock;
    private DateTime _startedAt;

    // ── Index cards ──
    [ObservableProperty] private string _qConText = "--";
    [ObservableProperty] private string _qNoxText = "--";
    [ObservableProperty] private double _sqi;
    [ObservableProperty] private string _sqiText = "--";
    [ObservableProperty] private string _hrText = "--";
    [ObservableProperty] private string _hrvText = "--";

    // ── Band percentages ──
    [ObservableProperty] private string _deltaPct = "0.0%";
    [ObservableProperty] private string _thetaPct = "0.0%";
    [ObservableProperty] private string _alphaPct = "0.0%";
    [ObservableProperty] private string _betaPct = "0.0%";
    [ObservableProperty] private string _gammaPct = "0.0%";

    // ── Status ──
    [ObservableProperty] private bool _isRunning;
    [ObservableProperty] private string _startButtonText = "▶  开始采集";
    [ObservableProperty] private string _statusText = "未连接";
    [ObservableProperty] private string _elapsedText = "00:00";
    [ObservableProperty] private string _recordPathText = "";

    // ── Plots ──
    public PlotModel EegBandsModel { get; }
    public PlotModel DsaModel { get; }
    public PlotModel HrHrvModel { get; }
    public PlotModel TrendModel { get; }

    private const int BandOrderCount = 5;       // delta,theta,alpha,beta,gamma
    private readonly List<double>[] _bandBuf = { new(), new(), new(), new(), new() };
    private const int EegWindow = 1250;          // 5 s @ 250 Hz
    private readonly LineSeries[] _bandSeries = new LineSeries[BandOrderCount];

    private readonly AreaSeries[] _dsaSeries = new AreaSeries[BandOrderCount];
    private readonly List<double[]> _dsaHist = new();   // each = cumulative bounds [c0..c5]
    private const int DsaHistMax = 180;          // 3 min @ 1 Hz

    private LineSeries _hrSeries = null!, _hrvSeries = null!;
    private LineSeries _trendQcon = null!, _trendQnox = null!, _trendHr = null!;
    private int _t;                              // seconds since start (x-axis for trend/dsa)

    public MainViewModel(AppConfig cfg, SerialPortService serial, PulseSerialService pulse,
        DataPipeline pipeline, RecordingService recording, EEGProcessingClient processing,
        ILogger<MainViewModel> logger)
    {
        _cfg = cfg; _serial = serial; _pulse = pulse; _pipeline = pipeline;
        _recording = recording; _processing = processing; _logger = logger;

        EegBandsModel = BuildEegModel();
        DsaModel = BuildDsaModel();
        HrHrvModel = BuildHrHrvModel();
        TrendModel = BuildTrendModel();

        _serial.ConnectionStatusChanged += s => OnUi(() => StatusText = s);
        _pipeline.ResultAvailable += r => OnUi(() => OnResult(r));
        _pulse.BpmReceived += OnBpm;

        _clock = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _clock.Tick += (_, _) =>
        {
            if (IsRunning) ElapsedText = (DateTime.Now - _startedAt).ToString(@"mm\:ss");
        };
        _clock.Start();
    }

    // ── Start / stop (single button) ──────────────────────────────────────────

    [RelayCommand]
    private void Toggle()
    {
        if (IsRunning) StopAcquisition();
        else StartAcquisition();
    }

    private void StartAcquisition()
    {
        _serial.Gain = _cfg.Serial.EegGain;
        _serial.Differential = _cfg.Serial.EegDifferential;

        if (!_serial.Connect(_cfg.Serial.EegPort, _cfg.Serial.EegBaud, _cfg.Serial.EegSampleRate))
        {
            StatusText = $"无法打开 {_cfg.Serial.EegPort}";
            return;
        }
        _pipeline.DeviceSampleRate = _serial.SampleRate;
        _ = _processing.ResetSessionAsync();
        _recording.Start(_serial.SampleRate);
        RecordPathText = _recording.SessionDirectory ?? "";
        _pipeline.Start();

        if (_cfg.Serial.PulseEnabled)
            _pulse.Connect(_cfg.Serial.PulsePort);

        _startedAt = DateTime.Now;
        IsRunning = true;
        StartButtonText = "■  停止采集";
        _logger.LogInformation("Acquisition started on {Port}", _cfg.Serial.EegPort);
    }

    private void StopAcquisition()
    {
        _pipeline.Stop();
        _recording.Stop();
        _serial.Disconnect();
        if (_pulse.IsConnected) _pulse.Disconnect();
        IsRunning = false;
        StartButtonText = "▶  开始采集";
        StatusText = "已停止";
    }

    public void Shutdown()
    {
        try { StopAcquisition(); } catch { }
    }

    // ── Result handling ─────────────────────────────────────────────────────────

    private void OnResult(ProcessedEEGResult r)
    {
        QConText = double.IsNaN(r.BIS) ? "--" : Math.Round(r.BIS).ToString("0");
        QNoxText = r.FNox.HasValue ? Math.Round(r.FNox.Value).ToString("0") : "--";
        Sqi = r.SQI;
        SqiText = $"{Math.Round(r.SQI)}%";
        if (r.HeartRate is > 0) HrText = Math.Round(r.HeartRate.Value).ToString("0");
        if (r.HRV_RMSSD is > 0) HrvText = $"{r.HRV_RMSSD.Value:0.0}";

        double tot = r.DeltaPower + r.ThetaPower + r.AlphaPower + r.BetaPower + r.GammaPower;
        if (tot <= 0) tot = 1;
        double d = r.DeltaPower / tot, th = r.ThetaPower / tot, a = r.AlphaPower / tot,
               b = r.BetaPower / tot, g = r.GammaPower / tot;
        DeltaPct = $"{d * 100:0.0}%"; ThetaPct = $"{th * 100:0.0}%";
        AlphaPct = $"{a * 100:0.0}%"; BetaPct = $"{b * 100:0.0}%"; GammaPct = $"{g * 100:0.0}%";

        _t++;
        UpdateEegBands(r);
        UpdateDsa(d, th, a, b, g);
        UpdateTrend(r);
    }

    private void OnBpm(int bpm)
    {
        OnUi(() =>
        {
            if (bpm > 0)
            {
                HrText = bpm.ToString();
                _pipeline.CurrentHeartRate = bpm;
            }
        });
        if (bpm > 0) _recording.RecordRawVital(DateTime.Now, bpm, 0, 0);
    }

    // ── EEG 5-band waveform panel ───────────────────────────────────────────────

    private void UpdateEegBands(ProcessedEEGResult r)
    {
        double[][] waves = { r.DeltaWave, r.ThetaWave, r.AlphaWave, r.BetaWave, r.GammaWave };
        for (int i = 0; i < BandOrderCount; i++)
        {
            if (waves[i].Length == 0) continue;
            _bandBuf[i].AddRange(waves[i]);
            if (_bandBuf[i].Count > EegWindow)
                _bandBuf[i].RemoveRange(0, _bandBuf[i].Count - EegWindow);
        }
        for (int i = 0; i < BandOrderCount; i++)
        {
            var buf = _bandBuf[i];
            var s = _bandSeries[i];
            s.Points.Clear();
            if (buf.Count == 0) continue;
            double mean = 0; for (int k = 0; k < buf.Count; k++) mean += buf[k]; mean /= buf.Count;
            double max = 1e-9; for (int k = 0; k < buf.Count; k++) max = Math.Max(max, Math.Abs(buf[k] - mean));
            double offset = i + 0.5;   // delta lowest, gamma highest
            for (int k = 0; k < buf.Count; k++)
                s.Points.Add(new DataPoint(k, offset + (buf[k] - mean) / max * 0.42));
        }
        EegBandsModel.InvalidatePlot(true);
    }

    // ── DSA stacked-area panel ──────────────────────────────────────────────────

    private void UpdateDsa(double d, double th, double a, double b, double g)
    {
        double c0 = d, c1 = c0 + th, c2 = c1 + a, c3 = c2 + b, c4 = c3 + g;
        _dsaHist.Add(new[] { 0.0, c0, c1, c2, c3, Math.Max(c4, 1e-6) });
        if (_dsaHist.Count > DsaHistMax) _dsaHist.RemoveAt(0);

        for (int i = 0; i < BandOrderCount; i++)
        {
            var area = _dsaSeries[i];
            area.Points.Clear(); area.Points2.Clear();
            for (int t = 0; t < _dsaHist.Count; t++)
            {
                area.Points.Add(new DataPoint(t, _dsaHist[t][i + 1]));   // upper bound
                area.Points2.Add(new DataPoint(t, _dsaHist[t][i]));      // lower bound
            }
        }
        DsaModel.InvalidatePlot(true);
    }

    // ── HR/HRV + trend panels ───────────────────────────────────────────────────

    private void UpdateTrend(ProcessedEEGResult r)
    {
        if (!double.IsNaN(r.BIS)) _trendQcon.Points.Add(new DataPoint(_t, r.BIS));
        if (r.FNox.HasValue) _trendQnox.Points.Add(new DataPoint(_t, r.FNox.Value));
        if (r.HeartRate is > 0) { _trendHr.Points.Add(new DataPoint(_t, r.HeartRate.Value)); _hrSeries.Points.Add(new DataPoint(_t, r.HeartRate.Value)); }
        if (r.HRV_RMSSD is > 0) _hrvSeries.Points.Add(new DataPoint(_t, r.HRV_RMSSD.Value));

        TrimSeries(_trendQcon, 1800); TrimSeries(_trendQnox, 1800); TrimSeries(_trendHr, 1800);
        TrimSeries(_hrSeries, 240); TrimSeries(_hrvSeries, 240);
        TrendModel.InvalidatePlot(true);
        HrHrvModel.InvalidatePlot(true);
    }

    private static void TrimSeries(LineSeries s, int max)
    {
        if (s.Points.Count > max) s.Points.RemoveRange(0, s.Points.Count - max);
    }

    // ── Plot construction ───────────────────────────────────────────────────────

    private static readonly OxyColor[] BandColors =
    {
        OxyColor.Parse("#8B5CF6"), OxyColor.Parse("#3B82F6"), OxyColor.Parse("#22C55E"),
        OxyColor.Parse("#F59E0B"), OxyColor.Parse("#EF4444"),
    };
    private static readonly string[] BandNames = { "δ", "θ", "α", "β", "γ" };

    private static PlotModel DarkModel() => new()
    {
        Background = OxyColors.Transparent,
        PlotAreaBorderColor = OxyColor.Parse("#1E2A38"),
        TextColor = OxyColor.Parse("#7D8B9A"),
        PlotMargins = new OxyThickness(36, 4, 8, 20),
    };

    private PlotModel BuildEegModel()
    {
        var m = DarkModel();
        m.PlotMargins = new OxyThickness(4, 4, 4, 4);
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = 0, Maximum = 5, IsAxisVisible = false });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        for (int i = 0; i < BandOrderCount; i++)
        {
            _bandSeries[i] = new LineSeries { Color = BandColors[i], StrokeThickness = 1.4 };
            m.Series.Add(_bandSeries[i]);
        }
        return m;
    }

    private PlotModel BuildDsaModel()
    {
        var m = DarkModel();
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = 0, Maximum = 1, IsAxisVisible = false });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        for (int i = 0; i < BandOrderCount; i++)
        {
            _dsaSeries[i] = new AreaSeries
            {
                Color = OxyColors.Transparent,
                Fill = BandColors[i],
                StrokeThickness = 0,
            };
            m.Series.Add(_dsaSeries[i]);
        }
        return m;
    }

    private PlotModel BuildHrHrvModel()
    {
        var m = DarkModel();
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Key = "hr", Minimum = 0, Maximum = 200, TextColor = OxyColor.Parse("#EF4444") });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Right, Key = "hrv", Minimum = 0, Maximum = 100, TextColor = OxyColor.Parse("#38BDF8") });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        _hrSeries = new LineSeries { Title = "HR", Color = OxyColor.Parse("#EF4444"), YAxisKey = "hr", MarkerType = MarkerType.Circle, MarkerSize = 2 };
        _hrvSeries = new LineSeries { Title = "HRV", Color = OxyColor.Parse("#38BDF8"), YAxisKey = "hrv", MarkerType = MarkerType.Circle, MarkerSize = 2 };
        m.Series.Add(_hrSeries); m.Series.Add(_hrvSeries);
        return m;
    }

    private PlotModel BuildTrendModel()
    {
        var m = DarkModel();
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Key = "idx", Minimum = 0, Maximum = 100 });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Right, Key = "hr", Minimum = 40, Maximum = 180, TextColor = OxyColor.Parse("#22C55E") });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        _trendQcon = new LineSeries { Title = "qCON", Color = OxyColor.Parse("#3B82F6"), YAxisKey = "idx", StrokeThickness = 2, LineStyle = LineStyle.Dash };
        _trendQnox = new LineSeries { Title = "qNOX", Color = OxyColor.Parse("#F59E0B"), YAxisKey = "idx", StrokeThickness = 2, LineStyle = LineStyle.Dash };
        _trendHr = new LineSeries { Title = "HR", Color = OxyColor.Parse("#22C55E"), YAxisKey = "hr", StrokeThickness = 1.5 };
        m.Series.Add(_trendQcon); m.Series.Add(_trendQnox); m.Series.Add(_trendHr);
        return m;
    }

    private static void OnUi(Action a)
    {
        var d = Application.Current?.Dispatcher;
        if (d == null || d.CheckAccess()) a();
        else d.BeginInvoke(a);
    }
}
