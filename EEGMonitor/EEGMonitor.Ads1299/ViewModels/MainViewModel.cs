using Ads1299Monitor.Configuration;
using Ads1299Monitor.Models;
using Ads1299Monitor.Services;
using Ads1299Monitor.Views;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.Measure;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Extensions;
using LiveChartsCore.SkiaSharpView.Painting;
using Microsoft.Extensions.Logging;
using SkiaSharp;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Threading;

namespace Ads1299Monitor.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly AppConfig _cfg;
    private readonly SerialPortService _serial;
    private readonly PulseSerialService _pulse;
    private readonly PpgSerialService _ppg;
    private readonly DataPipeline _pipeline;
    private readonly RecordingService _recording;
    private readonly EEGProcessingClient _processing;
    private readonly PythonServiceLauncher _launcher;
    private readonly ILogger<MainViewModel> _logger;
    private readonly DispatcherTimer _clock;
    private DateTime _startedAt;

    // ── Patient ──
    [ObservableProperty] private string _patientName = "";
    [ObservableProperty] private string _patientGender = "";
    [ObservableProperty] private string _patientAge = "";
    [ObservableProperty] private string _admissionNo = "";
    [ObservableProperty] private string _operatingRoom = "";
    [ObservableProperty] private string _surgeryName = "";
    [ObservableProperty] private string _anesthesiaMethod = "";

    // ── Status ──
    [ObservableProperty] private bool _isRunning;
    [ObservableProperty] private string _monitorStatusText = "未监测";
    [ObservableProperty] private string _monitorElapsed = "00:00:00";
    [ObservableProperty] private string _recordElapsed = "00:00:00";
    [ObservableProperty] private string _startButtonText = "▶  开始监测";
    [ObservableProperty] private string _nowText = "";
    [ObservableProperty] private bool _serviceOnline;
    [ObservableProperty] private string _serviceStatus = "推理服务: 检查中…";
    [ObservableProperty] private bool _electrodeAlertVisible;
    [ObservableProperty] private string _electrodeAlert = "";
    [ObservableProperty] private bool _noDataVisible;
    [ObservableProperty] private string _noDataText = "";
    private DateTime _lastDataAt;

    // ── Vitals (PR/SpO₂/PRV/PI come from the PPG 指夹模组; NIBP/RR have no source yet) ──
    [ObservableProperty] private string _prText = "--";        // 脉率 PR (was HR)
    [ObservableProperty] private string _spo2Text = "--";
    [ObservableProperty] private string _prvText = "--";       // 脉率变异性 PRV (was HRV)
    [ObservableProperty] private string _nibpText = "-- / --";
    [ObservableProperty] private string _mapText = "MAP --";
    [ObservableProperty] private string _rrText = "--";
    [ObservableProperty] private string _tpiText = "--";
    [ObservableProperty] private IReadOnlyList<double>? _prSpark;
    [ObservableProperty] private IReadOnlyList<double>? _spo2Spark;
    [ObservableProperty] private IReadOnlyList<double>? _prvSpark;
    [ObservableProperty] private IReadOnlyList<double>? _rrSpark;
    [ObservableProperty] private IReadOnlyList<double>? _tpiSpark;
    private readonly List<double> _prRoll = new(), _prvRoll = new(), _spo2Roll = new(), _piRoll = new();

    // ── Indices ──
    [ObservableProperty] private string _qConStatus = "--";
    [ObservableProperty] private string _qNoxStatus = "--";
    [ObservableProperty] private double _sqi;
    [ObservableProperty] private string _sqiText = "--";
    [ObservableProperty] private string _sqiQuality = "--";

    // SPI 伤害感受指数 (Surgical Pleth Index) — PPG-derived, replaces SQI in the indices panel.
    [ObservableProperty] private double _spi;
    [ObservableProperty] private string _spiText = "--";
    [ObservableProperty] private string _spiStatus = "--";
    private readonly List<double> _hbiRoll = new(), _ppgaRoll = new();   // for SPI histogram normalisation

    // ── Band % ──
    [ObservableProperty] private string _deltaPct = "--";
    [ObservableProperty] private string _thetaPct = "--";
    [ObservableProperty] private string _alphaPct = "--";
    [ObservableProperty] private string _betaPct = "--";
    [ObservableProperty] private string _gammaPct = "--";

    // ── Recording info ──
    [ObservableProperty] private string _recordFileName = "--";
    [ObservableProperty] private string _storagePath = "本地设备";
    [ObservableProperty] private string _freeSpaceText = "--";

    public ObservableCollection<EventItem> Events { get; } = new();

    // ── LiveCharts ──
    public ISeries[] EegSeries { get; }
    public Axis[] EegXAxes { get; }
    public Axis[] EegYAxes { get; }

    public ISeries[] TrendSeries { get; }
    public Axis[] TrendXAxes { get; }
    public Axis[] TrendYAxes { get; }
    public ObservableCollection<RectangularSection> TrendSections { get; } = new();

    // Trend series refs so the legend checkboxes can show/hide each line.
    private LineSeries<ObservablePoint> _sQcon = null!, _sQnox = null!, _sSpi = null!;

    // Bound to the trend-legend checkboxes (defaults match the previous static IsChecked states).
    [ObservableProperty] private bool _showQcon = true;
    [ObservableProperty] private bool _showQnox = true;
    [ObservableProperty] private bool _showSpi = true;
    partial void OnShowQconChanged(bool v) => _sQcon.IsVisible = v;
    partial void OnShowQnoxChanged(bool v) => _sQnox.IsVisible = v;
    partial void OnShowSpiChanged(bool v) => _sSpi.IsVisible = v;

    public ISeries[] DonutSeries { get; }
    public ISeries[] QConGauge { get; }
    public ISeries[] QNoxGauge { get; }

    private const int Lanes = 6;                  // raw, δ, θ, α, β, γ
    private const int EegWindow = 2500;           // 10 s @ 250 Hz
    private readonly List<double>[] _laneBuf = { new(), new(), new(), new(), new(), new() };
    private readonly LineSeries<ObservablePoint>[] _lane = new LineSeries<ObservablePoint>[Lanes];

    private readonly ObservableCollection<ObservablePoint> _trQcon = new(), _trQnox = new(), _trSpi = new();

    // 原始脉搏波 (PPG IR) rolling display buffer.
    private readonly object _pulseLock = new();
    private readonly List<double> _pulseRoll = new();
    private int _pulseDecim;
    [ObservableProperty] private IReadOnlyList<double>? _pulseWave;

    private readonly ObservableValue _dDelta = new(0), _dTheta = new(0), _dAlpha = new(0), _dBeta = new(0), _dGamma = new(0);
    private readonly ObservableValue _qConVal = new(0), _qNoxVal = new(0);
    // Trend X = elapsed seconds since acquisition start (wall-clock). Keying off this — instead of
    // an EEG-epoch counter — lets PR/SpO₂ keep trending even when the EEG board sends nothing
    // (OnResult never fires). TrendTimeLabel maps it back to HH:mm:ss.
    private double TrendX => _startedAt == default ? 0 : (DateTime.Now - _startedAt).TotalSeconds;

    private static SolidColorPaint Paint(string hex) => new(SKColor.Parse(hex));
    private static readonly string[] LaneHex = { "#E2E8F0", "#3B82F6", "#06B6D4", "#22C55E", "#F59E0B", "#A855F7" };

    public MainViewModel(AppConfig cfg, SerialPortService serial, PulseSerialService pulse,
        PpgSerialService ppg, DataPipeline pipeline, RecordingService recording, EEGProcessingClient processing,
        PythonServiceLauncher launcher, ILogger<MainViewModel> logger)
    {
        _cfg = cfg; _serial = serial; _pulse = pulse; _ppg = ppg; _pipeline = pipeline;
        _recording = recording; _processing = processing; _launcher = launcher; _logger = logger;

        var p = cfg.Patient;
        PatientName = p.Name;
        PatientGender = p.Gender == "男" ? "♂" : (p.Gender == "女" ? "♀" : p.Gender);
        PatientAge = $"{p.Age}岁"; AdmissionNo = p.AdmissionNo; OperatingRoom = p.OperatingRoom;
        SurgeryName = p.SurgeryName; AnesthesiaMethod = p.AnesthesiaMethod;

        // EEG: 6 stacked offset lanes
        for (int i = 0; i < Lanes; i++)
            _lane[i] = new LineSeries<ObservablePoint>
            {
                Values = Array.Empty<ObservablePoint>(),
                Stroke = new SolidColorPaint(SKColor.Parse(LaneHex[i]), 1.1f),
                Fill = null, GeometrySize = 0, LineSmoothness = 0, AnimationsSpeed = TimeSpan.Zero,
            };
        EegSeries = _lane.Cast<ISeries>().ToArray();
        EegXAxes = new[] { Hidden() };
        EegYAxes = new[] { new Axis { MinLimit = 0, MaxLimit = Lanes, LabelsPaint = null, SeparatorsPaint = null, ShowSeparatorLines = false } };

        // Trend (qCON / qNOX / SPI — all 0–100 on the left axis)
        _sQcon = TrendLine(_trQcon, "qCON", "#3B82F6", 0);
        _sQnox = TrendLine(_trQnox, "qNOX", "#F59E0B", 0);
        _sSpi  = TrendLine(_trSpi,  "SPI",  "#A855F7", 0);
        TrendSeries = new ISeries[] { _sQcon, _sQnox, _sSpi };
        // Trend X = wall-clock time axis (aligns data + event markers for intuitive review)
        TrendXAxes = new[]
        {
            new Axis
            {
                Labeler = TrendTimeLabel,
                LabelsPaint = Paint("#7D8B9A"), TextSize = 11, MinStep = 1, UnitWidth = 1,
                SeparatorsPaint = new SolidColorPaint(SKColor.Parse("#16202C"), 1),
            }
        };
        TrendYAxes = new[]
        {
            new Axis { MinLimit = 0, MaxLimit = 100, LabelsPaint = Paint("#7D8B9A"), TextSize = 11,
                       SeparatorsPaint = new SolidColorPaint(SKColor.Parse("#16202C"), 1) },
        };

        // Donut
        DonutSeries = new ISeries[]
        {
            DonutSlice(_dDelta, "#8B5CF6"), DonutSlice(_dTheta, "#3B82F6"), DonutSlice(_dAlpha, "#22C55E"),
            DonutSlice(_dBeta, "#F59E0B"), DonutSlice(_dGamma, "#EF4444"),
        };

        // Gauges
        QConGauge = BuildGauge(_qConVal, "#F59E0B");
        QNoxGauge = BuildGauge(_qNoxVal, "#22C55E");

        _serial.ConnectionStatusChanged += s => OnUi(() => MonitorStatusText = s);
        _pipeline.ResultAvailable += r => OnUi(() => OnResult(r));
        _pulse.BpmReceived += OnBpm;
        _ppg.ReadingReceived += OnPpg;
        _ppg.RawSampleReceived += OnPpgRaw;   // 无损存原始 PPG 波 + 脉搏波显示

        _clock = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _clock.Tick += (_, _) =>
        {
            NowText = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            if (IsRunning)
            {
                var el = (DateTime.Now - _startedAt).ToString(@"hh\:mm\:ss");
                MonitorElapsed = el; RecordElapsed = _recording.IsRecording ? el : "00:00:00";

                // No-data watchdog: serial port is open but the board sends nothing
                // (device powered off / cable / hung firmware). No frames ⇒ OnResult never
                // fires ⇒ _lastDataAt goes stale. Surface it instead of a silent blank UI.
                double stale = (DateTime.Now - _lastDataAt).TotalSeconds;
                bool noData = stale > 3.0;
                if (noData != NoDataVisible)
                {
                    NoDataVisible = noData;
                    NoDataText = noData
                        ? $"⚠ 未接收到采集数据({stale:F0}s)—— 串口已打开但无数据,请检查采集设备电源 / USB 连接"
                        : "";
                }
            }
        };
        _clock.Start();

        _ = InitServiceAsync();
    }

    private async Task InitServiceAsync()
    {
        OnUi(() => ServiceStatus = "推理服务: 启动中…");
        bool ok;
        try { ok = await _launcher.EnsureRunningAsync(); }
        catch { ok = false; }
        OnUi(() => { ServiceOnline = ok; ServiceStatus = ok ? "推理服务: 在线" : "推理服务: 离线"; });
    }

    // ── Start / stop ──
    [RelayCommand]
    private void Toggle() { if (IsRunning) StopAcquisition(); else StartAcquisition(); }

    private void StartAcquisition()
    {
        _serial.Gain = _cfg.Serial.EegGain;
        _serial.Differential = _cfg.Serial.EegDifferential;
        if (!_serial.Connect(_cfg.Serial.EegPort, _cfg.Serial.EegBaud, _cfg.Serial.EegSampleRate))
        { MonitorStatusText = $"无法打开 {_cfg.Serial.EegPort}"; return; }
        _pipeline.DeviceSampleRate = _serial.SampleRate;
        _ = _processing.ResetSessionAsync();
        _recording.Start(_serial.SampleRate);
        if (_recording.SessionDirectory is { } dir)
        { RecordFileName = Path.GetFileName(dir); StoragePath = dir; UpdateFreeSpace(dir); }
        _pipeline.Start();
        if (_cfg.Serial.PulseEnabled) _pulse.Connect(_cfg.Serial.PulsePort);
        if (_cfg.Serial.PpgEnabled && !_ppg.Connect(_cfg.Serial.PpgPort, _cfg.Serial.PpgBaud))
            _logger.LogWarning("PPG 模组 {Port} 打开失败 —— 脉率/血氧将无数据", _cfg.Serial.PpgPort);
        _startedAt = DateTime.Now; _lastDataAt = DateTime.Now; IsRunning = true;
        StartButtonText = "■  停止监测"; MonitorStatusText = "监测中";
    }

    private void StopAcquisition()
    {
        _pipeline.Stop(); _recording.Stop(); _serial.Disconnect();
        if (_pulse.IsConnected) _pulse.Disconnect();
        if (_ppg.IsConnected) _ppg.Disconnect();
        IsRunning = false; StartButtonText = "▶  开始监测"; MonitorStatusText = "已停止";
        ElectrodeAlertVisible = false; ElectrodeAlert = "";
        NoDataVisible = false; NoDataText = "";
    }

    public void Shutdown() { try { StopAcquisition(); } catch { } }

    // ── Events ──
    [RelayCommand]
    private void AddEvent()
    {
        var dlg = new EventDialog { Owner = Application.Current?.MainWindow };
        if (dlg.ShowDialog() == true && dlg.Result is { } ev)
            CommitEvent(ev);
    }

    [RelayCommand]
    private void OpenPlayback()
    {
        var dlg = new PlaybackDialog(_cfg.Recording.OutputDirectory) { Owner = Application.Current?.MainWindow };
        dlg.ShowDialog();
    }

    [RelayCommand]
    private void MarkMoment() =>
        CommitEvent(new EventItem { Category = "标记", Name = "标记当前时刻", Dose = "—", Operator = "—", Note = "—" });

    private void CommitEvent(EventItem ev)
    {
        ev.Time = DateTime.Now.ToString("HH:mm:ss");
        ev.TimeSeconds = TrendX;
        Events.Insert(0, ev);
        _recording.RecordEvent(ev);
        TrendSections.Add(new RectangularSection
        {
            Xi = TrendX, Xj = TrendX,
            Stroke = new SolidColorPaint(SKColor.Parse("#94A3B8"), 1) { PathEffect = new LiveChartsCore.SkiaSharpView.Painting.Effects.DashEffect(new float[] { 4, 4 }) },
            Label = ev.Name, LabelSize = 11, LabelPaint = Paint("#94A3B8"),
        });
    }

    // ── Result ──
    private void OnResult(ProcessedEEGResult r)
    {
        _lastDataAt = DateTime.Now;     // data is flowing → feeds the no-data watchdog

        // Electrode / lead-off gating: when the signal isn't valid (electrode not connected,
        // shorted, or saturated) the consciousness indices are meaningless → mask them and
        // raise a banner. The EEG waveform keeps updating so the user can see the noise.
        bool valid = r.SignalValid;
        ElectrodeAlert = r.ElectrodeStatus switch
        {
            "lead_off"  => "⚠ 电极未接入 / 导联脱落 —— 意识指标无效，请检查电极",
            "flat"      => "⚠ 无信号(电极短路或断开)—— 请检查电极连接",
            "saturated" => "⚠ 信号饱和 / 削顶 —— 超出量程，检查电极/增益",
            _            => "",
        };
        ElectrodeAlertVisible = !valid;

        if (valid)
        {
            if (!double.IsNaN(r.BIS))
            {
                _qConVal.Value = r.BIS; QConStatus = ConsciousnessStatus(r.BIS);
                // qNOX is no longer taken from the model's fnox; it is derived from qCON via the
                // CONOX linear relationship (see DeriveQnox).
                double qnox = DeriveQnox(r.BIS);
                _qNoxVal.Value = qnox; QNoxStatus = DepthStatus(qnox);
            }
        }
        else
        {
            _qConVal.Value = 0; _qNoxVal.Value = 0;
            QConStatus = "信号无效"; QNoxStatus = "信号无效";
        }
        // SQI + 频谱占比 are EEG-derived: when the electrode is off they're just floating-input
        // noise (SQI jitters, donut churns). Gate them on validity — show them only when valid,
        // otherwise blank/freeze so the UI doesn't imply live data.
        if (valid)
        {
            Sqi = r.SQI;
            SqiText = $"{Math.Round(r.SQI)}";
            SqiQuality = QualityStatus(r.SQI);

            double tot = r.DeltaPower + r.ThetaPower + r.AlphaPower + r.BetaPower + r.GammaPower; if (tot <= 0) tot = 1;
            double d = r.DeltaPower / tot, th = r.ThetaPower / tot, al = r.AlphaPower / tot, b = r.BetaPower / tot, g = r.GammaPower / tot;
            _dDelta.Value = Math.Round(d * 100, 1); _dTheta.Value = Math.Round(th * 100, 1); _dAlpha.Value = Math.Round(al * 100, 1);
            _dBeta.Value = Math.Round(b * 100, 1); _dGamma.Value = Math.Round(g * 100, 1);
            DeltaPct = $"{d * 100:0}%"; ThetaPct = $"{th * 100:0}%"; AlphaPct = $"{al * 100:0}%"; BetaPct = $"{b * 100:0}%"; GammaPct = $"{g * 100:0}%";
        }
        else
        {
            Sqi = 0; SqiText = "--"; SqiQuality = "无效";
            _dDelta.Value = _dTheta.Value = _dAlpha.Value = _dBeta.Value = _dGamma.Value = 0;
            DeltaPct = ThetaPct = AlphaPct = BetaPct = GammaPct = "--";
        }

        // 脉率 PR / SpO₂ / PRV / 灌注指数 come from the PPG 指夹模组 (see OnPpg); not the EEG result.

        UpdateEeg(r);
        if (valid && !double.IsNaN(r.BIS)) { AddTrend(_trQcon, r.BIS); AddTrend(_trQnox, DeriveQnox(r.BIS)); }
    }

    // ── PPG 指夹模组: 脉率 PR + 脉率变异性 PRV + 血氧 SpO₂ + 灌注指数 PI ──
    private void OnPpg(PpgSerialService.PpgReading r)
    {
        double pr = r.Pr > 0 ? r.Pr : r.DeviceHr;            // computed PR; fall back to device reading
        OnUi(() =>
        {
            // Vitals cards driven here, independent of the EEG pipeline. (PR/SpO₂ trend lines removed.)
            if (pr > 0) { PrText = Math.Round(pr).ToString("0"); Push(_prRoll, pr, v => PrSpark = v); }
            if (r.Spo2 > 0) { Spo2Text = r.Spo2.ToString(); Push(_spo2Roll, r.Spo2, v => Spo2Spark = v); }
            if (r.Prv > 0) { PrvText = $"{r.Prv:0}"; Push(_prvRoll, r.Prv, v => PrvSpark = v); }
            if (r.Pi > 0) { TpiText = $"{r.Pi:0.0}"; Push(_piRoll, r.Pi, v => TpiSpark = v); }

            // SPI 伤害感受指数 = 100 − (0.7·PPGAnorm + 0.3·HBInorm).  PPGA≈灌注指数(脉搏波幅),
            // HBI=心搏间期(60000/脉率 ms);两者按滚动百分位归一到 0–100(SPI 的“直方图变换”)。
            if (pr > 0 && r.Pi > 0)
            {
                double ppgaNorm = PushPercentile(_ppgaRoll, r.Pi);
                double hbiNorm = PushPercentile(_hbiRoll, 60000.0 / pr);
                double spi = Math.Clamp(100.0 - (0.7 * ppgaNorm + 0.3 * hbiNorm), 0, 100);
                Spi = spi; SpiText = $"{Math.Round(spi)}"; SpiStatus = SpiStatusText(spi);
                AddTrend(_trSpi, spi);
            }
        });
        if (pr > 0) _pipeline.CurrentHeartRate = (int)Math.Round(pr);
        _recording.RecordRawVital(r.Time, pr, r.Spo2, r.Pi);
    }

    // 原始 PPG 帧 (~125 Hz, PPG 线程): 无损录制 + 喂入脉搏波显示缓冲。逐样本刷 UI 太频繁,
    // 因此每 8 个样本(~15 Hz)推送一次最近 ~4 s 的快照给 PulseWave。
    private void OnPpgRaw(DateTime ts, int ir, int red)
    {
        _recording.RecordRawPpg(ts, ir, red);
        double[]? snap = null;
        lock (_pulseLock)
        {
            _pulseRoll.Add(ir);
            if (_pulseRoll.Count > 500) _pulseRoll.RemoveAt(0);
            if (++_pulseDecim >= 8) { _pulseDecim = 0; snap = _pulseRoll.ToArray(); }
        }
        if (snap != null) OnUi(() => PulseWave = snap);
    }

    // Legacy HKG-07D pulse sensor (PulseEnabled=false by default). When active and no PPG
    // module is present, it can still drive the PR display.
    private void OnBpm(int bpm)
    {
        OnUi(() => { if (bpm > 0) { PrText = bpm.ToString(); _pipeline.CurrentHeartRate = bpm; Push(_prRoll, bpm, v => PrSpark = v); } });
        if (bpm > 0) _recording.RecordRawVital(DateTime.Now, bpm, 0, 0);
    }

    private static void Push(List<double> roll, double v, Action<double[]> set)
    {
        roll.Add(v); if (roll.Count > 120) roll.RemoveAt(0); set(roll.ToArray());
    }

    // Percentile rank (0–100) of v within a rolling ~5 min window — SPI's histogram normalisation.
    private static double PushPercentile(List<double> roll, double v)
    {
        roll.Add(v); if (roll.Count > 300) roll.RemoveAt(0);   // ~5 min @ 1 Hz
        if (roll.Count < 5) return 50.0;                        // too little history → neutral
        int below = 0; for (int i = 0; i < roll.Count; i++) if (roll[i] < v) below++;
        return 100.0 * below / roll.Count;
    }

    // SPI 目标区间 20–50 为镇痛适当;>50 提示伤害感受↑(镇痛不足),<20 提示镇痛偏深。
    private static string SpiStatusText(double v) => v > 50 ? "镇痛不足" : v >= 20 ? "适当" : "镇痛偏深";

    private void AddTrend(ObservableCollection<ObservablePoint> s, double v)
    {
        s.Add(new ObservablePoint(TrendX, v));
        if (s.Count > 7200) s.RemoveAt(0);
    }

    private void UpdateEeg(ProcessedEEGResult r)
    {
        double[][] lanes = { r.FilteredEEG, r.DeltaWave, r.ThetaWave, r.AlphaWave, r.BetaWave, r.GammaWave };
        for (int i = 0; i < Lanes; i++)
        {
            if (lanes[i].Length > 0)
            {
                _laneBuf[i].AddRange(lanes[i]);
                if (_laneBuf[i].Count > EegWindow) _laneBuf[i].RemoveRange(0, _laneBuf[i].Count - EegWindow);
            }
            var buf = _laneBuf[i];
            if (buf.Count == 0) continue;
            double mean = 0; for (int k = 0; k < buf.Count; k++) mean += buf[k]; mean /= buf.Count;
            double max = 1e-9; for (int k = 0; k < buf.Count; k++) max = Math.Max(max, Math.Abs(buf[k] - mean));
            double offset = (Lanes - 1 - i) + 0.5;
            var pts = new ObservablePoint[buf.Count];
            for (int k = 0; k < buf.Count; k++) pts[k] = new ObservablePoint(k, offset + (buf[k] - mean) / max * 0.44);
            _lane[i].Values = pts;
        }
    }

    private void UpdateFreeSpace(string dir)
    {
        try
        {
            var root = Path.GetPathRoot(Path.GetFullPath(dir));
            if (root != null) FreeSpaceText = $"{new DriveInfo(root).AvailableFreeSpace / 1024.0 / 1024 / 1024:0} GB";
        }
        catch { FreeSpaceText = "--"; }
    }

    // ── Chart builders ──
    // Trend X tick (elapsed-second index) → wall-clock HH:mm:ss, aligned to acquisition start.
    private string TrendTimeLabel(double x) =>
        _startedAt == default ? "" : _startedAt.AddSeconds(x).ToString("HH:mm:ss");

    private static Axis Hidden() => new() { LabelsPaint = null, SeparatorsPaint = null, ShowSeparatorLines = false, TicksPaint = null };

    private static LineSeries<ObservablePoint> TrendLine(ObservableCollection<ObservablePoint> values, string name, string hex, int yAxis) => new()
    {
        Name = name, Values = values, Stroke = new SolidColorPaint(SKColor.Parse(hex), 1.8f),
        Fill = null, GeometrySize = 0, LineSmoothness = 0.3, ScalesYAt = yAxis, AnimationsSpeed = TimeSpan.Zero,
    };

    private static PieSeries<ObservableValue> DonutSlice(ObservableValue v, string hex) => new()
    {
        Values = new[] { v }, Fill = new SolidColorPaint(SKColor.Parse(hex)),
        MaxRadialColumnWidth = 26, DataLabelsPaint = null,
    };

    private static ISeries[] BuildGauge(ObservableValue v, string hex) => GaugeGenerator.BuildSolidGauge(
        new GaugeItem(v, s =>
        {
            s.Fill = new SolidColorPaint(SKColor.Parse(hex));
            s.DataLabelsPaint = new SolidColorPaint(SKColors.White);
            s.DataLabelsSize = 38;
            s.DataLabelsPosition = PolarLabelsPosition.ChartCenter;
            s.DataLabelsFormatter = pt => $"{pt.Coordinate.PrimaryValue:0}";
            s.InnerRadius = 72;
            // CornerRadius default is non-zero → the arc's rounded end-caps poke past the
            // value fill as dark "blobs". Square ends render a clean ring. MaxRadialColumnWidth
            // pins a fixed band thickness so value + background overlay exactly.
            s.CornerRadius = 0;
            s.MaxRadialColumnWidth = 18;
        }),
        new GaugeItem(GaugeItem.Background, s =>
        {
            s.InnerRadius = 72;
            s.CornerRadius = 0;
            s.MaxRadialColumnWidth = 18;
            s.Fill = new SolidColorPaint(SKColor.Parse("#1E2A38"));
        }));

    // qNOX derived from qCON via the CONOX (qCON 2000) population relationship
    // qCON = 0.79·qNOX + 5.8 (Melia et al., R²=0.84) ⇒ qNOX = (qCON − 5.8)/0.79.
    // Clamped to the 0–99 index range. NOTE: this is a statistical correlation, not the
    // device's true independent nociception algorithm — individual error can be large.
    private static double DeriveQnox(double qcon) => Math.Clamp((qcon - 5.8) / 0.79, 0, 99);

    private static string ConsciousnessStatus(double v) =>
        v >= 80 ? "清醒" : v >= 60 ? "轻度镇静" : v >= 40 ? "中度抑制" : v >= 20 ? "深度抑制" : "爆发抑制";
    private static string DepthStatus(double v) => v >= 80 ? "偏浅" : v >= 60 ? "适中" : v >= 40 ? "偏深" : "过深";
    private static string QualityStatus(double v) => v >= 80 ? "优" : v >= 60 ? "良" : v >= 40 ? "中" : "差";

    private static void OnUi(Action a)
    {
        var d = Application.Current?.Dispatcher;
        if (d == null || d.CheckAccess()) a(); else d.BeginInvoke(a);
    }
}
