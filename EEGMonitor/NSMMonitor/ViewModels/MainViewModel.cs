using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using NSMMonitor.Configuration;
using NSMMonitor.Models;
using NSMMonitor.Services;
using NSMMonitor.Services.Ppg;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Legends;
using OxyPlot.Series;

namespace NSMMonitor.ViewModels;

public sealed partial class MainViewModel : ObservableObject
{
    private const int EEG_WINDOW = 600;     // 显示最近 6 秒波形
    private const int EEG_SAMPLES_PER_PACKET = 100;
    private const int TREND_WINDOW_SEC = 300;
    private const int DSA_BINS = 44;        // 密度谱阵列频段数 (1-44 Hz)
    private const int DSA_COLS = 300;       // 频谱图时间列数（约 5 分钟，每帧一列）
    private const int PPG_WINDOW = 500;     // 脉搏波显示窗：125 Hz × 4 秒

    private readonly NsmSerialService _serial;
    private readonly NsmSimulatorService _simulator;
    private readonly NsmPlaybackService _playback;
    private readonly NsmRecordingService _recorder;
    private readonly ILogger<MainViewModel> _logger;

    private readonly NsmConfig _config;
    private readonly PpgSerialService _ppgSerial;
    private readonly PpgSimulatorService _ppgSimulator;
    private readonly PulseOximetryProcessor _oximetry;
    private readonly SpiCalculator _spiCalc;
    private IPpgDataSource _ppgSource;

    private INsmDataSource _source;
    private double _eegX;
    private DateTime _firstPacketTs;
    private bool _firstPacketSeen;

    // 回放时整场 SessionBuffer 已在连接时一次性预加载完毕。为 true 时，实时/拖动路径不再
    // 往 Session 追加（否则会与预加载的整场数据重复），报告始终反映整个文件。
    private bool _sessionPreloaded;

    // 融合的两个输入，各自异步到达（CSI 来自脑电包，SPI 来自血氧指标），故各存一份最近有效值
    private double _latestCsi = double.NaN;
    private double _latestSpi = double.NaN;

    // 血氧侧待渲染数据：后台线程写、UI 线程在 RenderTick 里取走。只保留最新一份，
    // 慢于产出速率时自然丢弃中间帧，不会积压。
    private double[]? _pendingWave;
    private PpgMetrics? _pendingMetrics;

    // 图表脏标志：Apply/UpdateCharts 只置位、不直接 InvalidatePlot；
    // 由 RenderTick（25Hz）统一刷新。回放 32× 时每秒几十包，逐包重绘会闪，合并后封顶 25Hz。
    private bool _eegDirty, _trendDirty, _dsaDirty;

    // 回放总时长（秒），用于把趋势 X 轴固定成 0→整场，拖动进度条后不从头重画。
    private double _playbackDurationSec;

    // 录制脉搏波：串口线程把原始 IR/RED 推入，Apply（UI 线程）在写每个 NSM 包时取空并附上。
    private readonly List<int> _recIr = new(256);
    private readonly List<int> _recRed = new(256);
    private readonly object _recWaveLock = new();

    private readonly LineSeries _eegSeries;
    private readonly LineSeries _ppgSeries;
    private readonly LineSeries _csiTrend;
    private readonly LineSeries _noxTrend;
    private readonly LineSeries _sefTrend;
    private readonly LineSeries _spiTrend;
    private readonly LineSeries _fusionTrend;

    private readonly HeatMapSeries _dsaSeries;
    private readonly List<byte[]> _dsaColumns = new(DSA_COLS);

    public MainViewModel(NsmSerialService serial, NsmSimulatorService simulator,
        NsmPlaybackService playback, NsmRecordingService recorder, ILogger<MainViewModel> logger,
        NsmConfig config, PpgSerialService ppgSerial, PpgSimulatorService ppgSimulator,
        SessionBuffer session)
    {
        _serial = serial;
        _simulator = simulator;
        _playback = playback;
        _recorder = recorder;
        _logger = logger;
        _source = serial;
        Session = session;

        _config = config;
        _ppgSerial = ppgSerial;
        _ppgSimulator = ppgSimulator;
        _ppgSource = ppgSerial;
        _oximetry = new PulseOximetryProcessor(config);
        _spiCalc = new SpiCalculator(config);
        FusionWeightLabel = config.Fusion.WeightLabel;
        Spo2SourceLabel = Spo2Calibration.SourceLabel(config.Spo2);

        _eegSeries = new LineSeries { Color = OxyColor.FromRgb(0x2F, 0xE6, 0xD6), StrokeThickness = 1.3 };
        _ppgSeries = new LineSeries { Color = OxyColor.FromRgb(0xFB, 0x71, 0x85), StrokeThickness = 1.5 };
        _csiTrend  = new LineSeries { Title = "CSI", Color = OxyColor.FromRgb(0xA7, 0x8B, 0xFA), StrokeThickness = 1.8 };
        _noxTrend  = new LineSeries { Title = "NOX", Color = OxyColor.FromRgb(0x2D, 0xD4, 0xBF), StrokeThickness = 1.6 };
        _sefTrend  = new LineSeries { Title = "SEF95", Color = OxyColor.FromRgb(0xF5, 0xC2, 0x42), StrokeThickness = 1.4, LineStyle = LineStyle.Dash };
        _spiTrend  = new LineSeries { Title = "SPI", Color = OxyColor.FromRgb(0xFB, 0x92, 0x3C), StrokeThickness = 1.6 };
        _fusionTrend = new LineSeries { Title = "融合", Color = OxyColor.FromRgb(0x81, 0x8C, 0xF8), StrokeThickness = 2.0 };

        _dsaSeries = new HeatMapSeries
        {
            X0 = 0, X1 = 1, Y0 = 1, Y1 = DSA_BINS,
            Interpolate = true,
            RenderMethod = HeatMapRenderMethod.Bitmap,
            Data = new double[1, DSA_BINS],
        };

        EegModel = BuildEegModel();
        PpgModel = BuildPpgModel();
        TrendModel = BuildTrendModel();
        DsaModel = BuildDsaModel();

        RefreshPorts();
        RefreshPpgPorts();
        WireSource(_serial);
        WireSource(_simulator);
        WireSource(_playback);
        WirePpgSource(_ppgSerial);
        WirePpgSource(_ppgSimulator);
        // 血氧侧一律「放下就走」：处理器回调只把最新结果存到字段，由 UI 定时器（RenderTick）取走。
        // 波形每 8 帧回调一次（实测 87 Hz → 约 11 次/秒），逐次往 UI 线程推会让串口线程反复阻塞，
        // 排队积压后表现为整个界面卡死。改成轮询后串口线程完全不接触 UI。
        _oximetry.MetricsReady += m => Interlocked.Exchange(ref _pendingMetrics, m);
        _oximetry.WaveformReady += w => Interlocked.Exchange(ref _pendingWave, w);

        // 配置里预选了端口就填进下拉框，省得每次开机都手动选
        if (!string.IsNullOrWhiteSpace(_config.Ppg.Port)) SelectedPpgPort = _config.Ppg.Port;
        if (!string.IsNullOrWhiteSpace(_config.Nsm.Port)) SelectedPort = _config.Nsm.Port;
        _playback.PlaybackCompleted += () => RunOnUI(OnPlaybackCompleted);
        _playback.ProgressChanged += p => RunOnUI(() => OnPlaybackProgress(p));
        _playback.Seeked += (pkts, idx) => RunOnUI(() => RebuildFromHistory(pkts, idx));

        // 调试/演示辅助：NSM_PLAYBACK=<文件> 直接进回放模式（优先于 AUTOCONNECT），验证趋势/报告用。
        var pbFile = Environment.GetEnvironmentVariable("NSM_PLAYBACK");
        if (!string.IsNullOrWhiteSpace(pbFile) && File.Exists(pbFile))
        {
            SelectedSourceMode = SourceMode.Playback;
            PlaybackFile = pbFile;
            PlaybackSpeed = double.TryParse(Environment.GetEnvironmentVariable("NSM_SPEED"), out var sp) ? sp : 8;
            ToggleConnection();
        }
        // NSM_AUTOCONNECT=1 时自动连上两路。血氧走哪一路看配置。
        else if (Environment.GetEnvironmentVariable("NSM_AUTOCONNECT") == "1")
        {
            ToggleConnection();
            SelectedPpgSourceMode = string.IsNullOrWhiteSpace(_config.Ppg.Port)
                ? PpgSourceMode.Simulator
                : PpgSourceMode.Serial;
            TogglePpgConnection();
        }
    }

    // ─────────────────────────── 图表 ───────────────────────────
    public PlotModel EegModel { get; }
    public PlotModel PpgModel { get; }
    public PlotModel TrendModel { get; }
    public PlotModel DsaModel { get; }

    /// <summary>整场会话缓冲，供报告弹窗读取。</summary>
    public SessionBuffer Session { get; }

    /// <summary>趋势卡副标题：实时滚动 5 分钟 / 回放显示整场。</summary>
    [ObservableProperty] private string _trendSubtitle = "CSI / SPI / 融合 / NOX / SEF95 — 近 5 分钟";

    /// <summary>当前 EEG 滚动缓冲的样本（供成分分离弹窗读取）。</summary>
    public double[] GetEegSamples()
    {
        var pts = _eegSeries.Points;
        var s = new double[pts.Count];
        for (int i = 0; i < pts.Count; i++) s[i] = pts[i].Y;
        return s;
    }

    /// <summary>当前采样率（Hz），未知时回退到 100。</summary>
    public int CurrentSampleRate => SampleRate > 0 ? SampleRate : 100;

    /// <summary>秒数 → 手术计时 mm:ss（≥1 小时显示 h:mm:ss）。负值加前导减号，用于"距现在"。</summary>
    public static string TimeLabel(double sec)
    {
        bool neg = sec < 0;
        var t = TimeSpan.FromSeconds(Math.Abs(sec));
        string s = t.TotalHours >= 1 ? $"{(int)t.TotalHours}:{t.Minutes:00}:{t.Seconds:00}" : $"{t.Minutes:00}:{t.Seconds:00}";
        return neg ? "-" + s : s;
    }

    private static readonly OxyColor AxisTick = OxyColor.FromRgb(0x3D, 0x5A, 0x7A);
    private static readonly OxyColor AxisGrid = OxyColor.FromRgb(0x18, 0x2C, 0x46);

    /// <summary>
    /// 滚动波形（EEG/脉搏）的底部时间轴：右缘为 0（现在），向左为负秒。
    /// 刻度只显示整数秒（单位在轴标题里，避免每格重复 "s"）。
    /// <paramref name="fixedRate"/> &gt; 0 时用该采样率换算（脉搏波为 125Hz），否则用当前脑电采样率。
    /// </summary>
    private LinearAxis RollingTimeAxis(string? title = null, double fixedRate = 0)
    {
        var ax = new LinearAxis
        {
            Position = AxisPosition.Bottom, IsAxisVisible = true,
            FontSize = 9, TextColor = AxisTick, TicklineColor = AxisGrid,
            Title = title, TitleColor = AxisTick, TitleFontSize = 9,
            MajorGridlineStyle = LineStyle.None, MinorTickSize = 0,
        };
        // 右缘=0 秒，向左递减；样本索引 → 秒用采样率换算，只留整数秒
        ax.LabelFormatter = v =>
        {
            double rate = fixedRate > 0 ? fixedRate : Math.Max(1, CurrentSampleRate);
            return $"{(v - ax.ActualMaximum) / rate:0}";
        };
        return ax;
    }

    private PlotModel BuildEegModel()
    {
        var m = new PlotModel { PlotMargins = new OxyThickness(40, 2, 6, 18), Padding = new OxyThickness(0), Background = OxyColors.Transparent };
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left, Minimum = -130, Maximum = 130,
            Title = "μV", FontSize = 9, TitleFontSize = 9,
            MajorStep = 100, MinorStep = 50, StringFormat = "0.#",
            TextColor = AxisTick, TitleColor = AxisTick, TicklineColor = AxisGrid,
            MajorGridlineStyle = LineStyle.None,
        });
        m.Axes.Add(RollingTimeAxis("时间 (s)"));
        m.Series.Add(_eegSeries);
        return m;
    }

    /// <summary>
    /// 脉搏波图：IR 通道去基线后的交流分量，幅度随灌注变化很大，
    /// 故 Y 轴不设固定量程，由 <see cref="UpdatePpgWave"/> 按当前窗口自适应。
    /// </summary>
    private PlotModel BuildPpgModel()
    {
        var m = new PlotModel { PlotMargins = new OxyThickness(10, 2, 6, 18), Padding = new OxyThickness(0), Background = OxyColors.Transparent };
        // Y：去基线交流分量，幅度是相对值（ADC 计数），标注"相对幅度"而不给会误导的绝对刻度
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -1, Maximum = 1, IsAxisVisible = false });
        // 脉搏波 125Hz（PPG_WINDOW=500≈4s），时间换算须用该率而非脑电率
        m.Axes.Add(RollingTimeAxis("时间 (s)", _config.Ppg.SampleRate > 0 ? _config.Ppg.SampleRate : 125));
        m.Series.Add(_ppgSeries);
        return m;
    }

    /// <summary>
    /// 用最近一段 IR 交流分量刷新脉搏波（阶段 3 由 PPG 服务调用）。
    /// Y 轴按窗口内峰峰值自适应，弱灌注时波形也能看清。
    /// </summary>
    public void UpdatePpgWave(double[] acSamples)
    {
        _ppgSeries.Points.Clear();
        if (acSamples.Length == 0) { PpgModel.InvalidatePlot(true); return; }

        int n = Math.Min(acSamples.Length, PPG_WINDOW);
        int from = acSamples.Length - n;
        double lo = double.MaxValue, hi = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double v = acSamples[from + i];
            _ppgSeries.Points.Add(new DataPoint(i, v));
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }

        double pad = Math.Max((hi - lo) * 0.12, 1e-6);
        PpgModel.Axes[0].Minimum = lo - pad;
        PpgModel.Axes[0].Maximum = hi + pad;
        PpgModel.Axes[1].Minimum = 0;
        PpgModel.Axes[1].Maximum = Math.Max(1, n - 1);
        PpgModel.InvalidatePlot(true);
    }

    private PlotModel BuildTrendModel()
    {
        var m = new PlotModel
        {
            PlotMargins = new OxyThickness(34, 4, 4, 22),
            Padding = new OxyThickness(0),
            Background = OxyColors.Transparent,
            TextColor = OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
        };
        m.Legends.Add(new Legend
        {
            LegendPosition = LegendPosition.TopRight,
            LegendTextColor = OxyColor.FromRgb(0xC8, 0xDC, 0xF0),
            LegendFontSize = 11,
        });
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left, Minimum = 0, Maximum = 100,
            MajorGridlineStyle = LineStyle.Solid, StringFormat = "0.#",
            MajorGridlineColor = OxyColor.FromRgb(0x18, 0x2C, 0x46),
            TextColor = OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            TicklineColor = OxyColor.FromRgb(0x18, 0x2C, 0x46),
        });
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom, Minimum = 0, Maximum = TREND_WINDOW_SEC,
            FontSize = 10, LabelFormatter = TimeLabel,
            TextColor = OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            TicklineColor = OxyColor.FromRgb(0x18, 0x2C, 0x46),
        });
        m.Series.Add(_csiTrend);
        m.Series.Add(_spiTrend);
        m.Series.Add(_fusionTrend);
        m.Series.Add(_noxTrend);
        m.Series.Add(_sefTrend);
        return m;
    }

    private PlotModel BuildDsaModel()
    {
        var m = new PlotModel
        {
            PlotMargins = new OxyThickness(34, 4, 4, 22),
            Padding = new OxyThickness(0),
            Background = OxyColors.Transparent,
            TextColor = OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
        };
        // 颜色轴：0-255 强度 → 频谱配色（蓝→青→绿→黄→红）
        m.Axes.Add(new LinearColorAxis
        {
            Position = AxisPosition.Right,
            Palette = OxyPalettes.Jet(256),
            Minimum = 0, Maximum = 255,
            IsAxisVisible = false,
            LowColor = OxyColors.Transparent,
        });
        // Y 轴：频率 1-44 Hz
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left, Minimum = 1, Maximum = DSA_BINS,
            Title = "Hz", FontSize = 10, TitleFontSize = 10,
            MajorStep = 10, MinorStep = 5, StringFormat = "0.#",
            TextColor = OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            TicklineColor = OxyColor.FromRgb(0x18, 0x2C, 0x46),
        });
        // X 轴：时间列（每列≈1 包≈1 秒），显示"距现在"相对时间
        var dx = new LinearAxis
        {
            Position = AxisPosition.Bottom, Minimum = 0, Maximum = DSA_COLS,
            FontSize = 9, TextColor = AxisTick, TicklineColor = AxisGrid,
            MajorGridlineStyle = LineStyle.None, MinorTickSize = 0,
        };
        dx.LabelFormatter = v => TimeLabel(v - dx.ActualMaximum);   // 右缘 0，向左为负 mm:ss
        m.Axes.Add(dx);
        m.Series.Add(_dsaSeries);
        return m;
    }

    // ─────────────────────────── 连接控制 ───────────────────────────
    public record SourceOption(SourceMode Mode, string Name)
    {
        public override string ToString() => Name;
    }
    public IReadOnlyList<SourceOption> SourceModes { get; } = new[]
    {
        new SourceOption(SourceMode.Simulator, "内置模拟器"),
        new SourceOption(SourceMode.Serial, "真实串口"),
        new SourceOption(SourceMode.Playback, "文件回放"),
    };

    [ObservableProperty] private ObservableCollection<string> _availablePorts = new();
    [ObservableProperty] private string? _selectedPort;
    [ObservableProperty] private SourceMode _selectedSourceMode = SourceMode.Simulator;
    [ObservableProperty] private bool _isConnected;
    [ObservableProperty] private bool _hasData;
    [ObservableProperty] private string _statusMessage = "就绪 — 选择数据源后点击连接";
    [ObservableProperty] private string _sourceName = "未连接";
    [ObservableProperty] private string _emptyHint = "等待数据 · 请选择数据源并连接";
    [ObservableProperty] private int _sampleRate;
    [ObservableProperty] private string _currentTime = DateTime.Now.ToString("HH:mm:ss");
    [ObservableProperty] private string? _playbackFile;

    // ── 回放进度与倍率 ──
    [ObservableProperty] private double _playbackPosition;
    [ObservableProperty] private double _playbackLength;
    [ObservableProperty] private string _playbackTimeDisplay = "00:00 / 00:00";
    [ObservableProperty] private double _playbackSpeed = 1.0;
    private bool _suppressSeek;

    public record SpeedOption(double Value, string Name)
    {
        public override string ToString() => Name;
    }
    public IReadOnlyList<SpeedOption> PlaybackSpeeds { get; } = new[]
    {
        new SpeedOption(0.5, "0.5×"), new SpeedOption(1, "1×"), new SpeedOption(2, "2×"),
        new SpeedOption(4, "4×"), new SpeedOption(8, "8×"), new SpeedOption(16, "16×"), new SpeedOption(32, "32×"),
    };

    public bool IsSerialMode => SelectedSourceMode == SourceMode.Serial;
    public bool IsPlaybackMode => SelectedSourceMode == SourceMode.Playback;
    public string PlaybackFileDisplay =>
        string.IsNullOrEmpty(PlaybackFile) ? "（未选择文件）" : Path.GetFileName(PlaybackFile);

    partial void OnPlaybackPositionChanged(double value)
    {
        if (_suppressSeek) return;                       // 来自回放进度的程序化更新，不触发跳转
        if (SelectedSourceMode == SourceMode.Playback && IsConnected)
            _playback.SeekToIndex((int)Math.Round(value));
    }

    partial void OnPlaybackSpeedChanged(double value) => _playback.Speed = value;

    private void OnPlaybackProgress(PlaybackProgress p)
    {
        _suppressSeek = true;
        PlaybackLength = Math.Max(1, p.Total - 1);
        PlaybackPosition = p.Index;
        _suppressSeek = false;
        _playbackDurationSec = p.Duration.TotalSeconds;   // 供趋势 X 轴固定为整场
        PlaybackTimeDisplay = $"{FormatClock(p.Elapsed)} / {FormatClock(p.Duration)}";
    }

    private static string FormatClock(TimeSpan t) =>
        t.TotalHours >= 1 ? $"{(int)t.TotalHours}:{t.Minutes:00}:{t.Seconds:00}" : $"{t.Minutes:00}:{t.Seconds:00}";

    partial void OnSelectedSourceModeChanged(SourceMode value)
    {
        OnPropertyChanged(nameof(IsSerialMode));
        OnPropertyChanged(nameof(IsPlaybackMode));
        if (IsConnected) return;
        StatusMessage = value switch
        {
            SourceMode.Simulator => "已选择内置模拟器",
            SourceMode.Serial => "已选择真实串口",
            SourceMode.Playback => "已选择文件回放",
            _ => StatusMessage,
        };
    }

    partial void OnPlaybackFileChanged(string? value) => OnPropertyChanged(nameof(PlaybackFileDisplay));

    [RelayCommand]
    private void RefreshPorts()
    {
        var ports = _serial.GetAvailablePorts().ToList();
        AvailablePorts = new ObservableCollection<string>(ports);
        if (SelectedPort == null || !ports.Contains(SelectedPort))
            SelectedPort = ports.FirstOrDefault();
    }

    // ── PPG 血氧模组：独立串口，与脑电互不影响（任一路断开不牵连另一路）──
    public record PpgSourceOption(PpgSourceMode Mode, string Name)
    {
        public override string ToString() => Name;
    }
    public IReadOnlyList<PpgSourceOption> PpgSourceModes { get; } = new[]
    {
        new PpgSourceOption(PpgSourceMode.Serial, "指夹模组"),
        new PpgSourceOption(PpgSourceMode.Simulator, "模拟器"),
    };

    [ObservableProperty] private ObservableCollection<string> _ppgPorts = new();
    [ObservableProperty] private string? _selectedPpgPort;
    [ObservableProperty] private PpgSourceMode _selectedPpgSourceMode = PpgSourceMode.Serial;

    public bool IsPpgSerialMode => SelectedPpgSourceMode == PpgSourceMode.Serial;
    partial void OnSelectedPpgSourceModeChanged(PpgSourceMode value) => OnPropertyChanged(nameof(IsPpgSerialMode));

    [RelayCommand]
    private void RefreshPpgPorts()
    {
        var ports = System.IO.Ports.SerialPort.GetPortNames().OrderBy(p => p).ToList();
        PpgPorts = new ObservableCollection<string>(ports);
        if (SelectedPpgPort == null || !ports.Contains(SelectedPpgPort))
            SelectedPpgPort = ports.FirstOrDefault();
    }

    [RelayCommand]
    private void TogglePpgConnection()
    {
        if (IsPpgConnected) { DisconnectPpg(); return; }

        _ppgSource = SelectedPpgSourceMode == PpgSourceMode.Simulator ? _ppgSimulator : _ppgSerial;

        if (SelectedPpgSourceMode == PpgSourceMode.Serial && string.IsNullOrEmpty(SelectedPpgPort))
        {
            PpgStatusMessage = "请先选择血氧串口";
            return;
        }

        _oximetry.Reset();
        _spiCalc.Reset();
        ClearPpgReadouts();

        bool ok = _ppgSource.Connect(SelectedPpgPort ?? "SIM", _config.Ppg.Baud);
        if (!ok) { PpgStatusMessage = "血氧连接失败"; return; }

        IsPpgConnected = true;
        PpgStatusMessage = _ppgSource.SourceName;
    }

    private void DisconnectPpg()
    {
        _ppgSource.Disconnect();
        IsPpgConnected = false;
        PpgStatusMessage = "血氧未连接";
        ClearPpgReadouts();
        UpdatePpgWave(Array.Empty<double>());
    }

    /// <summary>断开或重连时把血氧派生的读数全部复位，避免陈旧值留在屏幕上被误读。</summary>
    private void ClearPpgReadouts()
    {
        Spo2Value = double.NaN;      Spo2Display = "--";      Spo2Zone = "未标定 · 仅供参考";
        PulseRateValue = double.NaN; PulseRateDisplay = "--"; PulseRateZone = "";
        SpiValue = double.NaN;       SpiDisplay = "---";      SpiZone = "";
        PerfusionDisplay = "--";     PrvDisplay = "--";       RRatioDisplay = "--";
        PpgLinkDisplay = "--";
        _latestSpi = double.NaN;
        RecomputeFusion();
    }

    [RelayCommand]
    private void BrowsePlaybackFile()
    {
        var dlg = new OpenFileDialog
        {
            Title = "选择回放文件",
            Filter = "NSM 录制文件 (*.nsm)|*.nsm|所有文件 (*.*)|*.*",
            InitialDirectory = RecordingsDir(),
        };
        if (dlg.ShowDialog() == true) PlaybackFile = dlg.FileName;
    }

    [RelayCommand]
    private void ToggleConnection()
    {
        if (IsConnected) { Disconnect(); return; }

        _source = SelectedSourceMode switch
        {
            SourceMode.Simulator => _simulator,
            SourceMode.Serial => _serial,
            SourceMode.Playback => _playback,
            _ => _simulator,
        };

        if (SelectedSourceMode == SourceMode.Playback)
        {
            _playback.Speed = PlaybackSpeed;
            _suppressSeek = true;
            PlaybackPosition = 0;
            PlaybackLength = 1;
            _suppressSeek = false;
            PlaybackTimeDisplay = "00:00 / 00:00";
            // 回放的血氧来自脉搏波重建，先清干净；有脉搏波时 ReplayPulseWave 会点亮
            IsPpgConnected = false;
            _oximetry.Reset();
            _spiCalc.Reset();
            ClearPpgReadouts();
            _pulseBaseline = 0;
            _pulseDecim = 0;
        }

        bool ok = SelectedSourceMode switch
        {
            SourceMode.Simulator => _source.Connect("SIM"),
            SourceMode.Serial when string.IsNullOrEmpty(SelectedPort) => Fail("请先选择串口"),
            SourceMode.Serial => _source.Connect(SelectedPort!, 115200),
            SourceMode.Playback when string.IsNullOrEmpty(PlaybackFile) => Fail("请先选择回放文件"),
            SourceMode.Playback => _source.Connect(PlaybackFile!),
            _ => false,
        };

        if (ok)
        {
            IsConnected = true;
            HasData = false;
            EmptyHint = SelectedSourceMode == SourceMode.Playback ? "正在加载回放…" : "等待设备数据…";
            SourceName = _source.SourceName;
            ResetCharts();

            _playbackDurationSec = 0;
            TrendSubtitle = IsPlaybackMode
                ? "CSI / SPI / 融合 / NOX / SEF95 — 整场"
                : "CSI / SPI / 融合 / NOX / SEF95 — 近 5 分钟";
            Session.Reset(_source.SourceName, IsPlaybackMode);

            // 回放：整场数据在连接时一次性预加载进 Session，报告立即反映整个文件（不受回放/拖动进度影响）。
            // 之后实时回放/拖动只驱动上屏，不再往 Session 追加（见 _sessionPreloaded 的门控）。
            _sessionPreloaded = false;
            if (IsPlaybackMode)
            {
                PreloadPlaybackSession(_playback.Packets);
                _sessionPreloaded = true;
            }

            // 需求2：直接接入真实 NSM 串口时自动开录（模拟器/回放不录）。
            // 调试：NSM_AUTORECORD=1 时任意已连接模式都开录，供无硬件验证录制含脉搏波。
            bool autoRec = (SelectedSourceMode == SourceMode.Serial && _config.Recording.AutoRecord)
                           || Environment.GetEnvironmentVariable("NSM_AUTORECORD") == "1";
            if (autoRec && !IsRecording)
                StartRecording(AutoRecordPath());
        }
    }

    private bool Fail(string msg) { StatusMessage = msg; return false; }

    private void Disconnect()
    {
        if (IsRecording) StopRecording();
        bool wasPlayback = IsPlaybackMode;
        _source.Disconnect();
        IsConnected = false;
        HasData = false;
        EmptyHint = "已断开 · 请重新连接数据源";
        SourceName = "未连接";
        // CSI 随脑电一起失效；血氧仍可能在跑，但融合缺了一路就得显示 "---"
        _latestCsi = double.NaN;
        RecomputeFusion();
        // 回放模式下的血氧是脉搏波重建来的，一并收回
        if (wasPlayback && IsPpgConnected)
        {
            IsPpgConnected = false;
            ClearPpgReadouts();
            UpdatePpgWave(Array.Empty<double>());
        }
    }

    private void OnPlaybackCompleted()
    {
        if (IsRecording) StopRecording();
        IsConnected = false;
        SourceName = "未连接";
        EmptyHint = "回放结束 · 可重新选择文件";
        // 保留最后画面；血氧连接标志复位，避免下次误判
        if (IsPpgConnected) IsPpgConnected = false;
    }

    // ─────────────────────────── 录制 ───────────────────────────
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _recordingStatus = "";
    [ObservableProperty] private long _recordedPackets;

    [RelayCommand]
    private void ToggleRecording()
    {
        if (IsRecording) { StopRecording(); return; }
        if (!IsConnected) { StatusMessage = "请先连接数据源再开始录制"; return; }

        var dlg = new SaveFileDialog
        {
            Title = "保存录制文件",
            Filter = "NSM 录制文件 (*.nsm)|*.nsm",
            FileName = $"NSM_{DateTime.Now:yyyyMMdd_HHmmss}.nsm",
            InitialDirectory = RecordingsDir(),
        };
        if (dlg.ShowDialog() != true) return;
        StartRecording(dlg.FileName);
    }

    /// <summary>开始录制到指定文件。手动（对话框）与自动（连上真实设备）共用。</summary>
    private void StartRecording(string path)
    {
        lock (_recWaveLock) { _recIr.Clear(); _recRed.Clear(); }
        _recorder.Start(path);
        IsRecording = true;
        RecordedPackets = 0;
        RecordingStatus = $"● 录制中 — {Path.GetFileName(path)}";
        StatusMessage = $"自动录制：{Path.GetFileName(path)}";
    }

    private void StopRecording()
    {
        _recorder.Stop();
        IsRecording = false;
        RecordingStatus = $"已保存 {RecordedPackets} 包";
    }

    /// <summary>取空脉搏波缓存并附到 NSM 包上。无脉搏仪数据时原样返回。</summary>
    private NSMDataPacket AttachPulseWave(NSMDataPacket pkt)
    {
        if (!_config.Recording.IncludePulseWave) return pkt;

        int[] ir, red;
        lock (_recWaveLock)
        {
            if (_recIr.Count == 0) return pkt;
            ir = _recIr.ToArray();
            red = _recRed.ToArray();
            _recIr.Clear();
            _recRed.Clear();
        }

        int irDc = (int)Math.Round(ir.Average());
        int redDc = red.Length > 0 ? (int)Math.Round(red.Average()) : 0;
        var irAc = new short[ir.Length];
        var redAc = new short[red.Length];
        for (int i = 0; i < ir.Length; i++) irAc[i] = ClampShort(ir[i] - irDc);
        for (int i = 0; i < red.Length; i++) redAc[i] = ClampShort(red[i] - redDc);

        return pkt with { PulseWaveIr = irAc, PulseWaveRed = redAc, IrDc = irDc, RedDc = redDc };
    }

    private static short ClampShort(long v) => (short)Math.Clamp(v, short.MinValue, short.MaxValue);

    /// <summary>回放时把包内脉搏波（AC+DC 还原绝对值）逐样本走 OnRawPulse，重建血氧派生量与整场脉搏波。</summary>
    private void ReplayPulseWave(NSMDataPacket pkt)
    {
        if (pkt.PulseWaveIr == null) return;
        var ir = pkt.PulseWaveIr;
        var red = pkt.PulseWaveRed;
        if (!IsPpgConnected) IsPpgConnected = true;   // 让脉搏波卡与血氧格生效
        for (int i = 0; i < ir.Length; i++)
        {
            int irAbs = ir[i] + pkt.IrDc;
            int redAbs = (red != null && i < red.Length ? red[i] : 0) + pkt.RedDc;
            OnRawPulse(irAbs, redAbs, forRecord: false, pkt.LocalTimestamp);   // 用包时间，血氧才与趋势同轴
        }
    }

    /// <summary>
    /// 回放连接时一次性把整场数据灌入 <see cref="Session"/>：脑电 / 趋势(CSI·SPI·NOX·融合·SEF95·BS) /
    /// 频带占比 / DSA / 事件，以及（文件含脉搏波时）重建的血氧·脉搏·SPI 与脉搏波。
    ///
    /// 目的：报告「回放直接展示所有数据」—— 打开即见整个文件，不受实时回放推进或拖动进度限制。
    /// 灌完后置 <c>_sessionPreloaded=true</c>，实时/拖动路径的 <c>Session.Add*</c> 全部关闭以免重复。
    /// 计算口径与 <see cref="Apply"/> / <see cref="ApplyPpgMetrics"/> 逐条对齐；血氧/SPI 用<b>独立</b>
    /// 处理器实例重建，绝不触碰实时链路的 <c>_oximetry</c>/<c>_spiCalc</c> 状态。
    /// </summary>
    private void PreloadPlaybackSession(IReadOnlyList<NSMDataPacket> packets)
    {
        if (packets.Count == 0) return;

        var oxi = new PulseOximetryProcessor(_config);
        var spiCalc = new SpiCalculator(_config);
        double latestSpi = double.NaN;
        double pulseBaseline = 0; int pulseDecim = 0;

        // 血氧指标在 Push 内同步回调：复刻 ApplyPpgMetrics 的低灌注门控与设备/上位机取值口径
        oxi.MetricsReady += m =>
        {
            double spo2;
            if (m.LowPerfusion) spo2 = double.NaN;
            else
            {
                bool useDevice = _config.Ppg.PreferDeviceValues && m.DeviceSpo2 is > 0 and <= 100;
                double cand = useDevice ? m.DeviceSpo2 : m.Spo2;
                spo2 = double.IsFinite(cand) ? cand : double.NaN;
            }

            double pr = m.LowPerfusion ? double.NaN
                      : _config.Ppg.PreferDeviceValues && m.DeviceHr > 0 ? m.DeviceHr
                      : double.IsFinite(m.PulseRate) ? m.PulseRate
                      : m.DeviceHr > 0 ? m.DeviceHr : double.NaN;

            double spi = spiCalc.Update(m.PerfusionIndex, pr);   // pr 低灌注时已为 NaN，坏样本不污染 SPI 基线
            latestSpi = double.IsFinite(spi) ? spi : double.NaN;
            Session.AddVital(m.Time, spo2, pr, m.PerfusionIndex);
        };

        int curEvent = 0;
        double dPct = 0, tPct = 0, aPct = 0, bPct = 0, gPct = 0;   // linSum<=0 时沿用上一次，与 Apply 一致

        foreach (var pkt in packets)
        {
            // 1. 脉搏波重建（含则有）→ 血氧/脉搏/SPI + 整场脉搏波（口径同 ReplayPulseWave + OnRawPulse）
            var ir = pkt.PulseWaveIr;
            if (ir != null)
            {
                var red = pkt.PulseWaveRed;
                for (int i = 0; i < ir.Length; i++)
                {
                    int irAbs = ir[i] + pkt.IrDc;
                    int redAbs = (red != null && i < red.Length ? red[i] : 0) + pkt.RedDc;
                    oxi.Push(new PpgFrame(pkt.LocalTimestamp, irAbs, redAbs, 0, 0));
                    pulseBaseline = pulseBaseline == 0 ? irAbs : pulseBaseline + (irAbs - pulseBaseline) * 0.02;
                    if (++pulseDecim >= PulseSessionDecim) { pulseDecim = 0; Session.AddPulseSample(irAbs - pulseBaseline, 25); }
                }
            }

            // 2. CSI / NOX / BS / 融合（口径同 Apply）
            double csi = pkt.CSIValid && pkt.CSI <= 99 ? pkt.CSI : double.NaN;
            double nox = pkt.NOXValid && pkt.NOX <= 99 ? pkt.NOX : double.NaN;
            double bs  = pkt.BSValid ? pkt.BS : double.NaN;
            double fusion = IndexFusion.Combine(csi, latestSpi, _config.Fusion);

            // 3. 频带占比：dB → 线性功率 → 归一（含 γ）
            double linDelta = Math.Pow(10, pkt.DeltaPowerDb / 10.0);
            double linTheta = Math.Pow(10, pkt.ThetaPowerDb / 10.0);
            double linAlpha = Math.Pow(10, pkt.AlphaPowerDb / 10.0);
            double linBeta  = Math.Pow(10, pkt.BetaPowerDb / 10.0);
            double linGamma = Math.Pow(10, pkt.GammaPowerDb / 10.0);
            double linSum = linDelta + linTheta + linAlpha + linBeta + linGamma;
            if (linSum > 0)
            {
                dPct = 100 * linDelta / linSum; tPct = 100 * linTheta / linSum;
                aPct = 100 * linAlpha / linSum; bPct = 100 * linBeta / linSum; gPct = 100 * linGamma / linSum;
            }

            // 4. 事件：编号变化时记一条（复刻 HandleEvent 的 AddEvent 部分）
            if (pkt.EventNumber <= 0) curEvent = 0;
            else if (pkt.EventNumber != curEvent)
            {
                curEvent = pkt.EventNumber;
                Session.AddEvent(pkt.LocalTimestamp, $"#{pkt.EventNumber} {EventLabel(pkt.EventType)}");
            }

            // 5. 灌入整场缓冲
            Session.AddEegSamples(pkt.EEGSamplesUv, pkt.LocalTimestamp);
            Session.AddTrend(pkt.LocalTimestamp, csi, latestSpi, nox, fusion, pkt.SEF95, bs);
            Session.AddBands(pkt.LocalTimestamp, dPct, tPct, aPct, bPct, gPct);
            if (pkt.Dsa.Length == DSA_BINS) Session.AddDsa(pkt.Dsa);
        }
    }

    private string RecordingsDir()
    {
        var dir = string.IsNullOrWhiteSpace(_config.Recording.Directory)
            ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Recordings")
            : _config.Recording.Directory;
        Directory.CreateDirectory(dir);
        return dir;
    }

    private string AutoRecordPath() =>
        Path.Combine(RecordingsDir(), $"NSM_{DateTime.Now:yyyyMMdd_HHmmss}.nsm");

    // ─────────────────────────── 手动事件标注 ───────────────────────────
    public record EventTypeOption(NSMEventType Type, string Name)
    {
        public override string ToString() => Name;
    }
    public IReadOnlyList<EventTypeOption> EventTypes { get; } =
        Enum.GetValues<NSMEventType>().Select(t => new EventTypeOption(t, EventLabel(t))).ToList();

    [ObservableProperty] private NSMEventType _selectedAnnotationType = NSMEventType.Note;
    [ObservableProperty] private string _annotationNote = "";
    private int _manualEventCounter;

    [RelayCommand]
    private void AddAnnotation()
    {
        var note = string.IsNullOrWhiteSpace(AnnotationNote) ? "" : $" — {AnnotationNote.Trim()}";
        var label = $"手动: {EventLabel(SelectedAnnotationType)}{note}";
        Events.Insert(0, new NsmEventVm(
            DateTime.Now.ToString("HH:mm:ss"),
            label,
            double.IsNaN(CsiValue) ? null : (int)CsiValue,
            isManual: true));
        while (Events.Count > 50) Events.RemoveAt(Events.Count - 1);
        _manualEventCounter++;
        AnnotationNote = "";
        StatusMessage = $"已标注：{EventLabel(SelectedAnnotationType)}";
    }

    // ─────────────────────────── 数值指标 ───────────────────────────
    [ObservableProperty] private double _csiValue = double.NaN;
    [ObservableProperty] private string _csiDisplay = "---";
    [ObservableProperty] private string _csiZone = "";
    [ObservableProperty] private double _noxValue = double.NaN;
    [ObservableProperty] private string _noxDisplay = "---";
    [ObservableProperty] private string _noxZone = "";
    [ObservableProperty] private double _bsValue = double.NaN;
    [ObservableProperty] private string _bsDisplay = "---";
    [ObservableProperty] private double _sqiValue = double.NaN;
    [ObservableProperty] private string _sqiDisplay = "---";
    [ObservableProperty] private double _emgValue = double.NaN;
    [ObservableProperty] private string _emgDisplay = "---";
    [ObservableProperty] private string _sef95Display = "--";
    [ObservableProperty] private string _eogDisplay = "--";
    [ObservableProperty] private double _sef95Value;
    [ObservableProperty] private double _eogValue;
    [ObservableProperty] private double _blackImpedanceValue;
    [ObservableProperty] private double _whiteImpedanceValue;

    [ObservableProperty] private double _deltaPower;
    [ObservableProperty] private double _thetaPower;
    [ObservableProperty] private double _alphaPower;
    [ObservableProperty] private double _betaPower;
    [ObservableProperty] private double _gammaPower;

    // 频带比值（线性域）：δ/α（DAR）、β/α
    [ObservableProperty] private string _deltaAlphaRatioDisplay = "--";
    [ObservableProperty] private string _betaAlphaRatioDisplay = "--";

    // 各分波功率占比（%），由 dB 还原为线性功率后归一
    [ObservableProperty] private double _deltaPct;
    [ObservableProperty] private double _thetaPct;
    [ObservableProperty] private double _alphaPct;
    [ObservableProperty] private double _betaPct;
    [ObservableProperty] private double _gammaPct;

    [ObservableProperty] private string _blackImpedanceDisplay = "--";
    [ObservableProperty] private string _whiteImpedanceDisplay = "--";
    [ObservableProperty] private bool _hasElectrodeWarning;
    [ObservableProperty] private string _electrodeWarning = "";

    // ───────── PPG（AFE4490 指夹血氧模组，独立串口）─────────
    // 阶段 1 仅占位：全部显示 "--"，阶段 2/3/4 分别接入通讯、SpO₂/脉搏、SPI/融合。
    [ObservableProperty] private double _spo2Value = double.NaN;
    [ObservableProperty] private string _spo2Display = "--";
    /// <summary>SpO₂ 状态词。未接入前提示曲线来源，接入后变为 "正常 · 厂商曲线" 等。</summary>
    [ObservableProperty] private string _spo2Zone = "非诊断用";
    /// <summary>标定曲线来源，显示在血氧明细卡的芯片上。</summary>
    [ObservableProperty] private string _spo2SourceLabel = "厂商曲线";
    [ObservableProperty] private double _pulseRateValue = double.NaN;
    [ObservableProperty] private string _pulseRateDisplay = "--";
    /// <summary>脉搏状态词（心动过缓 / 正常 / 心动过速），阶段 3 由 PR 计算。</summary>
    [ObservableProperty] private string _pulseRateZone = "";
    [ObservableProperty] private string _perfusionDisplay = "--";      // PI 灌注指数 %
    [ObservableProperty] private string _prvDisplay = "--";            // 脉率变异性 RMSSD ms
    [ObservableProperty] private string _rRatioDisplay = "--";         // 比值 R（SpO₂ 中间量，便于排障）
    [ObservableProperty] private string _ppgLinkDisplay = "--";        // 有效帧率，协议 §6.3 链路健康判据
    [ObservableProperty] private bool _isPpgConnected;
    [ObservableProperty] private string _ppgStatusMessage = "血氧未连接";

    // ───────── SPI 伤害感受指数（由 PPG 导出）─────────
    [ObservableProperty] private double _spiValue = double.NaN;
    [ObservableProperty] private string _spiDisplay = "---";
    [ObservableProperty] private string _spiZone = "";

    // ───────── CSI ⊕ SPI 融合指数（权重见 appsettings.json）─────────
    [ObservableProperty] private double _fusionValue = double.NaN;
    [ObservableProperty] private string _fusionDisplay = "---";
    [ObservableProperty] private string _fusionZone = "";
    /// <summary>融合权重说明，如 "0.6·CSI + 0.4·SPI"；阶段 2 接入配置后由配置驱动。</summary>
    [ObservableProperty] private string _fusionWeightLabel = "0.6·CSI + 0.4·SPI";

    public ObservableCollection<NsmEventVm> Events { get; } = new();

    // ─────────────────────────── 数据接收 ───────────────────────────
    private void WireSource(INsmDataSource src)
    {
        src.StatusChanged += msg => RunOnUI(() => StatusMessage = msg);
        src.ErrorOccurred += ex => RunOnUI(() => StatusMessage = $"错误：{ex.Message}");
        src.NSMDataReceived += OnPacket;
    }

    private void OnPacket(NSMDataPacket pkt) => RunOnUI(() => Apply(pkt));

    // ─────────────────────────── 血氧数据接收 ───────────────────────────
    private void WirePpgSource(IPpgDataSource src)
    {
        src.StatusChanged += msg => RunOnUI(() => PpgStatusMessage = msg);
        src.ErrorOccurred += ex => RunOnUI(() =>
        {
            PpgStatusMessage = $"血氧错误：{ex.Message}";
            _logger.LogWarning(ex, "血氧链路异常");
        });
        // 帧在串口线程上到达，走统一入口 OnRawPulse（实时与回放共用）。
        src.FrameReceived += f => OnRawPulse(f.Ir, f.Red, forRecord: true, f.Timestamp);
    }

    // 抽稀 + EMA 去基线，供整场报告的脉搏波包络（约 25Hz 足够，报告是压缩视图）
    private double _pulseBaseline;
    private int _pulseDecim;
    private const int PulseSessionDecim = 5;   // 125Hz → 25Hz

    /// <summary>
    /// 原始脉搏波单样本入口，实时（串口线程）与回放（UI 线程、由 <see cref="ReplayPulseWave"/> 调）共用。
    /// forRecord=true 表示是实时链路，可缓存供录制；回放不再回录。
    /// </summary>
    private void OnRawPulse(int ir, int red, bool forRecord, DateTime ts)
    {
        _oximetry.Push(new Models.PpgFrame(ts, ir, red, 0, 0));   // ts：实时=帧到达时刻，回放=包时间，保证指标与趋势同轴

        if (forRecord && IsRecording && _config.Recording.IncludePulseWave)
            lock (_recWaveLock) { _recIr.Add(ir); _recRed.Add(red); }

        // 整场脉搏波（去基线后抽稀存入会话缓冲）；回放已整场预加载，不再追加
        _pulseBaseline = _pulseBaseline == 0 ? ir : _pulseBaseline + (ir - _pulseBaseline) * 0.02;
        if (++_pulseDecim >= PulseSessionDecim)
        {
            _pulseDecim = 0;
            if (!_sessionPreloaded) Session.AddPulseSample(ir - _pulseBaseline, 25);
        }
    }

    private void ApplyPpgMetrics(PpgMetrics m)
    {
        if (!IsPpgConnected) return;
        EnsureTimeBase(m.Time);

        // 灌注不足（探头脱落 / 佩戴不良）时，所有派生值一律不显示，并说明原因 ——
        // 详见 PpgConfig.MinPerfusionIndex：此时 R 仍算得出，给数会误导。
        string lowHint = $"低灌注 · 检查探头 (PI {m.PerfusionIndex:0.00}%)";

        // SpO₂：默认用上位机计算值（设备侧字段实测恒为 0，见 PpgConfig.PreferDeviceValues）
        bool useDevice = _config.Ppg.PreferDeviceValues && m.DeviceSpo2 is > 0 and <= 100;
        double spo2 = useDevice ? m.DeviceSpo2 : m.Spo2;
        if (m.LowPerfusion)
        {
            Spo2Value = double.NaN; Spo2Display = "--"; Spo2Zone = lowHint;
        }
        else if (double.IsFinite(spo2))
        {
            Spo2Value = spo2;
            Spo2Display = $"{Math.Round(spo2)}";
            string band = spo2 >= 95 ? "正常" : spo2 >= 90 ? "偏低" : "低氧";
            Spo2Zone = useDevice ? $"{band} · 设备直出" : $"{band} · {Spo2SourceLabel}";
        }
        else { Spo2Value = double.NaN; Spo2Display = "--"; Spo2Zone = "信号不足"; }

        // 脉率：上位机由 IR 波形峰检测算出；设备侧走的是另一套算法（hr_algo，跑在 RED 上），仅作兜底。
        double pr = _config.Ppg.PreferDeviceValues && m.DeviceHr > 0
            ? m.DeviceHr
            : double.IsFinite(m.PulseRate) ? m.PulseRate : (m.DeviceHr > 0 ? m.DeviceHr : double.NaN);
        if (m.LowPerfusion)
        {
            pr = double.NaN;
            PulseRateValue = double.NaN; PulseRateDisplay = "--"; PulseRateZone = "低灌注";
        }
        else if (double.IsFinite(pr))
        {
            PulseRateValue = pr;
            PulseRateDisplay = $"{Math.Round(pr)}";
            PulseRateZone = pr < 50 ? "心动过缓" : pr <= 100 ? "正常" : "心动过速";
        }
        else { PulseRateValue = double.NaN; PulseRateDisplay = "--"; PulseRateZone = ""; }

        PerfusionDisplay = double.IsFinite(m.PerfusionIndex) && m.PerfusionIndex > 0 ? $"{m.PerfusionIndex:0.00}" : "--";
        PrvDisplay = double.IsFinite(m.Prv) ? $"{m.Prv:0}" : "--";
        RRatioDisplay = double.IsFinite(m.RRatio) ? $"{m.RRatio:0.000}" : "--";

        // SPI + 融合：脉率被灌注门控置为 NaN 后，SpiCalculator 会在推入滚动窗口<b>之前</b>返回 NaN，
        // 所以脱落期间的坏样本不会污染百分位归一化的基线；融合缺了 SPI 这一路同样显示 "---"。
        double spi = _spiCalc.Update(m.PerfusionIndex, pr);
        _latestSpi = spi;
        if (double.IsFinite(spi))
        {
            SpiValue = spi;
            SpiDisplay = $"{Math.Round(spi)}";
            SpiZone = SpiCalculator.ZoneText(spi);
        }
        else
        {
            SpiValue = double.NaN;
            SpiDisplay = "---";
            SpiZone = m.LowPerfusion ? "低灌注" : "";
        }

        RecomputeFusion();
        AddPpgTrendPoints(m.Time);

        // 整场血氧（低灌注时 spo2/pr 为 NaN，报告成图时按 IsFinite 过滤）；回放已整场预加载，不再追加
        if (!_sessionPreloaded)
            Session.AddVital(m.Time, Spo2Value, PulseRateValue, m.PerfusionIndex);
    }

    /// <summary>
    /// 融合值在 CSI 或 SPI 任一路更新时重算，取双方最近一次有效值。
    /// 任一路无效即显示 "---"，不用单路数值冒充融合结果。
    /// </summary>
    private void RecomputeFusion()
    {
        double v = IndexFusion.Combine(_latestCsi, _latestSpi, _config.Fusion);
        FusionValue = v;
        FusionDisplay = double.IsFinite(v) ? $"{Math.Round(v)}" : "---";
        FusionZone = double.IsFinite(v) ? IndexFusion.ZoneText(v)
            : double.IsNaN(_latestSpi) && !double.IsNaN(_latestCsi) ? "缺 SPI"
            : double.IsNaN(_latestCsi) && !double.IsNaN(_latestSpi) ? "缺 CSI"
            : "";
    }

    private void AddPpgTrendPoints(DateTime ts)
    {
        if (!_firstPacketSeen) return;
        double tSec = (ts - _firstPacketTs).TotalSeconds;
        if (double.IsFinite(_latestSpi)) _spiTrend.Points.Add(new DataPoint(tSec, _latestSpi));
        if (double.IsFinite(FusionValue)) _fusionTrend.Points.Add(new DataPoint(tSec, FusionValue));
        if (!IsPlaybackMode) TrimTrend(tSec - TREND_WINDOW_SEC);   // 回放显示整场，不裁剪
        SetTrendAxis(tSec);
        _trendDirty = true;
    }

    /// <summary>趋势图时间轴的零点。脑电与血氧谁先来谁定基准，两路共用同一条时间轴。</summary>
    private void EnsureTimeBase(DateTime ts)
    {
        if (_firstPacketSeen) return;
        _firstPacketSeen = true;
        _firstPacketTs = ts;
    }

    private void Apply(NSMDataPacket pkt)
    {
        SampleRate = _source.SampleRate;
        if (!HasData) HasData = true;
        EnsureTimeBase(pkt.LocalTimestamp);

        if (IsRecording)
        {
            _recorder.Write(AttachPulseWave(pkt));
            RecordedPackets = _recorder.Count;
            RecordingStatus = $"● 录制中 — {RecordedPackets} 包";
        }

        // 回放：若包内带脉搏波，还原绝对 IR/RED 喂处理器，重建 SpO₂/脉搏/SPI/脉搏波（与实时同链）
        if (IsPlaybackMode) ReplayPulseWave(pkt);

        // CSI
        if (pkt.CSIValid && pkt.CSI <= 99)
        {
            CsiValue = pkt.CSI;
            CsiDisplay = pkt.CSI.ToString();
            CsiZone = pkt.CSI switch { < 40 => "过深麻醉", < 60 => "适宜区间", < 80 => "偏浅", _ => "清醒风险" };
        }
        else { CsiValue = double.NaN; CsiDisplay = "---"; CsiZone = "信号无效"; }
        _latestCsi = CsiValue;
        RecomputeFusion();

        // NOX
        if (pkt.NOXValid && pkt.NOX <= 99)
        {
            NoxValue = pkt.NOX;
            NoxDisplay = pkt.NOX.ToString();
            NoxZone = pkt.NOX switch { < 30 => "镇痛充分", <= 50 => "靶区", < 65 => "关注", _ => "镇痛不足!" };
        }
        else { NoxValue = double.NaN; NoxDisplay = "---"; NoxZone = ""; }

        BsValue  = pkt.BSValid ? pkt.BS : double.NaN;
        BsDisplay  = pkt.BSValid ? $"{pkt.BS}%" : "---";
        SqiValue = pkt.SQIValid ? pkt.SQI : double.NaN;
        SqiDisplay = pkt.SQIValid ? $"{pkt.SQI}%" : "---";
        EmgValue = pkt.EMGValid ? pkt.EMG : double.NaN;
        EmgDisplay = pkt.EMGValid ? pkt.EMG.ToString() : "---";
        Sef95Display = $"{pkt.SEF95} Hz";
        EogDisplay = pkt.EOG.ToString();
        Sef95Value = pkt.SEF95;
        EogValue = pkt.EOG;
        BlackImpedanceValue = Math.Min(pkt.BlackImpedance, 10);
        WhiteImpedanceValue = Math.Min(pkt.WhiteImpedance, 10);

        DeltaPower = pkt.DeltaPowerDb;
        ThetaPower = pkt.ThetaPowerDb;
        AlphaPower = pkt.AlphaPowerDb;
        BetaPower  = pkt.BetaPowerDb;
        GammaPower = pkt.GammaPowerDb;

        // 频带功率为 dB 值，线性比值 = 10^((dB差)/10)。δ/α 即 DAR（麻醉越深越大），β/α 反映快波占比。
        double deltaAlphaRatio = Math.Pow(10, (pkt.DeltaPowerDb - pkt.AlphaPowerDb) / 10.0);
        double betaAlphaRatio  = Math.Pow(10, (pkt.BetaPowerDb - pkt.AlphaPowerDb) / 10.0);
        DeltaAlphaRatioDisplay = deltaAlphaRatio.ToString("0.00");
        BetaAlphaRatioDisplay  = betaAlphaRatio.ToString("0.00");

        // 各分波占比：dB → 线性功率后归一化（含 γ）
        double linDelta = Math.Pow(10, pkt.DeltaPowerDb / 10.0);
        double linTheta = Math.Pow(10, pkt.ThetaPowerDb / 10.0);
        double linAlpha = Math.Pow(10, pkt.AlphaPowerDb / 10.0);
        double linBeta  = Math.Pow(10, pkt.BetaPowerDb / 10.0);
        double linGamma = Math.Pow(10, pkt.GammaPowerDb / 10.0);
        double linSum = linDelta + linTheta + linAlpha + linBeta + linGamma;
        if (linSum > 0)
        {
            DeltaPct = 100 * linDelta / linSum;
            ThetaPct = 100 * linTheta / linSum;
            AlphaPct = 100 * linAlpha / linSum;
            BetaPct  = 100 * linBeta  / linSum;
            GammaPct = 100 * linGamma / linSum;
        }

        // 电极阻抗
        BlackImpedanceDisplay = pkt.BlackImpedance >= 15 ? "过高" : pkt.BlackImpedance.ToString();
        WhiteImpedanceDisplay = pkt.WhiteImpedance >= 15 ? "过高" : pkt.WhiteImpedance.ToString();
        if (pkt.ElectrodeAlarm || pkt.ElectrodeInvalid || pkt.ImpedanceHigh
            || pkt.BlackImpedance >= 15 || pkt.WhiteImpedance >= 15)
        {
            var w = new List<string>();
            if (pkt.ElectrodeAlarm) w.Add("电极脱落");
            if (pkt.ElectrodeInvalid) w.Add("电极失效");
            if (pkt.ImpedanceHigh || pkt.BlackImpedance >= 15 || pkt.WhiteImpedance >= 15) w.Add("阻抗过高");
            ElectrodeWarning = "⚠ " + string.Join(" · ", w);
            HasElectrodeWarning = true;
        }
        else { HasElectrodeWarning = false; ElectrodeWarning = ""; }

        // 临床事件（按事件编号合并：持续中的事件只显示一条，并实时累加持续时长）
        HandleEvent(pkt);

        UpdateCharts(pkt);

        // 整场会话缓冲：实时/串口逐包累积；回放已在连接时整场预加载，这里不再追加（避免重复）。
        if (!_sessionPreloaded)
        {
            Session.AddEegSamples(pkt.EEGSamplesUv, pkt.LocalTimestamp);
            Session.AddTrend(pkt.LocalTimestamp, CsiValue, _latestSpi, NoxValue, FusionValue, pkt.SEF95, BsValue);
            Session.AddBands(pkt.LocalTimestamp, DeltaPct, ThetaPct, AlphaPct, BetaPct, GammaPct);
            if (pkt.Dsa.Length == DSA_BINS) Session.AddDsa(pkt.Dsa);
        }
    }

    // 当前正在持续的设备事件，用于消息合并
    private int _currentEventNumber;
    private DateTime _currentEventStart;
    private NsmEventVm? _currentEvent;

    private void HandleEvent(NSMDataPacket pkt)
    {
        if (pkt.EventNumber <= 0)
        {
            _currentEventNumber = 0;     // 事件结束
            _currentEvent = null;
            return;
        }

        if (pkt.EventNumber != _currentEventNumber)
        {
            // 新事件：插入一条，记录起点
            _currentEventNumber = pkt.EventNumber;
            _currentEventStart = pkt.LocalTimestamp;
            string label = $"#{pkt.EventNumber} {EventLabel(pkt.EventType)}";
            _currentEvent = new NsmEventVm(
                pkt.LocalTimestamp.ToString("HH:mm:ss"), label,
                pkt.CSIValid ? pkt.CSI : null);
            Events.Insert(0, _currentEvent);
            while (Events.Count > 50) Events.RemoveAt(Events.Count - 1);
            if (!_sessionPreloaded) Session.AddEvent(pkt.LocalTimestamp, label);   // 回放整场事件已预加载
        }
        else if (_currentEvent != null)
        {
            // 同一事件持续中：只更新持续时长，不新增条目
            _currentEvent.DurationSec = (int)Math.Round((pkt.LocalTimestamp - _currentEventStart).TotalSeconds);
        }
    }

    private void UpdateCharts(NSMDataPacket pkt)
    {
        // EEG 波形滚动
        foreach (var s in pkt.EEGSamplesUv)
        {
            _eegSeries.Points.Add(new DataPoint(_eegX, s));
            _eegX += 1;
        }
        while (_eegSeries.Points.Count > EEG_WINDOW) _eegSeries.Points.RemoveAt(0);
        if (_eegSeries.Points.Count > 0)
        {
            var xaxis = EegModel.Axes[1];
            xaxis.Minimum = _eegSeries.Points[0].X;
            xaxis.Maximum = _eegSeries.Points[^1].X;
        }
        _eegDirty = true;

        // 趋势：实时用滚动 TREND_WINDOW_SEC 秒窗；回放显示整场（0→当前），不裁剪
        double tSec = (pkt.LocalTimestamp - _firstPacketTs).TotalSeconds;
        if (pkt.CSIValid && pkt.CSI <= 99) _csiTrend.Points.Add(new DataPoint(tSec, pkt.CSI));
        if (pkt.NOXValid && pkt.NOX <= 99) _noxTrend.Points.Add(new DataPoint(tSec, pkt.NOX));
        _sefTrend.Points.Add(new DataPoint(tSec, pkt.SEF95));
        if (!IsPlaybackMode) TrimTrend(tSec - TREND_WINDOW_SEC);
        SetTrendAxis(tSec);
        _trendDirty = true;

        UpdateDsa(pkt);
    }

    private void UpdateDsa(NSMDataPacket pkt)
    {
        if (pkt.Dsa.Length != DSA_BINS) return;   // 紧凑记录格式无频谱数据
        _dsaColumns.Add(pkt.Dsa);
        while (_dsaColumns.Count > DSA_COLS) _dsaColumns.RemoveAt(0);
        RefreshDsa();
    }

    /// <summary>由 _dsaColumns 重建热力图数据并刷新。</summary>
    private void RefreshDsa()
    {
        int cols = _dsaColumns.Count;
        var data = new double[Math.Max(1, cols), DSA_BINS];
        for (int x = 0; x < cols; x++)
        {
            var col = _dsaColumns[x];
            for (int y = 0; y < DSA_BINS; y++) data[x, y] = col[y];
        }
        _dsaSeries.Data = data;
        _dsaSeries.X0 = 0;
        _dsaSeries.X1 = Math.Max(1, cols - 1);
        DsaModel.Axes[2].Minimum = 0;
        DsaModel.Axes[2].Maximum = Math.Max(1, cols - 1);
        _dsaDirty = true;
    }

    private void TrimTrend(double minX)
    {
        TrimBefore(_csiTrend, minX);
        TrimBefore(_noxTrend, minX);
        TrimBefore(_sefTrend, minX);
        TrimBefore(_spiTrend, minX);
        TrimBefore(_fusionTrend, minX);
    }

    private static void TrimBefore(LineSeries s, double minX)
    {
        int n = 0;
        while (n < s.Points.Count && s.Points[n].X < minX) n++;
        if (n > 0) s.Points.RemoveRange(0, n);
    }

    private void SetTrendAxis(double tSec)
    {
        var bx = TrendModel.Axes[1];
        if (IsPlaybackMode)
        {
            // 回放：显示 0→整场（回放总时长已知则用它，否则用当前进度），拖动进度条后不从头爬
            bx.Minimum = 0;
            bx.Maximum = Math.Max(Math.Max(_playbackDurationSec, tSec), TREND_WINDOW_SEC);
        }
        else if (tSec > TREND_WINDOW_SEC) { bx.Minimum = tSec - TREND_WINDOW_SEC; bx.Maximum = tSec; }
        else { bx.Minimum = 0; bx.Maximum = TREND_WINDOW_SEC; }
    }

    /// <summary>
    /// 拖动进度后从历史帧重建累积图表：填充 [0, index) 的趋势 / DSA / EEG / 事件，
    /// 而非清空——随后回放循环会正常推送第 index 帧继续累积。
    /// </summary>
    private void RebuildFromHistory(IReadOnlyList<NSMDataPacket> packets, int index)
    {
        _eegSeries.Points.Clear();
        _csiTrend.Points.Clear();
        _noxTrend.Points.Clear();
        _sefTrend.Points.Clear();
        _spiTrend.Points.Clear();
        _fusionTrend.Points.Clear();
        _dsaColumns.Clear();
        Events.Clear();
        _currentEventNumber = 0;
        _currentEvent = null;

        // 血氧重建状态一并复位：拖动后从新位置向前重新推导 SpO₂/SPI，避免带着跳转前的陈旧窗口
        _oximetry.Reset();
        _spiCalc.Reset();
        _latestSpi = double.NaN;

        if (packets.Count == 0 || index <= 0)
        {
            _firstPacketSeen = false;
            _eegX = 0;
            RefreshDsa();
            _eegDirty = _trendDirty = true;
            return;
        }

        _firstPacketSeen = true;
        _firstPacketTs = packets[0].LocalTimestamp;
        double tUpto = (packets[index - 1].LocalTimestamp - _firstPacketTs).TotalSeconds;
        // 回放显示整场，不裁剪；实时（理论上不会走到这里）才用滚动窗
        double minX = IsPlaybackMode ? double.NegativeInfinity : tUpto - TREND_WINDOW_SEC;

        // 趋势（窗口内）+ 事件（全程合并）
        for (int i = 0; i < index; i++)
        {
            var p = packets[i];
            HandleEvent(p);
            double t = (p.LocalTimestamp - _firstPacketTs).TotalSeconds;
            if (t < minX) continue;
            if (p.CSIValid && p.CSI <= 99) _csiTrend.Points.Add(new DataPoint(t, p.CSI));
            if (p.NOXValid && p.NOX <= 99) _noxTrend.Points.Add(new DataPoint(t, p.NOX));
            _sefTrend.Points.Add(new DataPoint(t, p.SEF95));
        }
        SetTrendAxis(tUpto);

        // DSA（近 DSA_COLS 列）
        for (int i = Math.Max(0, index - DSA_COLS); i < index; i++)
            if (packets[i].Dsa.Length == DSA_BINS) _dsaColumns.Add(packets[i].Dsa);
        RefreshDsa();

        // EEG（近 EEG_WINDOW 个样本）
        _eegX = 0;
        for (int i = Math.Max(0, index - (EEG_WINDOW / EEG_SAMPLES_PER_PACKET) - 1); i < index; i++)
            foreach (var s in packets[i].EEGSamplesUv) _eegSeries.Points.Add(new DataPoint(_eegX++, s));
        while (_eegSeries.Points.Count > EEG_WINDOW) _eegSeries.Points.RemoveAt(0);
        if (_eegSeries.Points.Count > 0)
        {
            EegModel.Axes[1].Minimum = _eegSeries.Points[0].X;
            EegModel.Axes[1].Maximum = _eegSeries.Points[^1].X;
        }

        _eegDirty = _trendDirty = true;
    }

    private void ResetCharts()
    {
        _eegSeries.Points.Clear();
        _csiTrend.Points.Clear();
        _noxTrend.Points.Clear();
        _sefTrend.Points.Clear();
        _spiTrend.Points.Clear();
        _fusionTrend.Points.Clear();
        _dsaColumns.Clear();
        _dsaSeries.Data = new double[1, DSA_BINS];
        _dsaSeries.X0 = 0;
        _dsaSeries.X1 = 1;
        _eegX = 0;
        _firstPacketSeen = false;
        Events.Clear();
        _currentEventNumber = 0;
        _currentEvent = null;
        EegModel.InvalidatePlot(true);
        TrendModel.InvalidatePlot(true);
        DsaModel.InvalidatePlot(true);
    }

    /// <summary>波形列的空态遮罩条件：两路都没数据才算"等待数据"。</summary>
    public bool HasAnyData => HasData || IsPpgConnected;

    partial void OnHasDataChanged(bool value) => OnPropertyChanged(nameof(HasAnyData));
    partial void OnIsPpgConnectedChanged(bool value) => OnPropertyChanged(nameof(HasAnyData));

    /// <summary>
    /// UI 线程渲染节拍（约 25 Hz）。把后台线程放下的最新脉搏波与指标取走并上屏。
    /// 取不到就直接返回，因此空闲时开销近乎为零。
    /// </summary>
    public void RenderTick()
    {
        var metrics = Interlocked.Exchange(ref _pendingMetrics, null);
        if (metrics != null) ApplyPpgMetrics(metrics);

        var wave = Interlocked.Exchange(ref _pendingWave, null);
        if (wave != null) UpdatePpgWave(wave);

        // 合并重绘：一拍最多各刷一次，避免回放高倍率下的闪烁
        if (_eegDirty) { _eegDirty = false; EegModel.InvalidatePlot(true); }
        if (_trendDirty) { _trendDirty = false; TrendModel.InvalidatePlot(true); }
        if (_dsaDirty) { _dsaDirty = false; DsaModel.InvalidatePlot(true); }
    }

    public void TickClock()
    {
        CurrentTime = DateTime.Now.ToString("HH:mm:ss");

        // 链路健康（协议 §6.3）：正常 125±2 帧/秒；坏帧长期应为 0
        if (IsPpgConnected)
        {
            long bad = _ppgSource.BadFrames;
            PpgLinkDisplay = bad > 0
                ? $"{_ppgSource.FrameRate:0} 帧/秒 · 坏帧 {bad}"
                : $"{_ppgSource.FrameRate:0} 帧/秒";
        }
    }

    private static string EventLabel(NSMEventType t) => t switch
    {
        NSMEventType.Induction => "麻醉诱导",
        NSMEventType.Intubation => "气管插管",
        NSMEventType.Maintenance => "麻醉维持",
        NSMEventType.Surgery => "手术/切皮",
        NSMEventType.Injection => "给药",
        NSMEventType.Note => "备注",
        NSMEventType.EndMaintenance => "维持结束",
        NSMEventType.Movement => "体动",
        _ => "一般事件",
    };

    /// <summary>
    /// 切到 UI 线程执行，<b>非阻塞</b>。
    ///
    /// 这里必须用 InvokeAsync 而不是 Invoke：调用方是 SerialPort.DataReceived 的线程池线程，
    /// 而 <c>SerialPort.Close()</c> 会等待正在执行的 DataReceived 回调返回。若回调阻塞在
    /// Invoke 上等 UI 线程，而 UI 线程正卡在 Close() 里等回调，就是死锁 —— 表现为整个程序卡死，
    /// 且进程退化成一个卡在驱动里、连 TerminateProcess 都杀不掉的僵尸。
    /// 用 InvokeAsync 后回调立即返回，Close() 永远等得到。
    /// </summary>
    private static void RunOnUI(Action action)
    {
        var app = Application.Current;
        if (app == null) { action(); return; }
        if (app.Dispatcher.CheckAccess()) { action(); return; }
        app.Dispatcher.InvokeAsync(action);
    }
}

public sealed partial class NsmEventVm : ObservableObject
{
    public NsmEventVm(string time, string label, int? csi, bool isManual = false)
    {
        Time = time;
        Label = label;
        Csi = csi;
        IsManual = isManual;
    }

    public string Time { get; }
    public string Label { get; }
    public int? Csi { get; }
    public bool IsManual { get; }

    /// <summary>事件持续秒数（合并显示）；0 表示瞬时事件。</summary>
    [ObservableProperty] private int _durationSec;
    partial void OnDurationSecChanged(int value) => OnPropertyChanged(nameof(DurationText));

    public string CsiText => Csi.HasValue ? $"CSI {Csi}" : "";
    public string DurationText => DurationSec <= 0
        ? ""
        : DurationSec < 60 ? $"持续 {DurationSec}秒"
        : $"持续 {DurationSec / 60}分{DurationSec % 60}秒";

    public Brush Accent => new SolidColorBrush(IsManual
        ? Color.FromRgb(0xF0, 0xA0, 0x20)   // 手动：琥珀色
        : Color.FromRgb(0x00, 0xC8, 0xFF)); // 设备：青色
}
