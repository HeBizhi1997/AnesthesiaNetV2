using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using NSMMonitor.Models;
using NSMMonitor.Services;
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

    private readonly NsmSerialService _serial;
    private readonly NsmSimulatorService _simulator;
    private readonly NsmPlaybackService _playback;
    private readonly NsmRecordingService _recorder;
    private readonly ILogger<MainViewModel> _logger;

    private INsmDataSource _source;
    private double _eegX;
    private DateTime _firstPacketTs;
    private bool _firstPacketSeen;

    private readonly LineSeries _eegSeries;
    private readonly LineSeries _csiTrend;
    private readonly LineSeries _noxTrend;
    private readonly LineSeries _sefTrend;

    private readonly HeatMapSeries _dsaSeries;
    private readonly List<byte[]> _dsaColumns = new(DSA_COLS);

    public MainViewModel(NsmSerialService serial, NsmSimulatorService simulator,
        NsmPlaybackService playback, NsmRecordingService recorder, ILogger<MainViewModel> logger)
    {
        _serial = serial;
        _simulator = simulator;
        _playback = playback;
        _recorder = recorder;
        _logger = logger;
        _source = serial;

        _eegSeries = new LineSeries { Color = OxyColor.FromRgb(0x2F, 0xE6, 0xD6), StrokeThickness = 1.3 };
        _csiTrend  = new LineSeries { Title = "CSI", Color = OxyColor.FromRgb(0xA7, 0x8B, 0xFA), StrokeThickness = 1.8 };
        _noxTrend  = new LineSeries { Title = "NOX", Color = OxyColor.FromRgb(0x2D, 0xD4, 0xBF), StrokeThickness = 1.6 };
        _sefTrend  = new LineSeries { Title = "SEF95", Color = OxyColor.FromRgb(0xF5, 0xC2, 0x42), StrokeThickness = 1.4, LineStyle = LineStyle.Dash };

        _dsaSeries = new HeatMapSeries
        {
            X0 = 0, X1 = 1, Y0 = 1, Y1 = DSA_BINS,
            Interpolate = true,
            RenderMethod = HeatMapRenderMethod.Bitmap,
            Data = new double[1, DSA_BINS],
        };

        EegModel = BuildEegModel();
        TrendModel = BuildTrendModel();
        DsaModel = BuildDsaModel();

        RefreshPorts();
        WireSource(_serial);
        WireSource(_simulator);
        WireSource(_playback);
        _playback.PlaybackCompleted += () => RunOnUI(OnPlaybackCompleted);
        _playback.ProgressChanged += p => RunOnUI(() => OnPlaybackProgress(p));
        _playback.Seeked += (pkts, idx) => RunOnUI(() => RebuildFromHistory(pkts, idx));

        // 调试/演示辅助：设置环境变量 NSM_AUTOCONNECT=1 时自动连接模拟器
        if (Environment.GetEnvironmentVariable("NSM_AUTOCONNECT") == "1")
            ToggleConnection();
    }

    // ─────────────────────────── 图表 ───────────────────────────
    public PlotModel EegModel { get; }
    public PlotModel TrendModel { get; }
    public PlotModel DsaModel { get; }

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

    private PlotModel BuildEegModel()
    {
        var m = new PlotModel { PlotMargins = new OxyThickness(0), Padding = new OxyThickness(0), Background = OxyColors.Transparent };
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -130, Maximum = 130, IsAxisVisible = false });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        m.Series.Add(_eegSeries);
        return m;
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
            MajorGridlineStyle = LineStyle.Solid,
            MajorGridlineColor = OxyColor.FromRgb(0x18, 0x2C, 0x46),
            TextColor = OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            TicklineColor = OxyColor.FromRgb(0x18, 0x2C, 0x46),
        });
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom, Minimum = 0, Maximum = TREND_WINDOW_SEC,
            Title = "时间 (秒)", FontSize = 10,
            TextColor = OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            TicklineColor = OxyColor.FromRgb(0x18, 0x2C, 0x46),
        });
        m.Series.Add(_csiTrend);
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
            MajorStep = 10, MinorStep = 5,
            TextColor = OxyColor.FromRgb(0x3D, 0x5A, 0x7A),
            TicklineColor = OxyColor.FromRgb(0x18, 0x2C, 0x46),
        });
        // X 轴：时间列（隐藏刻度，仅作滚动）
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom, Minimum = 0, Maximum = DSA_COLS,
            IsAxisVisible = false,
        });
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
        }
    }

    private bool Fail(string msg) { StatusMessage = msg; return false; }

    private void Disconnect()
    {
        if (IsRecording) StopRecording();
        _source.Disconnect();
        IsConnected = false;
        HasData = false;
        EmptyHint = "已断开 · 请重新连接数据源";
        SourceName = "未连接";
    }

    private void OnPlaybackCompleted()
    {
        if (IsRecording) StopRecording();
        IsConnected = false;
        SourceName = "未连接";
        EmptyHint = "回放结束 · 可重新选择文件";
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

        _recorder.Start(dlg.FileName);
        IsRecording = true;
        RecordedPackets = 0;
        RecordingStatus = $"● 录制中 — {Path.GetFileName(dlg.FileName)}";
    }

    private void StopRecording()
    {
        _recorder.Stop();
        IsRecording = false;
        RecordingStatus = $"已保存 {RecordedPackets} 包";
    }

    private static string RecordingsDir()
    {
        var dir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Recordings");
        Directory.CreateDirectory(dir);
        return dir;
    }

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

    // 频带比值（线性域）：α/δ、α/β
    [ObservableProperty] private string _alphaDeltaRatioDisplay = "--";
    [ObservableProperty] private string _alphaBetaRatioDisplay = "--";

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

    public ObservableCollection<NsmEventVm> Events { get; } = new();

    // ─────────────────────────── 数据接收 ───────────────────────────
    private void WireSource(INsmDataSource src)
    {
        src.StatusChanged += msg => RunOnUI(() => StatusMessage = msg);
        src.ErrorOccurred += ex => RunOnUI(() => StatusMessage = $"错误：{ex.Message}");
        src.NSMDataReceived += OnPacket;
    }

    private void OnPacket(NSMDataPacket pkt) => RunOnUI(() => Apply(pkt));

    private void Apply(NSMDataPacket pkt)
    {
        SampleRate = _source.SampleRate;
        if (!HasData) HasData = true;
        if (!_firstPacketSeen) { _firstPacketSeen = true; _firstPacketTs = pkt.LocalTimestamp; }

        if (IsRecording)
        {
            _recorder.Write(pkt);
            RecordedPackets = _recorder.Count;
            RecordingStatus = $"● 录制中 — {RecordedPackets} 包";
        }

        // CSI
        if (pkt.CSIValid && pkt.CSI <= 99)
        {
            CsiValue = pkt.CSI;
            CsiDisplay = pkt.CSI.ToString();
            CsiZone = pkt.CSI switch { < 40 => "过深麻醉", < 60 => "适宜区间", < 80 => "偏浅", _ => "清醒风险" };
        }
        else { CsiValue = double.NaN; CsiDisplay = "---"; CsiZone = "信号无效"; }

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

        // 频带功率为 dB 值，线性比值 = 10^((dB差)/10)
        double alphaDeltaRatio = Math.Pow(10, (pkt.AlphaPowerDb - pkt.DeltaPowerDb) / 10.0);
        double alphaBetaRatio  = Math.Pow(10, (pkt.AlphaPowerDb - pkt.BetaPowerDb) / 10.0);
        AlphaDeltaRatioDisplay = alphaDeltaRatio.ToString("0.00");
        AlphaBetaRatioDisplay  = alphaBetaRatio.ToString("0.00");

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
            _currentEvent = new NsmEventVm(
                pkt.LocalTimestamp.ToString("HH:mm:ss"),
                $"#{pkt.EventNumber} {EventLabel(pkt.EventType)}",
                pkt.CSIValid ? pkt.CSI : null);
            Events.Insert(0, _currentEvent);
            while (Events.Count > 50) Events.RemoveAt(Events.Count - 1);
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
        EegModel.InvalidatePlot(true);

        // 趋势（以录制时间轴为基准，滚动窗口保留近 TREND_WINDOW_SEC 秒）
        double tSec = (pkt.LocalTimestamp - _firstPacketTs).TotalSeconds;
        if (pkt.CSIValid && pkt.CSI <= 99) _csiTrend.Points.Add(new DataPoint(tSec, pkt.CSI));
        if (pkt.NOXValid && pkt.NOX <= 99) _noxTrend.Points.Add(new DataPoint(tSec, pkt.NOX));
        _sefTrend.Points.Add(new DataPoint(tSec, pkt.SEF95));
        TrimTrend(tSec - TREND_WINDOW_SEC);
        SetTrendAxis(tSec);
        TrendModel.InvalidatePlot(true);

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
        DsaModel.InvalidatePlot(true);
    }

    private void TrimTrend(double minX)
    {
        TrimBefore(_csiTrend, minX);
        TrimBefore(_noxTrend, minX);
        TrimBefore(_sefTrend, minX);
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
        if (tSec > TREND_WINDOW_SEC) { bx.Minimum = tSec - TREND_WINDOW_SEC; bx.Maximum = tSec; }
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
        _dsaColumns.Clear();
        Events.Clear();
        _currentEventNumber = 0;
        _currentEvent = null;

        if (packets.Count == 0 || index <= 0)
        {
            _firstPacketSeen = false;
            _eegX = 0;
            RefreshDsa();
            EegModel.InvalidatePlot(true);
            TrendModel.InvalidatePlot(true);
            return;
        }

        _firstPacketSeen = true;
        _firstPacketTs = packets[0].LocalTimestamp;
        double tUpto = (packets[index - 1].LocalTimestamp - _firstPacketTs).TotalSeconds;
        double minX = tUpto - TREND_WINDOW_SEC;

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

        EegModel.InvalidatePlot(true);
        TrendModel.InvalidatePlot(true);
    }

    private void ResetCharts()
    {
        _eegSeries.Points.Clear();
        _csiTrend.Points.Clear();
        _noxTrend.Points.Clear();
        _sefTrend.Points.Clear();
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

    public void TickClock() => CurrentTime = DateTime.Now.ToString("HH:mm:ss");

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

    private static void RunOnUI(Action action)
    {
        var app = Application.Current;
        if (app == null) { action(); return; }
        if (app.Dispatcher.CheckAccess()) action();
        else app.Dispatcher.Invoke(action);
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
