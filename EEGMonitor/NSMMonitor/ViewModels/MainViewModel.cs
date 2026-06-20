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
    private const int TREND_WINDOW_SEC = 300;

    private readonly NsmSerialService _serial;
    private readonly NsmSimulatorService _simulator;
    private readonly NsmPlaybackService _playback;
    private readonly NsmRecordingService _recorder;
    private readonly ILogger<MainViewModel> _logger;

    private INsmDataSource _source;
    private double _eegX;
    private DateTime _trendStart = DateTime.Now;

    private readonly LineSeries _eegSeries;
    private readonly LineSeries _csiTrend;
    private readonly LineSeries _noxTrend;
    private readonly LineSeries _sefTrend;

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

        EegModel = BuildEegModel();
        TrendModel = BuildTrendModel();

        RefreshPorts();
        WireSource(_serial);
        WireSource(_simulator);
        WireSource(_playback);
        _playback.PlaybackCompleted += () => RunOnUI(OnPlaybackCompleted);

        // 调试/演示辅助：设置环境变量 NSM_AUTOCONNECT=1 时自动连接模拟器
        if (Environment.GetEnvironmentVariable("NSM_AUTOCONNECT") == "1")
            ToggleConnection();
    }

    // ─────────────────────────── 图表 ───────────────────────────
    public PlotModel EegModel { get; }
    public PlotModel TrendModel { get; }

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

    public bool IsSerialMode => SelectedSourceMode == SourceMode.Serial;
    public bool IsPlaybackMode => SelectedSourceMode == SourceMode.Playback;
    public string PlaybackFileDisplay =>
        string.IsNullOrEmpty(PlaybackFile) ? "（未选择文件）" : Path.GetFileName(PlaybackFile);

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
            IsManual: true));
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

        // 临床事件
        if (pkt.EventNumber > 0)
        {
            Events.Insert(0, new NsmEventVm(
                pkt.LocalTimestamp.ToString("HH:mm:ss"),
                $"#{pkt.EventNumber} {EventLabel(pkt.EventType)}",
                pkt.CSIValid ? pkt.CSI : null));
            while (Events.Count > 50) Events.RemoveAt(Events.Count - 1);
        }

        UpdateCharts(pkt);
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

        // 趋势
        double tSec = (DateTime.Now - _trendStart).TotalSeconds;
        if (pkt.CSIValid && pkt.CSI <= 99) _csiTrend.Points.Add(new DataPoint(tSec, pkt.CSI));
        if (pkt.NOXValid && pkt.NOX <= 99) _noxTrend.Points.Add(new DataPoint(tSec, pkt.NOX));
        _sefTrend.Points.Add(new DataPoint(tSec, pkt.SEF95));

        var bx = TrendModel.Axes[1];
        if (tSec > TREND_WINDOW_SEC)
        {
            bx.Minimum = tSec - TREND_WINDOW_SEC;
            bx.Maximum = tSec;
        }
        TrendModel.InvalidatePlot(true);
    }

    private void ResetCharts()
    {
        _eegSeries.Points.Clear();
        _csiTrend.Points.Clear();
        _noxTrend.Points.Clear();
        _sefTrend.Points.Clear();
        _eegX = 0;
        _trendStart = DateTime.Now;
        Events.Clear();
        EegModel.InvalidatePlot(true);
        TrendModel.InvalidatePlot(true);
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

public sealed record NsmEventVm(string Time, string Label, int? Csi, bool IsManual = false)
{
    public string CsiText => Csi.HasValue ? $"CSI {Csi}" : "";
    public Brush Accent => new SolidColorBrush(IsManual
        ? Color.FromRgb(0xF0, 0xA0, 0x20)   // 手动：琥珀色
        : Color.FromRgb(0x00, 0xC8, 0xFF)); // 设备：青色
}
