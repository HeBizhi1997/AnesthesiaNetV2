using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using LiveChartsCore.SkiaSharpView.Painting.Effects;
using Newtonsoft.Json;
using SkiaSharp;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows.Threading;

namespace Ads1299Monitor.ViewModels;

/// <summary>Replays a recorded session: full-session trend (qCON/qNOX/SQI/HR) with event
/// markers + a moving cursor, per-second index readout, and the raw EEG waveform window
/// scrubbed to the cursor time. Reads inference.jsonl / events.jsonl / raw_signal.bin.</summary>
public partial class PlaybackViewModel : ObservableObject
{
    // ── one recorded second ──
    private sealed class Frame
    {
        public DateTime T;
        public double? QCon, QNox, Hr, Hrv, Spo2;
        public double Sqi, D, Th, Al, Be, Ga;
    }

    private sealed class InfDto
    {
        [JsonProperty("t")] public DateTime T { get; set; }
        [JsonProperty("qcon")] public double? QCon { get; set; }
        [JsonProperty("qnox")] public double? QNox { get; set; }
        [JsonProperty("sqi")] public double Sqi { get; set; }
        [JsonProperty("hr")] public double? Hr { get; set; }
        [JsonProperty("hrv")] public double? Hrv { get; set; }
        [JsonProperty("spo2")] public double? Spo2 { get; set; }
        [JsonProperty("bands")] public BandsDto? Bands { get; set; }
    }
    private sealed class BandsDto
    {
        public double DeltaPower { get; set; }
        public double ThetaPower { get; set; }
        public double AlphaPower { get; set; }
        public double BetaPower { get; set; }
        public double GammaPower { get; set; }
    }
    private sealed class EvtDto
    {
        [JsonProperty("Time")] public string Time { get; set; } = "";
        [JsonProperty("Category")] public string Category { get; set; } = "";
        [JsonProperty("Name")] public string Name { get; set; } = "";
    }

    private readonly string _root;
    private readonly List<Frame> _frames = new();
    private long[] _eegTicks = Array.Empty<long>();
    private float[] _eegUv = Array.Empty<float>();
    private double _eegFs = 250;
    private readonly DispatcherTimer _timer;

    public ObservableCollection<string> Sessions { get; } = new();
    [ObservableProperty] private string? _selectedSession;
    [ObservableProperty] private string _summary = "请选择一个会话并点击「加载」";

    [ObservableProperty] private bool _loaded;
    [ObservableProperty] private int _frameCount;
    [ObservableProperty] private int _currentIndex;
    [ObservableProperty] private double _speed = 1;
    [ObservableProperty] private bool _isPlaying;
    [ObservableProperty] private string _playPauseText = "▶ 播放";

    // readout
    [ObservableProperty] private string _curTime = "--:--:--";
    [ObservableProperty] private string _qConText = "--";
    [ObservableProperty] private string _qNoxText = "--";
    [ObservableProperty] private string _sqiText = "--";
    [ObservableProperty] private string _bandText = "δ-- θ-- α-- β-- γ--";

    // charts
    public ObservableCollection<ISeries> TrendSeries { get; } = new();
    public Axis[] TrendXAxes { get; }
    public Axis[] TrendYAxes { get; }
    public ObservableCollection<RectangularSection> TrendSections { get; } = new();
    public ObservableCollection<ISeries> EegSeries { get; } = new();
    public Axis[] EegXAxes { get; }
    public Axis[] EegYAxes { get; }

    private readonly ObservableCollection<ObservablePoint> _qcon = new(), _qnox = new(), _sqi = new(), _hr = new();
    private readonly ObservableCollection<ObservablePoint> _eeg = new();
    private RectangularSection _cursor = null!;

    private static SolidColorPaint P(string hex) => new(SKColor.Parse(hex));

    public PlaybackViewModel(string recordingsRoot)
    {
        _root = recordingsRoot;

        TrendSeries.Add(Line(_qcon, "qCON", "#3B82F6", 0));
        TrendSeries.Add(Line(_qnox, "qNOX", "#F59E0B", 0));
        TrendSeries.Add(Line(_sqi, "SQI", "#22C55E", 0));
        TrendSeries.Add(Line(_hr, "HR", "#EF4444", 1));
        TrendXAxes = new[] { new Axis { Labeler = TimeLabel, LabelsPaint = P("#7D8B9A"), TextSize = 11, MinStep = 1, UnitWidth = 1,
                                        SeparatorsPaint = P("#16202C") } };
        TrendYAxes = new[]
        {
            new Axis { MinLimit = 0, MaxLimit = 100, LabelsPaint = P("#7D8B9A"), TextSize = 11, SeparatorsPaint = P("#16202C") },
            new Axis { MinLimit = 40, MaxLimit = 160, Position = LiveChartsCore.Measure.AxisPosition.End, LabelsPaint = P("#EF4444"), TextSize = 11, ShowSeparatorLines = false },
        };

        EegSeries.Add(new LineSeries<ObservablePoint>
        {
            Values = _eeg, Stroke = P("#3B82F6"), Fill = null, GeometrySize = 0, LineSmoothness = 0, AnimationsSpeed = TimeSpan.Zero,
        });
        EegXAxes = new[] { new Axis { LabelsPaint = null, SeparatorsPaint = null, ShowSeparatorLines = false, TicksPaint = null } };
        EegYAxes = new[] { new Axis { LabelsPaint = null, SeparatorsPaint = null, ShowSeparatorLines = false, TicksPaint = null } };

        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(1000) };
        _timer.Tick += (_, _) => Step();

        RefreshSessions();
    }

    private void RefreshSessions()
    {
        Sessions.Clear();
        try
        {
            if (Directory.Exists(_root))
                foreach (var d in Directory.GetDirectories(_root)
                             .Where(d => File.Exists(Path.Combine(d, "inference.jsonl")))
                             .OrderByDescending(d => Directory.GetCreationTimeUtc(d)))
                    Sessions.Add(Path.GetFileName(d));
        }
        catch { /* ignore */ }
        if (Sessions.Count > 0) SelectedSession = Sessions[0];
        if (Sessions.Count == 0) Summary = $"未找到录制会话:{_root}";
    }

    [RelayCommand]
    private void Load()
    {
        if (string.IsNullOrEmpty(SelectedSession)) return;
        Pause();
        var dir = Path.Combine(_root, SelectedSession);
        _frames.Clear();
        foreach (var line in SafeReadLines(Path.Combine(dir, "inference.jsonl")))
        {
            InfDto? d = TryParse<InfDto>(line);
            if (d == null) continue;
            var b = d.Bands ?? new BandsDto();
            double tot = b.DeltaPower + b.ThetaPower + b.AlphaPower + b.BetaPower + b.GammaPower; if (tot <= 0) tot = 1;
            _frames.Add(new Frame
            {
                T = d.T, QCon = d.QCon, QNox = d.QNox, Sqi = d.Sqi, Hr = d.Hr, Hrv = d.Hrv, Spo2 = d.Spo2,
                D = b.DeltaPower / tot, Th = b.ThetaPower / tot, Al = b.AlphaPower / tot, Be = b.BetaPower / tot, Ga = b.GammaPower / tot,
            });
        }

        _qcon.Clear(); _qnox.Clear(); _sqi.Clear(); _hr.Clear();
        for (int i = 0; i < _frames.Count; i++)
        {
            var f = _frames[i];
            if (f.QCon.HasValue) _qcon.Add(new ObservablePoint(i, f.QCon.Value));
            if (f.QNox.HasValue) _qnox.Add(new ObservablePoint(i, f.QNox.Value));
            _sqi.Add(new ObservablePoint(i, f.Sqi));
            if (f.Hr.HasValue) _hr.Add(new ObservablePoint(i, f.Hr.Value));
        }

        // events → vertical markers (match by wall-clock time to nearest frame index)
        TrendSections.Clear();
        foreach (var line in SafeReadLines(Path.Combine(dir, "events.jsonl")))
        {
            var e = TryParse<EvtDto>(line);
            if (e == null) continue;
            int idx = NearestIndexByTime(e.Time);
            if (idx < 0) continue;
            TrendSections.Add(new RectangularSection
            {
                Xi = idx, Xj = idx, Stroke = new SolidColorPaint(SKColor.Parse("#94A3B8"), 1) { PathEffect = new DashEffect(new float[] { 4, 4 }) },
                Label = $"{e.Category}:{e.Name}", LabelSize = 10, LabelPaint = P("#94A3B8"),
            });
        }
        // cursor marker
        _cursor = new RectangularSection { Xi = 0, Xj = 0, Stroke = new SolidColorPaint(SKColor.Parse("#22D3EE"), 2) };
        TrendSections.Add(_cursor);

        LoadRawEeg(dir);

        FrameCount = _frames.Count;
        Loaded = _frames.Count > 0;
        CurrentIndex = 0;
        var dur = _frames.Count > 1 ? (_frames[^1].T - _frames[0].T) : TimeSpan.Zero;
        Summary = Loaded
            ? $"{SelectedSession}  ·  {_frames.Count} 帧 / {dur:hh\\:mm\\:ss}  ·  事件 {TrendSections.Count - 1} 个  ·  原始EEG {_eegUv.Length} 点@{_eegFs:0}Hz"
            : "该会话无可回放数据";
        UpdateReadout();
    }

    private void LoadRawEeg(string dir)
    {
        _eegTicks = Array.Empty<long>(); _eegUv = Array.Empty<float>(); _eegFs = 250;
        try
        {
            var meta = Path.Combine(dir, "raw_signal.meta.json");
            if (File.Exists(meta))
            {
                dynamic? m = JsonConvert.DeserializeObject(File.ReadAllText(meta));
                if (m?.eeg_sample_rate_hz != null) _eegFs = (double)m.eeg_sample_rate_hz;
            }
            var bin = Path.Combine(dir, "raw_signal.bin");
            if (!File.Exists(bin)) return;
            var ticks = new List<long>(); var uv = new List<float>();
            using var br = new BinaryReader(File.OpenRead(bin));
            long len = br.BaseStream.Length;
            while (br.BaseStream.Position + 9 <= len)
            {
                long t = br.ReadInt64(); byte tag = br.ReadByte();
                if (tag == 1) { if (br.BaseStream.Position + 4 > len) break; ticks.Add(t); uv.Add(br.ReadSingle()); }
                else if (tag == 2) { if (br.BaseStream.Position + 12 > len) break; br.ReadSingle(); br.ReadSingle(); br.ReadSingle(); }
                else break;
            }
            _eegTicks = ticks.ToArray(); _eegUv = uv.ToArray();
        }
        catch { /* best effort */ }
    }

    // ── playback control ──
    [RelayCommand]
    private void PlayPause()
    {
        if (!Loaded) return;
        if (IsPlaying) Pause();
        else
        {
            if (CurrentIndex >= FrameCount - 1) CurrentIndex = 0;
            IsPlaying = true; PlayPauseText = "⏸ 暂停";
            _timer.Interval = TimeSpan.FromMilliseconds(Math.Max(40, 1000.0 / Math.Max(0.5, Speed)));
            _timer.Start();
        }
    }

    private void Pause() { _timer.Stop(); IsPlaying = false; PlayPauseText = "▶ 播放"; }

    partial void OnSpeedChanged(double value)
    {
        if (IsPlaying) _timer.Interval = TimeSpan.FromMilliseconds(Math.Max(40, 1000.0 / Math.Max(0.5, value)));
    }

    private void Step()
    {
        if (CurrentIndex >= FrameCount - 1) { Pause(); return; }
        CurrentIndex++;
    }

    partial void OnCurrentIndexChanged(int value)
    {
        if (_cursor != null) { _cursor.Xi = value; _cursor.Xj = value; }
        UpdateReadout();
        UpdateEegWindow();
    }

    private void UpdateReadout()
    {
        if (_frames.Count == 0 || CurrentIndex < 0 || CurrentIndex >= _frames.Count) return;
        var f = _frames[CurrentIndex];
        CurTime = f.T.ToString("HH:mm:ss");
        QConText = f.QCon.HasValue ? $"{f.QCon.Value:0}" : "--";
        QNoxText = f.QNox.HasValue ? $"{f.QNox.Value:0}" : "--";
        SqiText = $"{f.Sqi:0}";
        BandText = $"δ{f.D*100:0}  θ{f.Th*100:0}  α{f.Al*100:0}  β{f.Be*100:0}  γ{f.Ga*100:0}";
    }

    private void UpdateEegWindow()
    {
        _eeg.Clear();
        if (_eegTicks.Length == 0 || CurrentIndex < 0 || CurrentIndex >= _frames.Count) return;
        long tEnd = _frames[CurrentIndex].T.Ticks;
        long tStart = tEnd - TimeSpan.FromSeconds(10).Ticks;
        int lo = LowerBound(_eegTicks, tStart);
        int x = 0;
        for (int i = lo; i < _eegTicks.Length && _eegTicks[i] <= tEnd; i++)
            _eeg.Add(new ObservablePoint(x++, _eegUv[i]));
    }

    // ── helpers ──
    private int NearestIndexByTime(string hhmmss)
    {
        if (_frames.Count == 0 || !TimeSpan.TryParse(hhmmss, out var ts)) return -1;
        int best = -1; double bestDiff = double.MaxValue;
        for (int i = 0; i < _frames.Count; i++)
        {
            double diff = Math.Abs((_frames[i].T.TimeOfDay - ts).TotalSeconds);
            if (diff < bestDiff) { bestDiff = diff; best = i; }
        }
        return bestDiff <= 5 ? best : best;   // accept nearest
    }

    private string TimeLabel(double x)
    {
        int i = (int)Math.Round(x);
        return (i >= 0 && i < _frames.Count) ? _frames[i].T.ToString("HH:mm:ss") : "";
    }

    private static int LowerBound(long[] a, long key)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] < key) lo = m + 1; else hi = m; }
        return lo;
    }

    private static IEnumerable<string> SafeReadLines(string path)
    {
        if (!File.Exists(path)) yield break;
        foreach (var l in File.ReadLines(path))
            if (!string.IsNullOrWhiteSpace(l)) yield return l;
    }

    private static T? TryParse<T>(string line) where T : class
    {
        try { return JsonConvert.DeserializeObject<T>(line); } catch { return null; }
    }

    private static LineSeries<ObservablePoint> Line(ObservableCollection<ObservablePoint> v, string name, string hex, int yAxis) => new()
    {
        Name = name, Values = v, Stroke = P(hex), Fill = null, GeometrySize = 0, LineSmoothness = 0.3,
        ScalesYAt = yAxis, AnimationsSpeed = TimeSpan.Zero,
    };

    public void Dispose() => _timer.Stop();
}
