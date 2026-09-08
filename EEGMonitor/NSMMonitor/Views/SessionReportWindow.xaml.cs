using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using NSMMonitor.Services;
using NSMMonitor.ViewModels;
using OxyPlot;
using OxyPlot.Annotations;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace NSMMonitor.Views;

/// <summary>
/// 整场麻醉回顾报告：从 <see cref="SessionBuffer"/> 快照一次性成图。
///
/// 设计要点（麻醉回顾视角）：
///   • 顶部 KPI 概览——靶区占比、爆发抑制、低氧事件，是术后回顾第一眼要看的；
///   • 所有时间序列面板<b>共用同一条手术计时轴</b>（0→总时长），临床事件竖线穿透全部面板，
///     一条竖线即可读出某事件时刻各指标的联动——这也修复了此前脑电轴与其余面板对不齐的 bug；
///   • 面板按临床优先级排：深度 → 爆发抑制 → 频谱 → 伤害感受 → 血氧 → 成分演变 → 原始脑电（折叠弱化）。
/// 不订阅实时更新，点"刷新"重取快照。
/// </summary>
public partial class SessionReportWindow : Window, INotifyPropertyChanged
{
    private const int MaxTrendPoints = 2000;
    private const int MaxEegColumns = 1600;
    private const int MaxDsaColumns = 800;

    private static readonly OxyColor Grid = OxyColor.FromRgb(0x18, 0x2C, 0x46);
    private static readonly OxyColor Tick = OxyColor.FromRgb(0x3D, 0x5A, 0x7A);
    private static readonly OxyColor LegendText = OxyColor.FromRgb(0xC8, 0xDC, 0xF0);
    private static readonly OxyColor EventColor = OxyColor.FromArgb(0xC0, 0xFB, 0xBF, 0x24);

    private readonly SessionBuffer _session;
    private double _duration;                    // 全部面板共用的时间轴上限
    private List<SessionBuffer.EventMark> _events = new();

    // 报告是静态视图：解绑 OxyPlot 的滚轮缩放，让滚轮改为滚动整页（否则鼠标停在图上时滚不动页面）。
    private static readonly IPlotController StaticController = MakeStaticController();
    private static IPlotController MakeStaticController()
    {
        var c = new PlotController();
        c.UnbindMouseWheel();   // 关掉滚轮缩放——报告是静态视图，滚轮应滚页
        return c;
    }

    public SessionReportWindow(SessionBuffer session)
    {
        _session = session;
        InitializeComponent();
        DataContext = this;
        // 报告各图横轴均为手术计时秒 → 套用「时间 + 各指标值」悬停框
        var timeTracker = Application.Current.TryFindResource("TimeTrackerTemplate") as ControlTemplate;
        foreach (var pv in new[] { CsiPlot, BsrPlot, DsaPlot, SpiPlot, VitalPlot, BandPlot, EegPlot })
        {
            pv.Controller = StaticController;
            if (timeTracker != null) pv.DefaultTrackerTemplate = timeTracker;
        }
        Loaded += (_, _) => Refresh();
        // 滚轮落到图上时转发给外层 ScrollViewer，实现整页滚动
        AddHandler(UIElement.PreviewMouseWheelEvent, new System.Windows.Input.MouseWheelEventHandler(OnPreviewWheel), true);
    }

    private void OnPreviewWheel(object sender, System.Windows.Input.MouseWheelEventArgs e)
    {
        if (e.Source is OxyPlot.Wpf.PlotView)
        {
            var sv = FindScrollViewer(this);
            if (sv != null)
            {
                sv.ScrollToVerticalOffset(sv.VerticalOffset - e.Delta);
                e.Handled = true;
            }
        }
    }

    private static ScrollViewer? FindScrollViewer(DependencyObject root)
    {
        for (int i = 0; i < System.Windows.Media.VisualTreeHelper.GetChildrenCount(root); i++)
        {
            var child = System.Windows.Media.VisualTreeHelper.GetChild(root, i);
            if (child is ScrollViewer sv) return sv;
            var found = FindScrollViewer(child);
            if (found != null) return found;
        }
        return null;
    }

    private string _headerInfo = "";
    public string HeaderInfo
    {
        get => _headerInfo;
        private set { _headerInfo = value; PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(HeaderInfo))); }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private void Refresh_Click(object sender, RoutedEventArgs e) => Refresh();

    public void Refresh()
    {
        var s = _session.GetSummary();
        bool empty = s.TrendCount == 0 && s.VitalCount == 0 && s.DurationSec <= 0;
        EmptyHint.Visibility = empty ? Visibility.Visible : Visibility.Collapsed;

        _duration = Math.Max(1, s.DurationSec);
        _events = _session.GetEvents();
        HeaderInfo = $"{s.SourceName} · 时长 {MainViewModel.TimeLabel(s.DurationSec)} · " +
                     $"指标 {s.TrendCount} 条 · 事件 {s.EventCount} 个";

        // 事件编号↔名称对照（图上只标编号，全名在此读）：如「#1 气管插管 @ 02:13」
        EventLegend.Text = _events.Count == 0
            ? "本段无临床事件标注"
            : string.Join("      ", _events.Select(ev => $"{ev.Label} @ {MainViewModel.TimeLabel(ev.T)}"));

        BuildKpi();
        BuildCsi();
        BuildBsr();
        BuildDsa();
        BuildSpi();
        BuildVitals();
        BuildBands();
        BuildEeg();
    }

    // ─────────────────────────── KPI 概览 ───────────────────────────
    private void BuildKpi()
    {
        var k = _session.GetKpi();
        KpiGrid.Children.Clear();

        AddKpi("总时长", MainViewModel.TimeLabel(k.DurationSec), "", false);
        AddKpi("CSI 靶区占比", Pct(k.CsiTargetPct), "40–60", false);
        AddKpi("过深占比", Pct(k.CsiDeepPct), "<40", k.CsiDeepPct >= 15);   // 过深偏多染红
        AddKpi("偏浅占比", Pct(k.CsiLightPct), ">60（唤醒风险）", k.CsiLightPct >= 15);
        AddKpi("平均爆发抑制", double.IsNaN(k.MeanBsr) ? "--" : $"{k.MeanBsr:0.0}%",
               $"累计 {MainViewModel.TimeLabel(k.SuppressionSec)}", k.SuppressionSec >= 60);
        AddKpi("SpO₂ 最低", double.IsNaN(k.Spo2Min) ? "--" : $"{k.Spo2Min:0}%",
               k.DesatEvents > 0 ? $"低氧 {k.DesatEvents} 次 / {MainViewModel.TimeLabel(k.DesatSec)}" : "无低氧事件",
               k.DesatEvents > 0);
        AddKpi("脉搏范围", double.IsNaN(k.PrMin) ? "--" : $"{k.PrMin:0}–{k.PrMax:0}",
               double.IsNaN(k.PrMean) ? "" : $"均 {k.PrMean:0} bpm", false);
        AddKpi("SPI 靶区占比", Pct(k.SpiTargetPct), "20–50", false);
    }

    private static string Pct(double v) => double.IsNaN(v) ? "--" : $"{v:0}%";

    private void AddKpi(string label, string value, string sub, bool warn)
    {
        var tile = new Border
        {
            CornerRadius = new CornerRadius(9),
            Background = new SolidColorBrush(Color.FromRgb(0x14, 0x1E, 0x33)),
            BorderBrush = new SolidColorBrush(warn ? Color.FromRgb(0x5A, 0x25, 0x2A) : Color.FromRgb(0x24, 0x34, 0x50)),
            BorderThickness = new Thickness(1),
            Margin = new Thickness(4),
            Padding = new Thickness(11, 9, 11, 9),
        };
        var sp = new StackPanel();
        sp.Children.Add(new TextBlock
        {
            Text = label, FontSize = 11,
            Foreground = new SolidColorBrush(Color.FromRgb(0xAE, 0xC2, 0xDC)),
        });
        sp.Children.Add(new TextBlock
        {
            Text = value, FontSize = 24, FontWeight = System.Windows.FontWeights.Bold, Margin = new Thickness(0, 3, 0, 0),
            Foreground = new SolidColorBrush(warn ? Color.FromRgb(0xFB, 0x5C, 0x6E) : Color.FromRgb(0xDD, 0xE7, 0xF5)),
        });
        if (!string.IsNullOrEmpty(sub))
            sp.Children.Add(new TextBlock
            {
                Text = sub, FontSize = 10, Margin = new Thickness(0, 2, 0, 0),
                Foreground = new SolidColorBrush(warn ? Color.FromRgb(0xFB, 0x5C, 0x6E) : Color.FromRgb(0x7C, 0x90, 0xB0)),
            });
        tile.Child = sp;
        KpiGrid.Children.Add(tile);
    }

    // ─────────────────────────── 面板 ───────────────────────────

    /// <summary>建一个共用时间轴（0→总时长，mm:ss）的模型，并加事件竖线（贯穿所有面板的关键）。</summary>
    private PlotModel NewTimeModel(double yMin, double yMax, bool showEventLabels = false)
    {
        var m = new PlotModel
        {
            PlotMargins = new OxyThickness(44, 4, 8, 20), Padding = new OxyThickness(0),
            Background = OxyColors.Transparent, TextColor = Tick,
        };
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom, Minimum = 0, Maximum = _duration,
            FontSize = 10, TextColor = Tick, TicklineColor = Grid,
            MajorGridlineStyle = LineStyle.Solid, MajorGridlineColor = Grid,
            LabelFormatter = MainViewModel.TimeLabel,
        });
        m.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left, Minimum = yMin, Maximum = yMax,
            FontSize = 10, TextColor = Tick, TicklineColor = Grid, StringFormat = "0.#",
            MajorGridlineStyle = LineStyle.Solid, MajorGridlineColor = Grid,
        });
        AddEventLines(m, showEventLabels);
        return m;
    }

    /// <summary>在每个面板画同样的事件竖线——一条竖线贯穿全部面板即可读出联动。
    /// 图上只标事件<b>编号</b>（如 #1），字号加大；全名在 CSI 卡的对照行读，避免图内堆叠中文拥挤。</summary>
    private void AddEventLines(PlotModel m, bool withLabels)
    {
        foreach (var ev in _events)
        {
            m.Annotations.Add(new LineAnnotation
            {
                Type = LineAnnotationType.Vertical, X = ev.T,
                Color = EventColor, LineStyle = LineStyle.Dash, StrokeThickness = 1,
                Text = withLabels ? EventNumber(ev.Label) : null,
                TextColor = OxyColor.FromRgb(0xFB, 0xBF, 0x24), FontSize = 13, FontWeight = OxyPlot.FontWeights.Bold,
                TextOrientation = AnnotationTextOrientation.Vertical,
                TextVerticalAlignment = OxyPlot.VerticalAlignment.Top,
                TextMargin = 2,
            });
        }
    }

    /// <summary>从事件标签「#1 气管插管」取出编号「#1」；无编号前缀则原样返回。</summary>
    private static string EventNumber(string label)
    {
        if (string.IsNullOrEmpty(label) || label[0] != '#') return label;
        int sp = label.IndexOf(' ');
        return sp > 0 ? label[..sp] : label;
    }

    /// <summary>目标区间绿带（或低氧红带）。</summary>
    private static void AddBand(PlotModel m, double lo, double hi, byte r, byte g, byte b)
    {
        m.Annotations.Insert(0, new RectangleAnnotation
        {
            MinimumX = double.NaN, MaximumX = double.NaN,   // 横向铺满
            MinimumY = lo, MaximumY = hi,
            Fill = OxyColor.FromArgb(0x30, r, g, b), StrokeThickness = 0,
        });
    }

    private static LineSeries Line(string title, byte r, byte g, byte b, LineStyle style = LineStyle.Solid, string? yAxisKey = null)
        => new()
        {
            Title = title, Color = OxyColor.FromRgb(r, g, b), StrokeThickness = 1.7, LineStyle = style,
            YAxisKey = yAxisKey,
        };

    private static void Fill(LineSeries ls, IEnumerable<(double t, double v)> pts)
    {
        foreach (var (t, v) in pts) if (double.IsFinite(v)) ls.Points.Add(new DataPoint(t, v));
    }

    private void AddLegend(PlotModel m) => m.Legends.Add(new OxyPlot.Legends.Legend
    {
        LegendPosition = OxyPlot.Legends.LegendPosition.TopRight, LegendTextColor = LegendText, LegendFontSize = 11,
        LegendBackground = OxyColor.FromArgb(0xB0, 0x0C, 0x14, 0x24), LegendPadding = 4,
    });

    // 1. CSI + 靶区 40–60 绿带（顶部面板带事件标签）
    private void BuildCsi()
    {
        var m = NewTimeModel(0, 100, showEventLabels: true);
        AddBand(m, 40, 60, 0x34, 0xD3, 0x99);
        var trend = SessionBuffer.Downsample(_session.GetTrend(), MaxTrendPoints);
        var csi = Line("CSI", 0xA7, 0x8B, 0xFA);
        Fill(csi, trend.Select(t => (t.T, t.Csi)));
        m.Series.Add(csi);
        CsiPlot.Model = m;
    }

    // 2. 爆发抑制 BSR
    private void BuildBsr()
    {
        var m = NewTimeModel(0, 100);
        var trend = SessionBuffer.Downsample(_session.GetTrend(), MaxTrendPoints);
        var bsr = new AreaSeries { Title = "BSR", Color = OxyColor.FromRgb(0xF4, 0x72, 0xB6), Fill = OxyColor.FromArgb(0x55, 0xF4, 0x72, 0xB6), StrokeThickness = 1.4 };
        foreach (var t in trend) if (double.IsFinite(t.Bs)) bsr.Points.Add(new DataPoint(t.T, t.Bs));
        m.Series.Add(bsr);
        BsrPlot.Model = m;
    }

    // 3. DSA 频谱（时间轴与其它面板一致：0→duration）
    private void BuildDsa()
    {
        var m = new PlotModel { PlotMargins = new OxyThickness(44, 4, 8, 20), Padding = new OxyThickness(0),
            Background = OxyColors.Transparent, TextColor = Tick };
        var (cols, bins) = _session.GetDsa();
        // 底轴：与其它面板一致的时间轴
        var bottom = new LinearAxis { Position = AxisPosition.Bottom, Minimum = 0, Maximum = _duration,
            FontSize = 10, TextColor = Tick, TicklineColor = Grid, LabelFormatter = MainViewModel.TimeLabel };
        m.Axes.Add(bottom);
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = 1, Maximum = bins > 0 ? bins : 44,
            Title = "Hz", FontSize = 10, TitleFontSize = 10, MajorStep = 10, MinorStep = 5, StringFormat = "0.#", TextColor = Tick, TicklineColor = Grid });

        if (cols.Length > 0 && bins > 0)
        {
            int step = Math.Max(1, cols.Length / MaxDsaColumns);
            int outCols = (cols.Length + step - 1) / step;
            var data = new double[outCols, bins];
            for (int c = 0; c < outCols; c++)
            {
                var col = cols[Math.Min(cols.Length - 1, c * step)];
                for (int y = 0; y < bins && y < col.Length; y++) data[c, y] = col[y];
            }
            m.Axes.Add(new LinearColorAxis { Position = AxisPosition.Right, Palette = OxyPalettes.Jet(256),
                Minimum = 0, Maximum = 255, IsAxisVisible = false, LowColor = OxyColors.Transparent });
            // DSA 列按时间铺到 [0, duration]，与其它面板同轴
            m.Series.Add(new HeatMapSeries { X0 = 0, X1 = _duration, Y0 = 1, Y1 = bins,
                Data = data, Interpolate = true, RenderMethod = HeatMapRenderMethod.Bitmap });
        }
        AddEventLines(m, false);
        DsaPlot.Model = m;
    }

    // 4. SPI + NOX，SPI 靶区 20–50 绿带
    private void BuildSpi()
    {
        var m = NewTimeModel(0, 100);
        AddBand(m, 20, 50, 0x34, 0xD3, 0x99);
        AddLegend(m);
        var trend = SessionBuffer.Downsample(_session.GetTrend(), MaxTrendPoints);
        var spi = Line("SPI", 0xFB, 0x92, 0x3C);
        var nox = Line("NOX", 0x2D, 0xD4, 0xBF);
        Fill(spi, trend.Select(t => (t.T, t.Spi)));
        Fill(nox, trend.Select(t => (t.T, t.Nox)));
        m.Series.Add(spi); m.Series.Add(nox);
        SpiPlot.Model = m;
    }

    // 5. 血氧 + 脉搏（双 Y 轴），SpO₂<90 红带
    private void BuildVitals()
    {
        var m = new PlotModel { PlotMargins = new OxyThickness(44, 4, 44, 20), Padding = new OxyThickness(0),
            Background = OxyColors.Transparent, TextColor = Tick };
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Minimum = 0, Maximum = _duration,
            FontSize = 10, TextColor = Tick, TicklineColor = Grid, MajorGridlineStyle = LineStyle.Solid, MajorGridlineColor = Grid,
            LabelFormatter = MainViewModel.TimeLabel });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Key = "spo2", Title = "SpO₂ %", FontSize = 10, TitleFontSize = 10,
            Minimum = 80, Maximum = 100, StringFormat = "0.#", TextColor = Tick, TicklineColor = Grid, MajorGridlineStyle = LineStyle.Solid, MajorGridlineColor = Grid });
        m.Axes.Add(new LinearAxis { Position = AxisPosition.Right, Key = "pr", Title = "脉搏 bpm", FontSize = 10, TitleFontSize = 10,
            Minimum = 30, Maximum = 150, StringFormat = "0.#", TextColor = Tick, TicklineColor = Grid });
        // <90 低氧红带（挂 spo2 轴）
        m.Annotations.Insert(0, new RectangleAnnotation { MinimumY = 80, MaximumY = 90, YAxisKey = "spo2",
            Fill = OxyColor.FromArgb(0x28, 0xFB, 0x5C, 0x6E), StrokeThickness = 0 });
        AddEventLines(m, false);
        AddLegend(m);

        var vitals = SessionBuffer.Downsample(_session.GetVitals(), MaxTrendPoints);
        var spo2 = Line("SpO₂", 0x38, 0xBD, 0xF8, yAxisKey: "spo2");
        var pr = Line("脉搏", 0xFB, 0x71, 0x85, yAxisKey: "pr");
        Fill(spo2, vitals.Select(v => (v.T, v.Spo2)));
        Fill(pr, vitals.Select(v => (v.T, v.Pr)));
        m.Series.Add(spo2); m.Series.Add(pr);
        VitalPlot.Model = m;
    }

    // 6. 脑电成分随时间：δθαβγ 100% 堆叠面积
    private void BuildBands()
    {
        var m = NewTimeModel(0, 100);
        AddLegend(m);
        var bands = SessionBuffer.Downsample(_session.GetBands(), MaxTrendPoints);

        // 堆叠：累加得到各层上边界，画成填充面积（从上层往下画，后画的盖住前面的底部）
        (string name, byte r, byte g, byte b, Func<SessionBuffer.BandSample, double> sel)[] layers =
        {
            ("δ", 0x60, 0xA5, 0xFA, s => s.Delta),
            ("θ", 0x22, 0xD3, 0xEE, s => s.Theta),
            ("α", 0x34, 0xD3, 0x99, s => s.Alpha),
            ("β", 0xFB, 0xBF, 0x24, s => s.Beta),
            ("γ", 0xFB, 0x71, 0x85, s => s.Gamma),
        };
        // 逐样本算累计上边界；从最外层（全部之和=100）往里画，每层面积填到 0
        for (int li = layers.Length - 1; li >= 0; li--)
        {
            var area = new AreaSeries { Title = layers[li].name, Color = OxyColors.Transparent,
                Fill = OxyColor.FromRgb(layers[li].r, layers[li].g, layers[li].b), StrokeThickness = 0,
                Tag = bands };   // 悬停框据此读取各成分原始占比（面积画的是累计值）
            foreach (var s in bands)
            {
                double cum = 0;
                for (int k = 0; k <= li; k++) cum += layers[k].sel(s);
                area.Points.Add(new DataPoint(s.T, cum));
                area.Points2.Add(new DataPoint(s.T, 0));
            }
            m.Series.Add(area);
        }
        BandPlot.Model = m;
    }

    // 7. 原始脑电包络（折叠区）
    private void BuildEeg()
    {
        var m = NewTimeModel(-140, 140);
        var (x, lo, hi) = _session.GetEegEnvelope(MaxEegColumns);
        if (x.Length > 0)
        {
            var area = new AreaSeries { Color = OxyColor.FromRgb(0x2F, 0xE6, 0xD6), Color2 = OxyColor.FromRgb(0x2F, 0xE6, 0xD6),
                Fill = OxyColor.FromArgb(0x66, 0x2F, 0xE6, 0xD6), StrokeThickness = 0.6 };
            for (int i = 0; i < x.Length; i++) { area.Points.Add(new DataPoint(x[i], hi[i])); area.Points2.Add(new DataPoint(x[i], lo[i])); }
            m.Series.Add(area);
        }
        EegPlot.Model = m;
    }
}
