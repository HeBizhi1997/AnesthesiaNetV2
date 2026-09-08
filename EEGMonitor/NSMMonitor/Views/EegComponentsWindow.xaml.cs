using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using NSMMonitor.Services;
using NSMMonitor.ViewModels;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace NSMMonitor.Views;

/// <summary>
/// 脑电成分分离弹窗：实时读取主界面当前 EEG 缓冲，按 δ/θ/α/β 频带做 FFT 带通分离并分行显示。
/// </summary>
public partial class EegComponentsWindow : Window
{
    private readonly MainViewModel _vm;
    private readonly DispatcherTimer _timer;

    private readonly (PlotModel Model, LineSeries Series, double Lo, double Hi, TextBlock DbText, TextBlock PctText)[] _bands;
    private readonly PlotModel _rawModel;
    private readonly LineSeries _rawSeries;

    public EegComponentsWindow(MainViewModel vm)
    {
        InitializeComponent();
        _vm = vm;

        (_rawModel, _rawSeries) = BuildPlot(OxyColor.FromRgb(0x2F, 0xE6, 0xD6), -130, 130);
        RawPlot.Model = _rawModel;

        var (dM, dS) = BuildPlot(OxyColor.FromRgb(0x60, 0xA5, 0xFA));
        var (tM, tS) = BuildPlot(OxyColor.FromRgb(0x22, 0xD3, 0xEE));
        var (aM, aS) = BuildPlot(OxyColor.FromRgb(0x34, 0xD3, 0x99));
        var (bM, bS) = BuildPlot(OxyColor.FromRgb(0xFB, 0xBF, 0x24), showTime: true);   // 底部 β 带共享时间轴
        DeltaPlot.Model = dM; ThetaPlot.Model = tM; AlphaPlot.Model = aM; BetaPlot.Model = bM;

        _bands = new[]
        {
            (dM, dS, 1.0, 3.0, DeltaDb, DeltaPctLbl),
            (tM, tS, 4.0, 7.0, ThetaDb, ThetaPctLbl),
            (aM, aS, 8.0, 13.0, AlphaDb, AlphaPctLbl),
            (bM, bS, 14.0, 30.0, BetaDb, BetaPctLbl),
        };

        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(300) };
        _timer.Tick += (_, _) => Refresh();
        Loaded += (_, _) => { Refresh(); _timer.Start(); };
        Closed += (_, _) => _timer.Stop();
    }

    private static readonly OxyColor AxisTick = OxyColor.FromRgb(0x3D, 0x5A, 0x7A);
    private static readonly OxyColor AxisGrid = OxyColor.FromRgb(0x18, 0x2C, 0x46);

    /// <summary>
    /// 单条成分波形的绘图：左侧固定显示 μV 幅度轴（整数/一位小数），
    /// 时间轴（秒，右缘=现在）只在最底部 <paramref name="showTime"/> 的那条上画一次，作为整叠的共享刻度。
    /// 所有面板左边距一致，波形纵向对齐，底部时间轴即可读出各成分同一时刻。
    /// </summary>
    private (PlotModel, LineSeries) BuildPlot(OxyColor color, double? min = null, double? max = null, bool showTime = false)
    {
        var m = new PlotModel
        {
            PlotMargins = new OxyThickness(52, 2, 8, showTime ? 30 : 4),
            Padding = new OxyThickness(0),
            Background = OxyColors.Transparent,
            TextColor = AxisTick,
        };
        var y = new LinearAxis
        {
            Position = AxisPosition.Left, Title = "μV",
            FontSize = 9, TitleFontSize = 9, StringFormat = "0.#",
            TextColor = AxisTick, TitleColor = AxisTick, TicklineColor = AxisGrid,
            MajorGridlineStyle = LineStyle.None,
        };
        if (min.HasValue && max.HasValue) { y.Minimum = min.Value; y.Maximum = max.Value; }
        m.Axes.Add(y);

        var x = new LinearAxis
        {
            Position = AxisPosition.Bottom, IsAxisVisible = showTime,
            FontSize = 9, TextColor = AxisTick, TicklineColor = AxisGrid,
            Title = showTime ? "时间 (s)" : null, TitleColor = AxisTick, TitleFontSize = 9,
            MajorGridlineStyle = LineStyle.None, MinorTickSize = 0,
        };
        // 右缘=0（现在），向左为负秒；样本索引按当前脑电采样率换算，只留整数秒
        x.LabelFormatter = v => $"{(v - x.ActualMaximum) / Math.Max(1, _vm.CurrentSampleRate):0}";
        m.Axes.Add(x);

        var s = new LineSeries { Color = color, StrokeThickness = 1.3 };
        m.Series.Add(s);
        return (m, s);
    }

    private void Refresh()
    {
        double[] samples = _vm.GetEegSamples();
        double fs = _vm.CurrentSampleRate;

        SetPoints(_rawSeries, samples);
        _rawModel.InvalidatePlot(true);

        // 第一遍：分离各成分并求线性功率（均方）
        var meanSquare = new double[_bands.Length];
        double total = 0;
        for (int i = 0; i < _bands.Length; i++)
        {
            var (model, series, lo, hi, _, _) = _bands[i];
            double[] comp = EegBandSeparator.BandPass(samples, fs, lo, hi);
            SetPoints(series, comp);
            model.InvalidatePlot(true);
            meanSquare[i] = MeanSquare(comp);
            total += meanSquare[i];
        }

        // 第二遍：由成分功率得到 dB 与占比（占比基于 δ/θ/α/β 四个成分）
        for (int i = 0; i < _bands.Length; i++)
        {
            double ms = meanSquare[i];
            _bands[i].DbText.Text = ms <= 1e-9 ? "-- dB" : $"{10 * Math.Log10(ms):0.0} dB";
            _bands[i].PctText.Text = total <= 1e-9 ? "--" : $"{100 * ms / total:0}%";
        }
    }

    private static double MeanSquare(double[] x)
    {
        if (x.Length == 0) return 0;
        double s = 0;
        foreach (var v in x) s += v * v;
        return s / x.Length;
    }

    private static void SetPoints(LineSeries series, double[] data)
    {
        series.Points.Clear();
        for (int i = 0; i < data.Length; i++)
            series.Points.Add(new DataPoint(i, data[i]));
    }
}
