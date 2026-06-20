using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Effects;

namespace NSMMonitor.Controls;

/// <summary>
/// 圆环仪表盘：可设置数值、量程、起始角与扫掠角，绘制带圆角端点的进度弧，
/// 中心通过 <see cref="CenterContent"/> 承载自定义内容（大号数字 + 标签）。
/// </summary>
public partial class RingGauge : UserControl
{
    public RingGauge()
    {
        InitializeComponent();
        SizeChanged += (_, _) => Redraw();
        Loaded += (_, _) => Redraw();
    }

    public static readonly DependencyProperty ValueProperty = DependencyProperty.Register(
        nameof(Value), typeof(double), typeof(RingGauge), new PropertyMetadata(0.0, OnVisualChanged));
    public double Value { get => (double)GetValue(ValueProperty); set => SetValue(ValueProperty, value); }

    public static readonly DependencyProperty MaximumProperty = DependencyProperty.Register(
        nameof(Maximum), typeof(double), typeof(RingGauge), new PropertyMetadata(100.0, OnVisualChanged));
    public double Maximum { get => (double)GetValue(MaximumProperty); set => SetValue(MaximumProperty, value); }

    public static readonly DependencyProperty RingThicknessProperty = DependencyProperty.Register(
        nameof(RingThickness), typeof(double), typeof(RingGauge), new PropertyMetadata(12.0, OnVisualChanged));
    public double RingThickness { get => (double)GetValue(RingThicknessProperty); set => SetValue(RingThicknessProperty, value); }

    public static readonly DependencyProperty StartAngleProperty = DependencyProperty.Register(
        nameof(StartAngle), typeof(double), typeof(RingGauge), new PropertyMetadata(135.0, OnVisualChanged));
    public double StartAngle { get => (double)GetValue(StartAngleProperty); set => SetValue(StartAngleProperty, value); }

    public static readonly DependencyProperty SweepAngleProperty = DependencyProperty.Register(
        nameof(SweepAngle), typeof(double), typeof(RingGauge), new PropertyMetadata(270.0, OnVisualChanged));
    public double SweepAngle { get => (double)GetValue(SweepAngleProperty); set => SetValue(SweepAngleProperty, value); }

    public static readonly DependencyProperty TrackBrushProperty = DependencyProperty.Register(
        nameof(TrackBrush), typeof(Brush), typeof(RingGauge),
        new PropertyMetadata(new SolidColorBrush(Color.FromRgb(0x1E, 0x2A, 0x44)), OnVisualChanged));
    public Brush TrackBrush { get => (Brush)GetValue(TrackBrushProperty); set => SetValue(TrackBrushProperty, value); }

    public static readonly DependencyProperty ProgressBrushProperty = DependencyProperty.Register(
        nameof(ProgressBrush), typeof(Brush), typeof(RingGauge),
        new PropertyMetadata(new SolidColorBrush(Color.FromRgb(0x2F, 0xE6, 0xD6)), OnVisualChanged));
    public Brush ProgressBrush { get => (Brush)GetValue(ProgressBrushProperty); set => SetValue(ProgressBrushProperty, value); }

    public static readonly DependencyProperty GlowColorProperty = DependencyProperty.Register(
        nameof(GlowColor), typeof(Color), typeof(RingGauge), new PropertyMetadata(Colors.Transparent, OnVisualChanged));
    public Color GlowColor { get => (Color)GetValue(GlowColorProperty); set => SetValue(GlowColorProperty, value); }

    public static readonly DependencyProperty CenterContentProperty = DependencyProperty.Register(
        nameof(CenterContent), typeof(object), typeof(RingGauge), new PropertyMetadata(null));
    public object? CenterContent { get => GetValue(CenterContentProperty); set => SetValue(CenterContentProperty, value); }

    private static void OnVisualChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) => ((RingGauge)d).Redraw();

    private void Redraw()
    {
        double w = Root.ActualWidth, h = Root.ActualHeight;
        if (w <= 0 || h <= 0) return;

        double size = Math.Min(w, h);
        double cx = w / 2, cy = h / 2;
        double r = size / 2 - RingThickness / 2 - 2;
        if (r <= 0) return;

        TrackPath.Stroke = TrackBrush;
        TrackPath.StrokeThickness = RingThickness;
        TrackPath.Data = BuildArc(cx, cy, r, StartAngle, SweepAngle);

        double v = Value;
        if (double.IsNaN(v) || double.IsInfinity(v)) v = 0;
        v = Math.Clamp(v, 0, Maximum);
        double frac = Maximum > 0 ? v / Maximum : 0;

        ProgressPath.Stroke = ProgressBrush;
        ProgressPath.StrokeThickness = RingThickness;
        ProgressPath.Data = BuildArc(cx, cy, r, StartAngle, SweepAngle * frac);
        ProgressPath.Effect = GlowColor.A == 0
            ? null
            : new DropShadowEffect { Color = GlowColor, BlurRadius = 14, ShadowDepth = 0, Opacity = 0.75 };
    }

    private static Geometry BuildArc(double cx, double cy, double r, double startAngle, double sweep)
    {
        if (sweep <= 0.05) return Geometry.Empty;
        if (sweep >= 360) sweep = 359.999;

        Point start = PointOnCircle(cx, cy, r, startAngle);
        Point end = PointOnCircle(cx, cy, r, startAngle + sweep);
        bool isLarge = sweep > 180;

        var fig = new PathFigure { StartPoint = start, IsClosed = false };
        fig.Segments.Add(new ArcSegment(end, new Size(r, r), 0, isLarge, SweepDirection.Clockwise, true));
        var geo = new PathGeometry();
        geo.Figures.Add(fig);
        geo.Freeze();
        return geo;
    }

    private static Point PointOnCircle(double cx, double cy, double r, double angleDeg)
    {
        double rad = angleDeg * Math.PI / 180.0;
        return new Point(cx + r * Math.Cos(rad), cy + r * Math.Sin(rad));
    }
}
