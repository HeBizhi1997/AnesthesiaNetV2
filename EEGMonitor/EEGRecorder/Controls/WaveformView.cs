using System.Windows;
using System.Windows.Media;
using EEGRecorder.Controls;

namespace EEGRecorder.Controls;

/// <summary>Scrolling waveform painter. Snapshots its <see cref="Source"/> every render tick (~vsync,
/// throttled) and draws a polyline with a centre baseline. Auto-scales to the visible window so both
/// tiny EEG and large PPG swings stay framed.</summary>
public sealed class WaveformView : FrameworkElement
{
    public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(
        nameof(Source), typeof(WaveformSource), typeof(WaveformView),
        new FrameworkPropertyMetadata(null));

    public static readonly DependencyProperty StrokeProperty = DependencyProperty.Register(
        nameof(Stroke), typeof(Brush), typeof(WaveformView),
        new FrameworkPropertyMetadata(Brushes.DeepSkyBlue));

    public static readonly DependencyProperty StrokeThicknessProperty = DependencyProperty.Register(
        nameof(StrokeThickness), typeof(double), typeof(WaveformView),
        new FrameworkPropertyMetadata(1.2));

    public WaveformSource? Source { get => (WaveformSource?)GetValue(SourceProperty); set => SetValue(SourceProperty, value); }
    public Brush Stroke { get => (Brush)GetValue(StrokeProperty); set => SetValue(StrokeProperty, value); }
    public double StrokeThickness { get => (double)GetValue(StrokeThicknessProperty); set => SetValue(StrokeThicknessProperty, value); }

    private double[] _scratch = Array.Empty<double>();
    private Pen? _pen;
    private readonly Pen _baselinePen = new(new SolidColorBrush(Color.FromArgb(0x22, 0xFF, 0xFF, 0xFF)), 1);
    private TimeSpan _last;

    public WaveformView()
    {
        _baselinePen.Freeze();
        Loaded += (_, _) => CompositionTarget.Rendering += OnTick;
        Unloaded += (_, _) => CompositionTarget.Rendering -= OnTick;
    }

    private void OnTick(object? sender, EventArgs e)
    {
        // Throttle to ~33 fps regardless of monitor refresh.
        if (e is RenderingEventArgs r)
        {
            if (r.RenderingTime - _last < TimeSpan.FromMilliseconds(30)) return;
            _last = r.RenderingTime;
        }
        InvalidateVisual();
    }

    protected override void OnRender(DrawingContext dc)
    {
        double w = ActualWidth, h = ActualHeight;
        if (w < 2 || h < 2) return;

        // Centre baseline.
        dc.DrawLine(_baselinePen, new Point(0, h / 2), new Point(w, h / 2));

        var src = Source;
        if (src == null) return;
        if (_scratch.Length != src.Capacity) _scratch = new double[src.Capacity];
        int n = src.Snapshot(_scratch);
        if (n < 2) return;

        double min = double.MaxValue, max = double.MinValue;
        for (int i = 0; i < n; i++) { double v = _scratch[i]; if (v < min) min = v; if (v > max) max = v; }
        double span = max - min;
        if (span < 1e-9) span = 1;

        if (_pen == null || !ReferenceEquals(_penBrush, Stroke) || _pen.Thickness != StrokeThickness)
        {
            _penBrush = Stroke;
            _pen = new Pen(Stroke, StrokeThickness);
            _pen.Freeze();
        }

        var geo = new StreamGeometry();
        using (var ctx = geo.Open())
        {
            for (int i = 0; i < n; i++)
            {
                double x = w * i / (n - 1);
                double y = h - (_scratch[i] - min) / span * h * 0.9 - h * 0.05;
                if (i == 0) ctx.BeginFigure(new Point(x, y), false, false);
                else ctx.LineTo(new Point(x, y), true, false);
            }
        }
        geo.Freeze();
        dc.DrawGeometry(null, _pen, geo);
    }

    private Brush? _penBrush;
}
