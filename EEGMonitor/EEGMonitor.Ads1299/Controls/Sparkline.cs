using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Media;

namespace Ads1299Monitor.Controls;

/// <summary>Minimal trend sparkline drawn in OnRender. Bind <see cref="Values"/> to a rolling
/// double[]; the VM raises PropertyChanged with a fresh array each update.</summary>
public sealed class Sparkline : FrameworkElement
{
    public static readonly DependencyProperty ValuesProperty = DependencyProperty.Register(
        nameof(Values), typeof(IReadOnlyList<double>), typeof(Sparkline),
        new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty StrokeProperty = DependencyProperty.Register(
        nameof(Stroke), typeof(Brush), typeof(Sparkline),
        new FrameworkPropertyMetadata(Brushes.LimeGreen, FrameworkPropertyMetadataOptions.AffectsRender));

    // Fixed clinical Y-range. When set (not NaN), the trace position reflects the ABSOLUTE value
    // (e.g. SpO₂ 99 sits near the top of a 90–100 scale) instead of auto-scaling tiny wiggles to
    // full height — which made the waveform look unrelated to the big number beside it.
    public static readonly DependencyProperty MinProperty = DependencyProperty.Register(
        nameof(Min), typeof(double), typeof(Sparkline),
        new FrameworkPropertyMetadata(double.NaN, FrameworkPropertyMetadataOptions.AffectsRender));
    public static readonly DependencyProperty MaxProperty = DependencyProperty.Register(
        nameof(Max), typeof(double), typeof(Sparkline),
        new FrameworkPropertyMetadata(double.NaN, FrameworkPropertyMetadataOptions.AffectsRender));

    public IReadOnlyList<double>? Values
    {
        get => (IReadOnlyList<double>?)GetValue(ValuesProperty);
        set => SetValue(ValuesProperty, value);
    }
    public Brush Stroke { get => (Brush)GetValue(StrokeProperty); set => SetValue(StrokeProperty, value); }
    public double Min { get => (double)GetValue(MinProperty); set => SetValue(MinProperty, value); }
    public double Max { get => (double)GetValue(MaxProperty); set => SetValue(MaxProperty, value); }

    protected override void OnRender(DrawingContext dc)
    {
        var vals = Values;
        double w = ActualWidth, h = ActualHeight;
        if (vals == null || vals.Count < 2 || w < 2 || h < 2) return;

        // Use the fixed range when supplied; otherwise fall back to auto-scaling.
        double min = Min, max = Max;
        if (double.IsNaN(min) || double.IsNaN(max) || max <= min)
        {
            min = vals.Min(); max = vals.Max();
            if (max - min < 1e-9) { min -= 1; max += 1; }
        }

        var pen = new Pen(Stroke, 1.5);
        pen.Freeze();
        var geo = new StreamGeometry();
        using (var ctx = geo.Open())
        {
            for (int i = 0; i < vals.Count; i++)
            {
                double x = w * i / (vals.Count - 1);
                double norm = (vals[i] - min) / (max - min);
                if (norm < 0) norm = 0; else if (norm > 1) norm = 1;     // clamp out-of-range to the edge
                double y = h - norm * h * 0.9 - h * 0.05;
                if (i == 0) ctx.BeginFigure(new Point(x, y), false, false);
                else ctx.LineTo(new Point(x, y), true, false);
            }
        }
        geo.Freeze();
        dc.DrawGeometry(null, pen, geo);
    }
}
