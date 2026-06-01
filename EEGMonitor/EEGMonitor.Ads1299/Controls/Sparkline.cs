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

    public IReadOnlyList<double>? Values
    {
        get => (IReadOnlyList<double>?)GetValue(ValuesProperty);
        set => SetValue(ValuesProperty, value);
    }
    public Brush Stroke { get => (Brush)GetValue(StrokeProperty); set => SetValue(StrokeProperty, value); }

    protected override void OnRender(DrawingContext dc)
    {
        var vals = Values;
        double w = ActualWidth, h = ActualHeight;
        if (vals == null || vals.Count < 2 || w < 2 || h < 2) return;

        double min = vals.Min(), max = vals.Max();
        if (max - min < 1e-9) { min -= 1; max += 1; }

        var pen = new Pen(Stroke, 1.5);
        pen.Freeze();
        var geo = new StreamGeometry();
        using (var ctx = geo.Open())
        {
            for (int i = 0; i < vals.Count; i++)
            {
                double x = w * i / (vals.Count - 1);
                double y = h - (vals[i] - min) / (max - min) * h * 0.9 - h * 0.05;
                if (i == 0) ctx.BeginFigure(new Point(x, y), false, false);
                else ctx.LineTo(new Point(x, y), true, false);
            }
        }
        geo.Freeze();
        dc.DrawGeometry(null, pen, geo);
    }
}
