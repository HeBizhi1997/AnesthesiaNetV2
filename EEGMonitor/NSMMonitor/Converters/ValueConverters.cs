using System.Globalization;
using System.Text;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using NSMMonitor.Services;
using NSMMonitor.ViewModels;
using OxyPlot;
using OxyPlot.Series;

namespace NSMMonitor.Converters;

internal static class Palette
{
    public static readonly Brush Green = New(0x00, 0xE6, 0x76);
    public static readonly Brush Amber = New(0xFF, 0xB3, 0x00);
    public static readonly Brush Red   = New(0xFF, 0x45, 0x60);
    public static readonly Brush Gray  = New(0x3D, 0x5A, 0x7A);
    public static readonly Brush Cyan  = New(0x00, 0xC8, 0xFF);
    public static Brush New(byte r, byte g, byte b) => new SolidColorBrush(Color.FromRgb(r, g, b));
}

/// <summary>CSI（麻醉深度）→ 颜色：40-60 绿，20-40/60-80 黄，其余红。</summary>
public sealed class CsiColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not double v || double.IsNaN(v)) return Palette.Gray;
        return v switch
        {
            >= 40 and <= 60 => Palette.Green,
            (>= 20 and < 40) or (> 60 and <= 80) => Palette.Amber,
            _ => Palette.Red,
        };
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>NOX（镇痛指数）→ 颜色：≤50 绿，50-65 黄，>65 红。</summary>
public sealed class NoxColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not double v || double.IsNaN(v)) return Palette.Gray;
        return v switch
        {
            <= 50 => Palette.Green,
            < 65 => Palette.Amber,
            _ => Palette.Red,
        };
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>SQI（信号质量）→ 颜色：≥70 绿，50-70 黄，&lt;50 红。</summary>
public sealed class SqiColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not double v || double.IsNaN(v)) return Palette.Gray;
        return v switch
        {
            >= 70 => Palette.Green,
            >= 50 => Palette.Amber,
            _ => Palette.Red,
        };
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>SPI（伤害感受指数，PPG 导出）→ 颜色：20-50 靶区绿，&lt;20 镇痛偏深黄，&gt;50 镇痛不足红。</summary>
public sealed class SpiColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not double v || double.IsNaN(v)) return Palette.Gray;
        return v switch
        {
            >= 20 and <= 50 => Palette.Green,
            < 20 => Palette.Amber,
            _ => Palette.Red,
        };
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>SpO₂ → 颜色：≥95 绿，90-94 黄，&lt;90 红。</summary>
public sealed class Spo2ColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not double v || double.IsNaN(v)) return Palette.Gray;
        return v switch
        {
            >= 95 => Palette.Green,
            >= 90 => Palette.Amber,
            _ => Palette.Red,
        };
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>脉率 PR → 颜色：50-100 绿，40-49 / 101-120 黄，其余红。</summary>
public sealed class PrColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not double v || double.IsNaN(v)) return Palette.Gray;
        return v switch
        {
            >= 50 and <= 100 => Palette.Green,
            (>= 40 and < 50) or (> 100 and <= 120) => Palette.Amber,
            _ => Palette.Red,
        };
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>
/// 数值 → 生命体征格底部区间条的网格列宽。
/// 参数格式 "min,max,part"，part 为 fill（0→value 段）或 rest（value→max 段）。
/// 无效值（NaN）时 fill 归零，条形整体留空。
/// </summary>
public sealed class RangeBarConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        double v = value is double d ? d : (value is int i ? i : double.NaN);
        var parts = (parameter as string ?? "0,100,fill").Split(',');
        bool fill = parts.Length < 3 || parts[2].Trim() == "fill";

        if (double.IsNaN(v)) return new GridLength(fill ? 0 : 1, GridUnitType.Star);

        double min = double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var lo) ? lo : 0;
        double max = double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var hi) ? hi : 100;
        double frac = max > min ? Math.Clamp((v - min) / (max - min), 0, 1) : 0;
        return new GridLength(fill ? frac : 1 - frac, GridUnitType.Star);
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>bool → Visibility，参数 "Inverse" 取反。</summary>
public sealed class BoolToVisConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool flag = value is bool b && b;
        if (parameter as string == "Inverse") flag = !flag;
        return flag ? Visibility.Visible : Visibility.Collapsed;
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>bool 取反（用于 IsEnabled 等需要 bool 的场景）。</summary>
public sealed class InverseBoolConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        => !(value is bool b && b);
    public object ConvertBack(object value, Type t, object p, CultureInfo c)
        => !(value is bool b && b);
}

/// <summary>频带功率 dB（约 -40..40）→ 进度条宽度比例 0..1（用于条形可视化）。</summary>
public sealed class DbToFractionConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        double v = value is double d ? d : (value is int i ? i : 0);
        double frac = (v + 40) / 80.0;          // -40→0, 40→1
        return Math.Max(0.02, Math.Min(1.0, frac));
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}

/// <summary>
/// OxyPlot 悬停框文本：把默认的「X: 秒 / Y: 值」改写为「时间 mm:ss + 该时刻各指标值」。
///
/// 绑定对象是整个 <see cref="TrackerHitResult"/>：
///   • 时间 = <c>DataPoint.X</c>（横轴为手术计时秒）经 <see cref="MainViewModel.TimeLabel"/> 换算成 mm:ss；
///   • 若 <c>PlotModel.Tag</c> 挂了整场频带样本（脑电成分面积图），列出 δ/θ/α/β/γ 各自原始占比；
///   • 否则列出模型内所有带标题折线在该时刻的值（趋势/血氧/伤害感受一次性看全）。
/// 数值一律最多一位小数。
/// </summary>
public sealed class TrackerTextConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not TrackerHitResult hit) return value?.ToString() ?? "";
        double x = hit.DataPoint.X;
        var model = hit.PlotModel ?? hit.Series?.PlotModel;

        var sb = new StringBuilder();
        sb.Append("时间   ").Append(MainViewModel.TimeLabel(x));

        // 脑电成分面积图：序列 Tag 上挂着整场频带样本，直接给出各成分原始占比（面积序列画的是累计值，不能直接读）
        if (hit.Series?.Tag is IReadOnlyList<SessionBuffer.BandSample> bands && bands.Count > 0)
        {
            var b = NearestBand(bands, x);
            sb.Append('\n').Append("δ   ").Append(b.Delta.ToString("0.0", culture)).Append('%');
            sb.Append('\n').Append("θ   ").Append(b.Theta.ToString("0.0", culture)).Append('%');
            sb.Append('\n').Append("α   ").Append(b.Alpha.ToString("0.0", culture)).Append('%');
            sb.Append('\n').Append("β   ").Append(b.Beta.ToString("0.0", culture)).Append('%');
            sb.Append('\n').Append("γ   ").Append(b.Gamma.ToString("0.0", culture)).Append('%');
            return sb.ToString();
        }

        // 通用：列出模型内所有带标题的折线/面积序列在该时刻的值
        if (model != null)
        {
            foreach (var s in model.Series)
            {
                if (s is LineSeries ls && !string.IsNullOrEmpty(ls.Title) && ls.Points.Count > 0)
                {
                    double y = ValueAt(ls.Points, x);
                    if (double.IsFinite(y))
                        sb.Append('\n').Append(ls.Title).Append("   ").Append(y.ToString("0.#", culture));
                }
            }
        }
        return sb.ToString();
    }

    private static SessionBuffer.BandSample NearestBand(IReadOnlyList<SessionBuffer.BandSample> b, double x)
    {
        int lo = 0, hi = b.Count - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (b[mid].T < x) lo = mid + 1; else hi = mid; }
        if (lo > 0 && Math.Abs(b[lo - 1].T - x) <= Math.Abs(b[lo].T - x)) lo--;
        return b[lo];
    }

    private static double ValueAt(IList<DataPoint> pts, double x)
    {
        int n = pts.Count;
        if (n == 0) return double.NaN;
        int lo = 0, hi = n - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (pts[mid].X < x) lo = mid + 1; else hi = mid; }
        if (lo > 0 && Math.Abs(pts[lo - 1].X - x) <= Math.Abs(pts[lo].X - x)) lo--;
        return pts[lo].Y;
    }

    public object ConvertBack(object value, Type t, object p, CultureInfo c) => Binding.DoNothing;
}

/// <summary>
/// 频带功率 dB → 网格列宽 GridLength（星比例），用于横向渐变条。
/// 参数 "fill" 返回填充比例，"rest" 返回剩余比例。映射区间 -25..12 dB。
/// </summary>
public sealed class DbToGridLengthConverter : IValueConverter
{
    private const double Min = -25, Max = 12;

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        double v = value is double d ? d : (value is int i ? i : 0);
        double frac = Math.Clamp((v - Min) / (Max - Min), 0.03, 1.0);
        bool fill = parameter as string == "fill";
        return new GridLength(fill ? frac : 1 - frac, GridUnitType.Star);
    }
    public object ConvertBack(object value, Type t, object p, CultureInfo c) => DependencyProperty.UnsetValue;
}
