using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

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
