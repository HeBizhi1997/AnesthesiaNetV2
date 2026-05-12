using OxyPlot;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace EEGMonitor.Views.Converters;

/// <summary>
/// Colors BIS value: green (40-60), yellow (60-80 or 20-40), red (&lt;20 or &gt;80).
/// </summary>
[ValueConversion(typeof(double), typeof(Brush))]
public class BISColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not double bis || double.IsNaN(bis))
            return Brushes.Gray;
        return bis switch
        {
            >= 40 and <= 60 => new SolidColorBrush(Color.FromRgb(0x00, 0xE6, 0x76)), // vivid green
            (>= 20 and < 40) or (> 60 and <= 80) => new SolidColorBrush(Color.FromRgb(0xFF, 0xB3, 0x00)), // amber
            _ => new SolidColorBrush(Color.FromRgb(0xFF, 0x45, 0x60))  // coral red
        };
    }
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        DependencyProperty.UnsetValue;
}

/// <summary>
/// Converts double.NaN to "---" else passes value through.
/// </summary>
public class NaNToStringConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double d && double.IsNaN(d)) return "---";
        return value;
    }
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        DependencyProperty.UnsetValue;
}

/// <summary>
/// true → Visible, false → Collapsed.
/// ConverterParameter="TrueText|FalseText" converts bool to string.
/// ConverterParameter="#RRGGBB1|#RRGGBB2" converts bool to string brush.
/// </summary>
public class BoolToVisConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool flag = value is bool b && b;
        if (parameter is string param)
        {
            var parts = param.Split('|');
            if (parts.Length == 2)
            {
                var chosen = flag ? parts[0] : parts[1];
                if (targetType == typeof(Visibility))
                    return flag ? Visibility.Visible : Visibility.Collapsed;
                if (chosen.StartsWith('#') && targetType == typeof(Brush))
                {
                    try
                    {
                        var c = (Color)(ColorConverter.ConvertFromString(chosen) ?? Colors.Transparent);
                        return new SolidColorBrush(c);
                    }
                    catch { return Brushes.Transparent; }
                }
                return chosen;
            }
        }
        return flag ? Visibility.Visible : Visibility.Collapsed;
    }
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        DependencyProperty.UnsetValue;
}

/// <summary>
/// Colors fNox (inverse of BIS): green (&lt;40 adequate), amber (40-60 monitor), red (&gt;60 alert).
/// </summary>
[ValueConversion(typeof(double), typeof(Brush))]
public class NSIColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not double v || double.IsNaN(v))
            return Brushes.Gray;
        return v switch
        {
            <= 50 => new SolidColorBrush(Color.FromRgb(0x00, 0xE6, 0x76)), // green — target zone 30-50
            < 65  => new SolidColorBrush(Color.FromRgb(0xFF, 0xB3, 0x00)), // amber — monitor
            _     => new SolidColorBrush(Color.FromRgb(0xFF, 0x45, 0x60))  // red — analgesia failure
        };
    }
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        DependencyProperty.UnsetValue;
}

/// <summary>false → Visible, true → Collapsed.</summary>
public class InverseBoolToVisConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is bool b && b ? Visibility.Collapsed : Visibility.Visible;
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        DependencyProperty.UnsetValue;
}
