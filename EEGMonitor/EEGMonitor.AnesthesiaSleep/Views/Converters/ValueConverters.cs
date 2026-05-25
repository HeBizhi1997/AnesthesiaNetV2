using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using EEGMonitor.AnesthesiaSleep.Infrastructure.Simulation;
using EEGMonitor.AnesthesiaSleep.Models;

namespace EEGMonitor.AnesthesiaSleep.Views.Converters;

public class StageColorConverter : IValueConverter
{
    public object? Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is SleepStage stage)
        {
            var color = stage switch
            {
                SleepStage.Wake => Color.FromRgb(0x6B, 0x70, 0x80),
                SleepStage.N1 => Color.FromRgb(0xF0, 0xA0, 0x20),
                SleepStage.N2 => Color.FromRgb(0x00, 0xC8, 0xFF),
                SleepStage.N3 => Color.FromRgb(0x5B, 0x8E, 0xFF),
                SleepStage.REM => Color.FromRgb(0xE0, 0x40, 0xC0),
                SleepStage.Movement => Color.FromRgb(0xFF, 0x45, 0x60),
                _ => Color.FromRgb(0x40, 0x40, 0x40)
            };
            return new SolidColorBrush(color);
        }
        return new SolidColorBrush(Color.FromRgb(0x40, 0x40, 0x40));
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class SpO2ColorConverter : IValueConverter
{
    public object? Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double spo2)
        {
            var color = spo2 switch
            {
                >= 95 => Color.FromRgb(0x00, 0xE6, 0x76),
                >= 90 => Color.FromRgb(0xFF, 0xB3, 0x00),
                >= 85 => Color.FromRgb(0xFF, 0x8C, 0x00),
                _ => Color.FromRgb(0xFF, 0x45, 0x60)
            };
            return new SolidColorBrush(color);
        }
        return new SolidColorBrush(Color.FromRgb(0x40, 0x40, 0x40));
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class StageLabelConverter : IValueConverter
{
    public object? Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is SleepStage stage)
        {
            return stage switch
            {
                SleepStage.Wake => "清醒",
                SleepStage.N1 => "N1（浅睡）",
                SleepStage.N2 => "N2（中睡）",
                SleepStage.N3 => "N3（深睡）",
                SleepStage.REM => "REM（快动眼）",
                SleepStage.Movement => "体动",
                _ => "---"
            };
        }
        return "---";
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class BoolToVisConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool b = value is true;
        if (parameter is string param && param.Contains('|'))
        {
            var parts = param.Split('|');
            if (parts.Length >= 2)
                return b ? ParseValue(parts[0]) : ParseValue(parts[1]);
        }
        return b ? Visibility.Visible : Visibility.Collapsed;
    }

    private static object ParseValue(string s)
    {
        if (s.StartsWith("#") && s.Length == 7)
            return new SolidColorBrush((Color)ColorConverter.ConvertFromString(s));
        return s switch
        {
            "Visible" => Visibility.Visible,
            "Collapsed" => Visibility.Collapsed,
            "Hidden" => Visibility.Hidden,
            _ => s
        };
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class InverseBoolToVisConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return value is true ? Visibility.Collapsed : Visibility.Visible;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class ScenarioNameConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is SleepScenarioType s)
            return SleepSimulator.GetScenarioName(s);
        return "";
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class ScenarioDescConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is SleepScenarioType s)
            return SleepSimulator.GetScenarioDescription(s);
        return "";
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class PercentageToStarWidthConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double pct && !double.IsNaN(pct) && pct > 0)
            return new GridLength(Math.Max(pct, 1), GridUnitType.Star);
        return new GridLength(0, GridUnitType.Star);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public class NaNToStringConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double d)
        {
            if (double.IsNaN(d))
                return "---";
            string fmt = parameter is string s && s.StartsWith("F") ? s : "F0";
            return d.ToString(fmt);
        }
        return value?.ToString() ?? "---";
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}
