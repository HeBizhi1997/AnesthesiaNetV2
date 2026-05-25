using CommunityToolkit.Mvvm.ComponentModel;
using EEGMonitor.AnesthesiaSleep.Models;
using EEGMonitor.Models;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class SpO2TrendViewModel : ObservableObject
{
    private const double DesatThreshold3 = 3.0;
    private const double DesatThreshold4 = 4.0;
    private const int BaselineWindowSeconds = 120;
    private const int DesatMinDuration = 10;

    private readonly Queue<(DateTime Time, double SpO2)> _spo2History = new(28800);
    private readonly List<DesaturationEvent> _desatEvents = new();
    private readonly List<(DateTime Time, double SpO2)> _recentSpo2 = new(128);

    private PlotModel? _trendModel;
    private OxyColor _bgColor = OxyColor.FromRgb(0x09, 0x0E, 0x1C);
    private OxyColor _textColor = OxyColor.FromRgb(0xC8, 0xDC, 0xF0);
    private OxyColor _gridColor = OxyColor.FromRgb(0x18, 0x2C, 0x46);
    private double _baselineSpO2 = 97.0;
    private int _odi3Count;
    private int _odi4Count;

    [ObservableProperty] private double _odi3 = double.NaN;
    [ObservableProperty] private double _odi4 = double.NaN;
    [ObservableProperty] private double _meanSpO2 = double.NaN;
    [ObservableProperty] private double _minSpO2 = double.NaN;
    [ObservableProperty] private double _currentSpO2 = double.NaN;
    [ObservableProperty] private double _t90Percent;
    [ObservableProperty] private string _spo2AlertText = "";
    [ObservableProperty] private bool _hasSpO2Alert;

    public PlotModel SpO2TrendModel => _trendModel ??= BuildTrendModel();

    public void OnResultReceived(ProcessedEEGResult result)
    {
        if (!result.SpO2.HasValue || result.SpO2.Value < 50 || result.SpO2.Value > 100)
            return;

        double spo2 = result.SpO2.Value;
        var now = result.Timestamp;

        _spo2History.Enqueue((now, spo2));
        while (_spo2History.Count > 28800) _spo2History.Dequeue();

        _recentSpo2.Add((now, spo2));
        while (_recentSpo2.Count > 128) _recentSpo2.RemoveAt(0);

        CurrentSpO2 = spo2;

        UpdateBaseline();
        DetectDesaturations();
        RecalculateMetrics();
        UpdateTrendChart();
    }

    public void Reset()
    {
        _spo2History.Clear();
        _desatEvents.Clear();
        _recentSpo2.Clear();
        _baselineSpO2 = 97.0;
        _odi3Count = 0; _odi4Count = 0;
        Odi3 = double.NaN; Odi4 = double.NaN;
        MeanSpO2 = double.NaN; MinSpO2 = double.NaN;
        CurrentSpO2 = double.NaN; T90Percent = 0;
        Spo2AlertText = ""; HasSpO2Alert = false;
        _trendModel = null;
        OnPropertyChanged(nameof(SpO2TrendModel));
    }

    public void ApplyTheme(bool isNightMode)
    {
        _bgColor = isNightMode ? OxyColor.FromRgb(0x0E, 0x09, 0x08) : OxyColor.FromRgb(0x09, 0x0E, 0x1C);
        _textColor = isNightMode ? OxyColor.FromRgb(0xE8, 0xD4, 0xB8) : OxyColor.FromRgb(0xC8, 0xDC, 0xF0);
        _gridColor = isNightMode ? OxyColor.FromRgb(0x2A, 0x1A, 0x12) : OxyColor.FromRgb(0x18, 0x2C, 0x46);
        _trendModel = null;
        OnPropertyChanged(nameof(SpO2TrendModel));
    }

    public List<DesaturationEvent> GetDesaturationEvents() => new(_desatEvents);

    private void UpdateBaseline()
    {
        if (_recentSpo2.Count < 10) return;
        var sorted = _recentSpo2.Select(x => x.SpO2).OrderByDescending(x => x).ToList();
        int topN = Math.Max(5, sorted.Count / 10);
        _baselineSpO2 = sorted.Take(topN).Average();
    }

    private void DetectDesaturations()
    {
        if (_recentSpo2.Count < DesatMinDuration) return;

        double currentMin = _recentSpo2.Min(x => x.SpO2);
        double drop = _baselineSpO2 - currentMin;

        if (drop >= DesatThreshold3 && _recentSpo2.Count >= DesatMinDuration)
        {
            bool isNew = _desatEvents.Count == 0 ||
                         (_recentSpo2[0].Time - _desatEvents[^1].Timestamp).TotalSeconds > 30;

            if (isNew)
            {
                var evt = new DesaturationEvent
                {
                    Timestamp = _recentSpo2[0].Time,
                    SpO2Nadir = currentMin,
                    BaselineSpO2 = _baselineSpO2,
                    DesaturationPercent = drop,
                    IsODI3 = drop >= DesatThreshold3,
                    IsODI4 = drop >= DesatThreshold4,
                    DurationSeconds = _recentSpo2.Count
                };
                _desatEvents.Add(evt);
                if (drop >= DesatThreshold3) _odi3Count++;
                if (drop >= DesatThreshold4) _odi4Count++;

                if (currentMin < 90)
                {
                    Spo2AlertText = $"SpO2 {currentMin:F0}%";
                    HasSpO2Alert = true;
                }
                else
                {
                    HasSpO2Alert = false;
                }
            }
        }
    }

    private void RecalculateMetrics()
    {
        if (_spo2History.Count == 0) return;
        var values = _spo2History.Select(x => x.SpO2).ToList();
        MeanSpO2 = values.Average();
        MinSpO2 = values.Min();
        T90Percent = (double)values.Count(v => v < 90) / values.Count * 100.0;

        double hours = _spo2History.Count / 3600.0;
        Odi3 = hours > 0 ? _odi3Count / hours : 0;
        Odi4 = hours > 0 ? _odi4Count / hours : 0;
    }

    private PlotModel BuildTrendModel()
    {
        var model = new PlotModel { Title = "SpO2 Trend", TitleColor = _textColor };
        model.Background = _bgColor;
        model.PlotAreaBackground = _bgColor;
        model.TextColor = _textColor;

        model.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left,
            Minimum = 50, Maximum = 100,
            Title = "SpO2 %",
            TitleColor = _textColor, TextColor = _textColor,
            AxislineColor = _gridColor, TicklineColor = _gridColor,
            MajorGridlineColor = _gridColor,
            StringFormat = "F0",
        });

        model.Axes.Add(new DateTimeAxis
        {
            Position = AxisPosition.Bottom,
            Title = "Time",
            TitleColor = _textColor, TextColor = _textColor,
            AxislineColor = _gridColor, TicklineColor = _gridColor,
            MajorGridlineColor = _gridColor
        });

        return model;
    }

    private void UpdateTrendChart()
    {
        var model = SpO2TrendModel; // use property to trigger lazy creation
        model.Series.Clear();

        var history = _spo2History.ToArray();
        if (history.Length < 2) return;

        int step = Math.Max(1, history.Length / 300);
        var line = new LineSeries { Color = OxyColor.FromRgb(0x00, 0xE6, 0x76), StrokeThickness = 1 };
        for (int i = 0; i < history.Length; i += step)
        {
            line.Points.Add(new DataPoint(DateTimeAxis.ToDouble(history[i].Time), history[i].SpO2));
        }
        model.Series.Add(line);

        model.InvalidatePlot(true);
    }
}
