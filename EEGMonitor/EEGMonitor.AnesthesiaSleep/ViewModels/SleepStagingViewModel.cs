using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using EEGMonitor.AnesthesiaSleep.Infrastructure.SleepStaging;
using EEGMonitor.AnesthesiaSleep.Models;
using EEGMonitor.Models;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class SleepStagingViewModel : ObservableObject
{
    private readonly SleepStageRuleEngine _engine;
    private readonly List<ProcessedEEGResult> _secondBuffer = new(32);
    private readonly List<SleepEpoch> _epochHistory = new(960);
    private readonly List<SleepStage> _stageWindow = new(10);

    private PlotModel? _hypnogramModel;
    private OxyColor _bgColor = OxyColor.FromRgb(0x09, 0x0E, 0x1C);
    private OxyColor _textColor = OxyColor.FromRgb(0xC8, 0xDC, 0xF0);
    private OxyColor _gridColor = OxyColor.FromRgb(0x18, 0x2C, 0x46);

    [ObservableProperty] private SleepStage _currentStage = SleepStage.Unknown;
    [ObservableProperty] private string _currentStageLabel = "---";
    [ObservableProperty] private double _currentConfidence;
    [ObservableProperty] private int _epochCount;
    [ObservableProperty] private ObservableCollection<SleepEpoch> _recentEpochs = new();

    public event Action<SleepEpoch>? EpochCompleted;

    public PlotModel HypnogramModel => _hypnogramModel ??= BuildHypnogramModel();

    public SleepStagingViewModel(SleepStageRuleEngine engine)
    {
        _engine = engine;
    }

    public void OnResultReceived(ProcessedEEGResult result)
    {
        _secondBuffer.Add(result);
        if (_secondBuffer.Count >= 30)
        {
            var epoch = _engine.ClassifyEpoch(_secondBuffer);
            _secondBuffer.Clear();

            _stageWindow.Add(epoch.Stage);
            if (_stageWindow.Count > 10) _stageWindow.RemoveAt(0);

            var smoothed = _engine.SmoothStage(epoch.Stage, _stageWindow);
            if (smoothed != epoch.Stage)
            {
                epoch.Stage = smoothed;
            }

            _epochHistory.Add(epoch);
            if (_epochHistory.Count > 960) _epochHistory.RemoveAt(0);

            CurrentStage = epoch.Stage;
            CurrentStageLabel = epoch.Stage switch
            {
                SleepStage.Wake => "清醒",
                SleepStage.N1 => "N1",
                SleepStage.N2 => "N2",
                SleepStage.N3 => "N3",
                SleepStage.REM => "REM",
                SleepStage.Movement => "体动",
                _ => "---"
            };
            CurrentConfidence = epoch.Confidence;
            EpochCount = _epochHistory.Count;
            RecentEpochs.Add(epoch);
            while (RecentEpochs.Count > 120) RecentEpochs.RemoveAt(0);

            UpdateHypnogram();
            EpochCompleted?.Invoke(epoch);
        }
    }

    public void Reset()
    {
        _secondBuffer.Clear();
        _epochHistory.Clear();
        _stageWindow.Clear();
        RecentEpochs.Clear();
        EpochCount = 0;
        CurrentStage = SleepStage.Unknown;
        CurrentStageLabel = "---";
        CurrentConfidence = 0;
        _hypnogramModel = null;
        OnPropertyChanged(nameof(HypnogramModel));
    }

    public void ApplyTheme(bool isNightMode)
    {
        _bgColor = isNightMode ? OxyColor.FromRgb(0x0E, 0x09, 0x08) : OxyColor.FromRgb(0x09, 0x0E, 0x1C);
        _textColor = isNightMode ? OxyColor.FromRgb(0xE8, 0xD4, 0xB8) : OxyColor.FromRgb(0xC8, 0xDC, 0xF0);
        _gridColor = isNightMode ? OxyColor.FromRgb(0x2A, 0x1A, 0x12) : OxyColor.FromRgb(0x18, 0x2C, 0x46);
        _hypnogramModel = null;
        OnPropertyChanged(nameof(HypnogramModel));
    }

    private PlotModel BuildHypnogramModel()
    {
        var model = new PlotModel { Title = "Hypnogram", TitleColor = _textColor };
        model.Background = _bgColor;
        model.PlotAreaBackground = _bgColor;
        model.TextColor = _textColor;

        var yAxis = new CategoryAxis
        {
            Position = AxisPosition.Left,
            Title = "Stage",
            TitleColor = _textColor,
            TextColor = _textColor,
            AxislineColor = _gridColor,
            TicklineColor = _gridColor
        };
        yAxis.Labels.Add("W");
        yAxis.Labels.Add("REM");
        yAxis.Labels.Add("N1");
        yAxis.Labels.Add("N2");
        yAxis.Labels.Add("N3");
        yAxis.Labels.Add("MVT");

        model.Axes.Add(yAxis);

        var xAxis = new LinearAxis
        {
            Position = AxisPosition.Bottom,
            Title = "Time (epochs)",
            TitleColor = _textColor,
            TextColor = _textColor,
            AxislineColor = _gridColor,
            TicklineColor = _gridColor,
            MajorGridlineColor = _gridColor,
            StringFormat = "F0",
        };
        model.Axes.Add(xAxis);

        return model;
    }

    private void UpdateHypnogram()
    {
        var model = HypnogramModel; // use property to trigger lazy creation
        model.Series.Clear();

        if (_epochHistory.Count < 2) return;

        var colors = new Dictionary<SleepStage, OxyColor>
        {
            [SleepStage.Wake] = OxyColor.FromRgb(0x6B, 0x70, 0x80),
            [SleepStage.N1] = OxyColor.FromRgb(0xF0, 0xA0, 0x20),
            [SleepStage.N2] = OxyColor.FromRgb(0x00, 0xC8, 0xFF),
            [SleepStage.N3] = OxyColor.FromRgb(0x5B, 0x8E, 0xFF),
            [SleepStage.REM] = OxyColor.FromRgb(0xE0, 0x40, 0xC0),
            [SleepStage.Movement] = OxyColor.FromRgb(0xFF, 0x45, 0x60),
            [SleepStage.Unknown] = OxyColor.FromRgb(0x30, 0x30, 0x30)
        };

        int start = Math.Max(0, _epochHistory.Count - 120);
        for (int i = start; i < _epochHistory.Count; i++)
        {
            var epoch = _epochHistory[i];
            double y = epoch.Stage switch
            {
                SleepStage.Wake => 5,
                SleepStage.REM => 4,
                SleepStage.N1 => 3,
                SleepStage.N2 => 2,
                SleepStage.N3 => 1,
                SleepStage.Movement => 0,
                _ => 0
            };

            var s = new LineSeries
            {
                Color = colors.GetValueOrDefault(epoch.Stage, OxyColors.Gray),
                StrokeThickness = 2
            };
            s.Points.Add(new DataPoint(i - start, y));
            model.Series.Add(s);
        }

        model.InvalidatePlot(true);
    }
}
