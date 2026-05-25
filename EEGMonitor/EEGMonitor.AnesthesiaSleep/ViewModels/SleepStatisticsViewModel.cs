using CommunityToolkit.Mvvm.ComponentModel;
using EEGMonitor.AnesthesiaSleep.Models;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class SleepStatisticsViewModel : ObservableObject
{
    private readonly List<SleepEpoch> _allEpochs = new();
    private DateTime? _recordingStart;
    private DateTime? _lastEpochEnd;

    [ObservableProperty] private double _tstMinutes;
    [ObservableProperty] private double _sePercent;
    [ObservableProperty] private double _solMinutes;
    [ObservableProperty] private double _wasoMinutes;
    [ObservableProperty] private double _n1Percent;
    [ObservableProperty] private double _n2Percent;
    [ObservableProperty] private double _n3Percent;
    [ObservableProperty] private double _remPercent;
    [ObservableProperty] private double _remLatencyMinutes;
    [ObservableProperty] private int _arousalCount;
    [ObservableProperty] private double _arousalIndex;
    [ObservableProperty] private double _recordingDurationMinutes;
    [ObservableProperty] private int _totalEpochs;
    [ObservableProperty] private int _artifactEpochs;

    public SleepStatistics GetSnapshot()
    {
        return new SleepStatistics
        {
            TSTMinutes = TstMinutes,
            SEPercent = SePercent,
            SOLMinutes = SolMinutes,
            WASOMinutes = WasoMinutes,
            N1Minutes = N1Percent / 100.0 * TstMinutes,
            N1Percent = N1Percent,
            N2Minutes = N2Percent / 100.0 * TstMinutes,
            N2Percent = N2Percent,
            N3Minutes = N3Percent / 100.0 * TstMinutes,
            N3Percent = N3Percent,
            REMMinutes = RemPercent / 100.0 * TstMinutes,
            REMPercent = RemPercent,
            REMLatencyMinutes = RemLatencyMinutes,
            ArousalCount = ArousalCount,
            ArousalIndex = ArousalIndex,
            RecordingDurationMinutes = RecordingDurationMinutes,
            TotalEpochs = TotalEpochs,
            ArtifactEpochs = ArtifactEpochs,
            CalculatedAt = DateTime.Now
        };
    }

    public void OnEpochCompleted(SleepEpoch epoch)
    {
        if (_recordingStart == null)
            _recordingStart = epoch.StartTime;

        _allEpochs.Add(epoch);
        _lastEpochEnd = epoch.EndTime;
        Recalculate();
    }

    public void Reset()
    {
        _allEpochs.Clear();
        _recordingStart = null;
        _lastEpochEnd = null;
        TstMinutes = 0; SePercent = 0; SolMinutes = 0; WasoMinutes = 0;
        N1Percent = 0; N2Percent = 0; N3Percent = 0; RemPercent = 0;
        RemLatencyMinutes = 0; ArousalCount = 0; ArousalIndex = 0;
        RecordingDurationMinutes = 0; TotalEpochs = 0; ArtifactEpochs = 0;
    }

    private void Recalculate()
    {
        var valid = _allEpochs.Where(e => !e.IsArtifact).ToList();
        var sleepEpochs = valid.Where(e => e.Stage != SleepStage.Wake).ToList();

        TotalEpochs = _allEpochs.Count;
        ArtifactEpochs = _allEpochs.Count(e => e.IsArtifact);

        if (_recordingStart != null && _lastEpochEnd != null)
            RecordingDurationMinutes = (_lastEpochEnd.Value - _recordingStart.Value).TotalMinutes;

        int totalValid = valid.Count;
        if (totalValid == 0) return;

        TstMinutes = sleepEpochs.Count * 30.0 / 60.0;
        SePercent = totalValid > 0 ? (double)sleepEpochs.Count / totalValid * 100.0 : 0;

        // SOL: time to first sleep epoch (any stage except Wake)
        var firstSleep = valid.FirstOrDefault(e => e.Stage != SleepStage.Wake);
        SolMinutes = firstSleep != null ? (firstSleep.StartTime - _recordingStart!.Value).TotalMinutes : 0;

        // WASO: wake epochs after sleep onset
        if (firstSleep != null)
        {
            var wasoEpochs = valid.SkipWhile(e => e != firstSleep).Count(e => e.Stage == SleepStage.Wake);
            WasoMinutes = wasoEpochs * 30.0 / 60.0;
        }

        // Stage percentages
        if (sleepEpochs.Count > 0)
        {
            N1Percent = (double)sleepEpochs.Count(e => e.Stage == SleepStage.N1) / sleepEpochs.Count * 100.0;
            N2Percent = (double)sleepEpochs.Count(e => e.Stage == SleepStage.N2) / sleepEpochs.Count * 100.0;
            N3Percent = (double)sleepEpochs.Count(e => e.Stage == SleepStage.N3) / sleepEpochs.Count * 100.0;
            RemPercent = (double)sleepEpochs.Count(e => e.Stage == SleepStage.REM) / sleepEpochs.Count * 100.0;
        }

        // REM latency: time to first REM after sleep onset
        if (firstSleep != null)
        {
            var firstREM = sleepEpochs.FirstOrDefault(e => e.Stage == SleepStage.REM);
            RemLatencyMinutes = firstREM != null ? (firstREM.StartTime - firstSleep.StartTime).TotalMinutes : 0;
        }

        // Arousal index: count Wake epochs within sleep as arousals, per hour
        ArousalCount = valid.SkipWhile(e => e.Stage == SleepStage.Wake).Count(e => e.Stage == SleepStage.Wake);
        ArousalIndex = TstMinutes > 0 ? ArousalCount / (TstMinutes / 60.0) : 0;
    }
}
