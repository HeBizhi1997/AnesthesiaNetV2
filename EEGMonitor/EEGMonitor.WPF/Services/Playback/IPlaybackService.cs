using EEGMonitor.Models;

namespace EEGMonitor.Services.Playback;

public interface IPlaybackService
{
    bool IsPlaying { get; }
    double PlaybackSpeed { get; set; }
    TimeSpan CurrentPosition { get; }
    TimeSpan TotalDuration { get; }
    RecordingSession? Session { get; }

    event Action<ProcessedEEGResult>? ResultAvailable;
    event Action<ClinicalEvent>? EventReached;
    event Action? PlaybackCompleted;

    Task LoadSessionAsync(string sessionDirectory);
    Task PlayAsync(TimeSpan? startFrom = null);
    void Pause();
    void Seek(TimeSpan position);
    void Stop();
}
