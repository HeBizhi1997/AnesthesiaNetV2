using EEGMonitor.Models;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System.IO;

namespace EEGMonitor.Services.Playback;

public sealed class PlaybackService : IPlaybackService
{
    private readonly ILogger<PlaybackService> _logger;
    private List<ProcessedEEGResult> _results = new();
    private List<ClinicalEvent> _events = new();
    private CancellationTokenSource? _cts;
    private Task? _playbackTask;

    public bool IsPlaying { get; private set; }
    public double PlaybackSpeed { get; set; } = 1.0;
    public TimeSpan CurrentPosition { get; private set; }
    public TimeSpan TotalDuration { get; private set; }
    public RecordingSession? Session { get; private set; }

    public event Action<ProcessedEEGResult>? ResultAvailable;
    public event Action<ClinicalEvent>? EventReached;
    public event Action? PlaybackCompleted;

    public PlaybackService(ILogger<PlaybackService> logger)
    {
        _logger = logger;
    }

    public async Task LoadSessionAsync(string sessionDirectory)
    {
        Stop();
        var sessionJson = await File.ReadAllTextAsync(Path.Combine(sessionDirectory, "session.json"));
        Session = JsonConvert.DeserializeObject<RecordingSession>(sessionJson)!;
        Session.IsPlayback = true;

        var processedPath = Path.Combine(sessionDirectory, "processed.jsonl");
        _results = (await File.ReadAllLinesAsync(processedPath))
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .Select(l => JsonConvert.DeserializeObject<ProcessedEEGResult>(l)!)
            .OrderBy(r => r.Timestamp)
            .ToList();

        _events = Session.Events.OrderBy(e => e.Timestamp).ToList();

        TotalDuration = _results.Count > 0
            ? _results[^1].Timestamp - _results[0].Timestamp
            : TimeSpan.Zero;

        _logger.LogInformation("Session loaded: {Count} epochs, Duration={Duration}", _results.Count, TotalDuration);
    }

    public async Task PlayAsync(TimeSpan? startFrom = null)
    {
        if (_results.Count == 0) return;
        Stop();

        _cts = new CancellationTokenSource();
        var ct = _cts.Token;
        var startOffset = startFrom ?? TimeSpan.Zero;

        _playbackTask = Task.Run(async () =>
        {
            IsPlaying = true;
            var origin = _results[0].Timestamp;
            var startResultIdx = _results.FindIndex(r => r.Timestamp - origin >= startOffset);
            if (startResultIdx < 0) startResultIdx = 0;

            var eventIdx = _events.FindIndex(e => e.Timestamp - origin >= startOffset);
            if (eventIdx < 0) eventIdx = 0;

            var wallStart = DateTime.Now - startOffset / PlaybackSpeed;

            for (int i = startResultIdx; i < _results.Count && !ct.IsCancellationRequested; i++)
            {
                var result = _results[i];
                var sessionOffset = result.Timestamp - origin;
                CurrentPosition = sessionOffset;

                // Emit events that fall before this result
                while (eventIdx < _events.Count)
                {
                    var evOffset = _events[eventIdx].Timestamp - origin;
                    if (evOffset <= sessionOffset)
                    {
                        EventReached?.Invoke(_events[eventIdx]);
                        eventIdx++;
                    }
                    else break;
                }

                ResultAvailable?.Invoke(result);

                // Wait until real-time aligned moment
                if (i < _results.Count - 1)
                {
                    var nextOffset = _results[i + 1].Timestamp - origin;
                    var delay = (nextOffset - sessionOffset) / PlaybackSpeed;
                    if (delay > TimeSpan.Zero)
                        await Task.Delay(delay, ct).ContinueWith(_ => { }); // swallow cancel
                }
            }

            IsPlaying = false;
            if (!ct.IsCancellationRequested)
            {
                PlaybackCompleted?.Invoke();
                _logger.LogInformation("Playback completed");
            }
        }, ct);

        await Task.CompletedTask;
    }

    public void Pause()
    {
        if (!IsPlaying) return;
        _cts?.Cancel();
        IsPlaying = false;
    }

    public void Seek(TimeSpan position)
    {
        var wasPlaying = IsPlaying;
        Pause();
        CurrentPosition = position;
        if (wasPlaying) PlayAsync(position).GetAwaiter().GetResult();
    }

    public void Stop()
    {
        _cts?.Cancel();
        IsPlaying = false;
        CurrentPosition = TimeSpan.Zero;
    }
}
