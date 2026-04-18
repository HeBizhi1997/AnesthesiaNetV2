using EEGMonitor.Infrastructure.Pipeline;
using EEGMonitor.Models;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using System.Net.Http;
using System.Text;

namespace EEGMonitor.Services.Simulation;

/// <summary>
/// Loads a VitalDB .vital file via the Python simulation API and replays its
/// EEG + vitals into the DataPipeline at the configured speed, simulating a
/// real-time device feed.
///
/// Timer fires every CHUNK_SIZE/sampleRate/speed milliseconds.
/// Each tick: POST /simulate/chunk → inject samples into DataPipeline.
/// </summary>
public sealed class VitalSimulatorService : IVitalSimulatorService
{
    private const int CHUNK_SIZE = 256;

    private readonly HttpClient _http;
    private readonly DataPipeline _pipeline;
    private readonly ILogger<VitalSimulatorService> _logger;

    private string? _sessionId;
    private int _currentSample;
    private int _totalSamples;
    private int _sampleRate = 256;
    private System.Timers.Timer? _timer;
    private bool _fetchInProgress;

    private static readonly JsonSerializerSettings _jsonSettings = new()
    {
        ContractResolver = new DefaultContractResolver { NamingStrategy = new SnakeCaseNamingStrategy() },
        NullValueHandling = NullValueHandling.Ignore,
    };

    public bool IsRunning { get; private set; }
    public double PlaybackSpeed { get; set; } = 1.0;
    public TimeSpan CurrentPosition => TimeSpan.FromSeconds(_currentSample / (double)_sampleRate);
    public TimeSpan TotalDuration => TimeSpan.FromSeconds(_totalSamples / (double)_sampleRate);

    public event Action<TimeSpan>? PositionChanged;
    public event Action? SimulationCompleted;
    public event Action? SimulationStopped;

    public VitalSimulatorService(
        IHttpClientFactory httpFactory,
        DataPipeline pipeline,
        ILogger<VitalSimulatorService> logger)
    {
        _http = httpFactory.CreateClient("simulator");
        _pipeline = pipeline;
        _logger = logger;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public async Task<VitalFileInfo> QueryFileAsync(string filePath, CancellationToken ct = default)
    {
        var body = JsonConvert.SerializeObject(new { file_path = filePath });
        var resp = await _http.PostAsync("simulate/info",
            new StringContent(body, System.Text.Encoding.UTF8, "application/json"), ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadAsStringAsync(ct);
        return JsonConvert.DeserializeObject<VitalFileInfo>(json, _jsonSettings)!;
    }

    public async Task LoadAsync(SimulationConfig cfg, CancellationToken ct = default)
    {
        // Close previous session if any
        await CloseSessionAsync();

        _logger.LogInformation("Loading vital file: {File} at {Rate}Hz", cfg.FilePath, cfg.TargetSampleRate);

        var body = JsonConvert.SerializeObject(new
        {
            file_path = cfg.FilePath,
            eeg1_track = cfg.Eeg1Track,
            eeg2_track = cfg.Eeg2Track,
            pulse_track = cfg.PulseTrack,
            spo2_track = cfg.Spo2Track,
            hr_track = cfg.HrTrack,
            target_sample_rate = cfg.TargetSampleRate,
        });

        var resp = await _http.PostAsync("simulate/load",
            new StringContent(body, System.Text.Encoding.UTF8, "application/json"), ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadAsStringAsync(ct);

        dynamic d = JsonConvert.DeserializeObject<dynamic>(json)!;
        _sessionId = (string)d.session_id;
        _totalSamples = (int)d.total_samples;
        _sampleRate = cfg.TargetSampleRate;
        PlaybackSpeed = cfg.PlaybackSpeed;
        _currentSample = 0;

        _logger.LogInformation("Loaded: {Samples} samples ({Dur:F0}s) session={Id}",
            _totalSamples, _totalSamples / (double)_sampleRate, _sessionId?[..8]);
    }

    public void Start()
    {
        if (_sessionId == null) throw new InvalidOperationException("Call LoadAsync first.");
        if (IsRunning) return;

        IsRunning = true;
        StartTimer();
        _logger.LogInformation("Simulation started at {Speed}×", PlaybackSpeed);
    }

    public void Pause()
    {
        IsRunning = false;
        _timer?.Stop();
        _logger.LogDebug("Simulation paused at {Pos}", CurrentPosition);
    }

    public void Resume()
    {
        if (_sessionId == null || IsRunning) return;
        IsRunning = true;
        StartTimer();
        _logger.LogDebug("Simulation resumed");
    }

    public void Seek(TimeSpan position)
    {
        _currentSample = (int)(position.TotalSeconds * _sampleRate);
        _currentSample = Math.Clamp(_currentSample, 0, _totalSamples - 1);
        PositionChanged?.Invoke(CurrentPosition);
    }

    public void Stop()
    {
        IsRunning = false;
        _timer?.Stop();
        _timer?.Dispose();
        _timer = null;
        _currentSample = 0;
        _ = CloseSessionAsync();
        _logger.LogInformation("Simulation stopped");
        SimulationStopped?.Invoke();
    }

    // ── Timer + chunk fetching ────────────────────────────────────────────────

    private void StartTimer()
    {
        _timer?.Dispose();
        var intervalMs = CHUNK_SIZE * 1000.0 / _sampleRate / PlaybackSpeed;
        _timer = new System.Timers.Timer(intervalMs);
        _timer.Elapsed += OnTimerElapsed;
        _timer.AutoReset = true;
        _timer.Start();
    }

    private async void OnTimerElapsed(object? sender, System.Timers.ElapsedEventArgs e)
    {
        if (!IsRunning || _fetchInProgress) return;
        _fetchInProgress = true;
        try
        {
            await FetchAndInjectAsync();
        }
        finally
        {
            _fetchInProgress = false;
        }
    }

    private async Task FetchAndInjectAsync()
    {
        if (_sessionId == null) return;

        var body = JsonConvert.SerializeObject(new
        {
            session_id = _sessionId,
            start_sample = _currentSample,
            count = CHUNK_SIZE,
        });

        HttpResponseMessage resp;
        try
        {
            resp = await _http.PostAsync("simulate/chunk",
                new StringContent(body, System.Text.Encoding.UTF8, "application/json"));
            resp.EnsureSuccessStatusCode();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Chunk fetch failed – pausing simulation");
            Pause();
            return;
        }

        var json = await resp.Content.ReadAsStringAsync();
        dynamic chunk = JsonConvert.DeserializeObject<dynamic>(json)!;

        var baseTime = DateTime.Now;
        int i = 0;
        foreach (var s in chunk.samples)
        {
            var ts = baseTime.AddSeconds(i / (double)_sampleRate);
            _pipeline.InjectSample(new EEGSample(
                Timestamp: ts,
                Channels: new[] { (double)s.eeg1, (double)s.eeg2 },
                SpO2: s.spo2 == null ? (double?)null : (double)s.spo2,
                HeartRate: s.hr == null ? (double?)null : (double)s.hr,
                PulseWaveValue: (double)s.pulse
            ));
            i++;
        }

        _currentSample = (int)chunk.next_sample;
        PositionChanged?.Invoke(CurrentPosition);

        if ((bool)chunk.is_finished)
        {
            IsRunning = false;
            _timer?.Stop();
            _logger.LogInformation("Simulation file finished");
            SimulationCompleted?.Invoke();
        }
    }

    private async Task CloseSessionAsync()
    {
        if (_sessionId == null) return;
        try
        {
            await _http.DeleteAsync($"simulate/{_sessionId}");
        }
        catch { /* best effort */ }
        _sessionId = null;
    }

    public void Dispose()
    {
        Stop();
        _timer?.Dispose();
        _http.Dispose();
    }
}
