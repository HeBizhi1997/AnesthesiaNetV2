namespace EEGMonitor.Services.Simulation;

// ── DTOs returned from Python service ────────────────────────────────────────

public class VitalTrackMeta
{
    public string Name { get; set; } = "";
    public float Srate { get; set; }
    public bool IsWaveform { get; set; }
}

public class VitalFileInfo
{
    public double DurationSeconds { get; set; }
    public List<VitalTrackMeta> Tracks { get; set; } = new();
    public string RecommendedEeg1 { get; set; } = "";
    public string RecommendedEeg2 { get; set; } = "";
    public string RecommendedPulse { get; set; } = "";
    public string RecommendedSpo2 { get; set; } = "";
    public string RecommendedHr { get; set; } = "";
}

public class SimulationConfig
{
    public string FilePath { get; set; } = "";
    public string Eeg1Track { get; set; } = "BIS/EEG1_WAV";
    public string Eeg2Track { get; set; } = "BIS/EEG2_WAV";
    public string PulseTrack { get; set; } = "SNUADC/PLETH";
    public string Spo2Track { get; set; } = "Solar8000/PLETH_SPO2";
    public string HrTrack { get; set; } = "Solar8000/HR";
    public int TargetSampleRate { get; set; } = 256;
    public double PlaybackSpeed { get; set; } = 1.0;
}

// ── Interface ─────────────────────────────────────────────────────────────────

public interface IVitalSimulatorService : IDisposable
{
    bool IsRunning { get; }
    double PlaybackSpeed { get; set; }
    TimeSpan CurrentPosition { get; }
    TimeSpan TotalDuration { get; }

    event Action<TimeSpan>? PositionChanged;
    event Action? SimulationCompleted;
    event Action? SimulationStopped;    // fired by explicit Stop() call

    /// <summary>Fast query: return track list without loading waveform data.</summary>
    Task<VitalFileInfo> QueryFileAsync(string filePath, CancellationToken ct = default);

    /// <summary>Load file + resample into Python service memory (takes a few seconds).</summary>
    Task LoadAsync(SimulationConfig config, CancellationToken ct = default);

    void Start();
    void Pause();
    void Resume();
    void Stop();
    void Seek(TimeSpan position);
}
