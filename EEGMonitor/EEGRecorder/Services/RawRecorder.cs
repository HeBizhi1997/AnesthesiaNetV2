using System.IO;
using System.Text.Json;
using EEGRecorder.Configuration;

namespace EEGRecorder.Services;

/// <summary>
/// Lossless raw logger. Per session folder {OutputDir}/{subject}_{yyyyMMdd_HHmmss}/ :
///
///   eeg.bin   连续 float32 (little-endian) CH0 µV 流,无时间戳(均匀采样,见 meta)。
///             numpy: np.fromfile("eeg.bin", "&lt;f4")
///   ppg.bin   每帧 18 字节定长记录: [int64 ticks][int32 ir][int32 red][uint8 spo2][uint8 hr]
///             numpy dtype: [("ticks","&lt;i8"),("ir","&lt;i4"),("red","&lt;i4"),("spo2","u1"),("hr","u1")]
///   meta.json 采样率(标称+实测)、增益、端口、起止时间、计数等。
///
/// 录制即原始 —— 不做任何滤波/重采样。
/// </summary>
public sealed class RawRecorder : IDisposable
{
    private readonly object _eegLock = new();
    private readonly object _ppgLock = new();
    private BinaryWriter? _eeg;
    private BinaryWriter? _ppg;
    private RecorderConfig? _cfg;

    private DateTime _startedLocal;
    private long _firstEegTicks, _lastEegTicks;

    public bool IsRecording { get; private set; }
    public string? SessionDir { get; private set; }
    public long EegSamples { get; private set; }
    public long PpgFrames { get; private set; }

    public string Start(RecorderConfig cfg)
    {
        if (IsRecording) Stop();
        _cfg = cfg;
        EegSamples = PpgFrames = 0;
        _firstEegTicks = _lastEegTicks = 0;
        _startedLocal = DateTime.Now;

        var root = string.IsNullOrWhiteSpace(cfg.Recording.OutputDirectory)
            ? Path.Combine(AppContext.BaseDirectory, "recordings_raw")
            : cfg.Recording.OutputDirectory;
        var folder = $"{Safe(cfg.Session.Subject)}_{_startedLocal:yyyyMMdd_HHmmss}";
        var dir = Path.Combine(root, folder);
        Directory.CreateDirectory(dir);
        SessionDir = dir;

        _eeg = new BinaryWriter(File.Open(Path.Combine(dir, "eeg.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));
        _ppg = new BinaryWriter(File.Open(Path.Combine(dir, "ppg.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));

        IsRecording = true;
        WriteMeta(final: false);
        return dir;
    }

    public void WriteEeg(float[] ch0)
    {
        var w = _eeg;
        if (!IsRecording || w == null || ch0.Length == 0) return;
        lock (_eegLock)
        {
            long now = DateTime.Now.Ticks;
            if (_firstEegTicks == 0) _firstEegTicks = now;
            _lastEegTicks = now;
            foreach (var v in ch0) w.Write(v);
            EegSamples += ch0.Length;
        }
    }

    public void WritePpg(IReadOnlyList<PpgFrame> frames)
    {
        var w = _ppg;
        if (!IsRecording || w == null || frames.Count == 0) return;
        lock (_ppgLock)
        {
            foreach (var f in frames)
            {
                w.Write(f.Ticks);
                w.Write(f.Ir);
                w.Write(f.Red);
                w.Write(f.Spo2);
                w.Write(f.Hr);
            }
            PpgFrames += frames.Count;
        }
    }

    public void Stop()
    {
        if (!IsRecording) return;
        IsRecording = false;
        lock (_eegLock) { _eeg?.Flush(); _eeg?.Dispose(); _eeg = null; }
        lock (_ppgLock) { _ppg?.Flush(); _ppg?.Dispose(); _ppg = null; }
        WriteMeta(final: true);
    }

    private void WriteMeta(bool final)
    {
        if (SessionDir == null || _cfg == null) return;
        var stopped = final ? DateTime.Now : (DateTime?)null;

        double eegMeasured = 0;
        if (_firstEegTicks > 0 && _lastEegTicks > _firstEegTicks && EegSamples > 1)
        {
            double secs = (_lastEegTicks - _firstEegTicks) / (double)TimeSpan.TicksPerSecond;
            if (secs > 0) eegMeasured = (EegSamples - 1) / secs;
        }
        double ppgMeasured = 0;
        if (final && PpgFrames > 1)
        {
            double secs = (DateTime.Now - _startedLocal).TotalSeconds;
            if (secs > 0) ppgMeasured = PpgFrames / secs;
        }

        var meta = new
        {
            format_version = 1,
            subject = _cfg.Session.Subject,
            note = _cfg.Session.Note,
            started = _startedLocal.ToString("o"),
            stopped = stopped?.ToString("o"),
            eeg = new
            {
                file = "eeg.bin",
                dtype = "<f4",
                unit = "uV",
                channels = 1,
                channel_note = "ADS1299 CH0 (board interleaves 8, only CH0 is the electrode)",
                port = _cfg.Serial.EegPort,
                baud = _cfg.Serial.EegBaud,
                nominal_rate_hz = _cfg.Serial.EegSampleRate,
                measured_rate_hz = Math.Round(eegMeasured, 3),
                gain = _cfg.Serial.EegGain,
                differential = _cfg.Serial.EegDifferential,
                samples = EegSamples,
                first_sample = _firstEegTicks > 0 ? new DateTime(_firstEegTicks).ToString("o") : null,
            },
            ppg = new
            {
                file = "ppg.bin",
                enabled = _cfg.Serial.PpgEnabled,
                record = "[<i8 ticks][<i4 ir][<i4 red][u1 spo2][u1 hr]",
                record_bytes = 18,
                port = _cfg.Serial.PpgPort,
                baud = _cfg.Serial.PpgBaud,
                measured_rate_hz = Math.Round(ppgMeasured, 3),
                frames = PpgFrames,
            },
        };
        try
        {
            File.WriteAllText(Path.Combine(SessionDir, "meta.json"),
                JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch { /* best effort */ }
    }

    private static string Safe(string? s, int max = 40)
    {
        if (string.IsNullOrWhiteSpace(s)) return "subj";
        var invalid = Path.GetInvalidFileNameChars();
        var cleaned = new string(s.Trim().Select(c => invalid.Contains(c) || c == ' ' ? '_' : c).ToArray());
        if (cleaned.Length > max) cleaned = cleaned[..max];
        return cleaned.Length == 0 ? "subj" : cleaned;
    }

    public void Dispose() => Stop();
}
