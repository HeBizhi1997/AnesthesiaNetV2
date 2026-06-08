using Ads1299Monitor.Configuration;
using Ads1299Monitor.Models;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System.IO;

namespace Ads1299Monitor.Services;

/// <summary>
/// Persists a session into TWO files under {OutputDirectory}/{yyyyMMdd_HHmmss}/ :
///
///   raw_signal.bin     原始信号（脑电 + 生命体征），按真实采集频率/精度无损存储。
///                      Tagged little-endian binary records:
///                        [int64 ticks][byte tag]
///                          tag 1 = EEG  : [float32 ch0_µV]
///                          tag 2 = VITAL: [float32 hr][float32 spo2][float32 pulse]
///                      (sidecar raw_signal.meta.json documents the layout + sample rate)
///
///   inference.jsonl    推理结果，按秒一行（每个 1 秒 epoch 一条 JSON）。
///
/// The output directory is read from appsettings.json (Recording:OutputDirectory).
/// </summary>
public sealed class RecordingService : IDisposable
{
    private const byte TAG_EEG = 1;
    private const byte TAG_VITAL = 2;
    private const byte TAG_PPG = 3;

    private readonly ILogger<RecordingService> _logger;
    private readonly RecordingConfig _cfg;
    private readonly PatientConfig _patient;

    private BinaryWriter? _rawWriter;
    private StreamWriter? _inferenceWriter;
    private StreamWriter? _eventWriter;
    private readonly object _rawLock = new();
    private readonly object _infLock = new();
    private readonly object _evtLock = new();

    public bool IsRecording { get; private set; }
    public string? SessionDirectory { get; private set; }

    public RecordingService(ILogger<RecordingService> logger, AppConfig config)
    {
        _logger = logger;
        _cfg = config.Recording;
        _patient = config.Patient;
    }

    /// <summary>Make a string safe + bounded for use as a path segment.</summary>
    private static string SafeSeg(string? s, int maxLen = 40)
    {
        if (string.IsNullOrWhiteSpace(s)) return "未填";
        var invalid = Path.GetInvalidFileNameChars();
        var cleaned = new string(s.Trim().Select(c => invalid.Contains(c) || c == ' ' ? '_' : c).ToArray());
        if (cleaned.Length > maxLen) cleaned = cleaned.Substring(0, maxLen);
        return cleaned.Length == 0 ? "未填" : cleaned;
    }

    public void Start(int sampleRate)
    {
        if (IsRecording) Stop();
        if (!_cfg.Enabled) { _logger.LogInformation("Recording disabled in config"); return; }

        var root = string.IsNullOrWhiteSpace(_cfg.OutputDirectory)
            ? Path.Combine(AppContext.BaseDirectory, "Recordings")
            : _cfg.OutputDirectory;
        // Folder name: 患者名称_住院号_手术名称_开始时间
        var startTime = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var folder = $"{SafeSeg(_patient.Name)}_{SafeSeg(_patient.AdmissionNo)}_{SafeSeg(_patient.SurgeryName)}_{startTime}";
        var dir = Path.Combine(root, folder);
        Directory.CreateDirectory(dir);
        SessionDirectory = dir;

        _rawWriter = new BinaryWriter(File.Open(Path.Combine(dir, "raw_signal.bin"), FileMode.Create));
        _inferenceWriter = new StreamWriter(Path.Combine(dir, "inference.jsonl"), append: false);
        _eventWriter = new StreamWriter(Path.Combine(dir, "events.jsonl"), append: false);

        File.WriteAllText(Path.Combine(dir, "raw_signal.meta.json"), JsonConvert.SerializeObject(new
        {
            format = "tagged-le-binary",
            record = "[int64 ticks][byte tag]; tag1 EEG=[float32 uv]; tag2 VITAL=[float32 pr][float32 spo2][float32 pi]; tag3 PPG=[int32 ir][int32 red]",
            eeg_sample_rate_hz = sampleRate,
            started = DateTime.Now.ToString("o"),
        }, Formatting.Indented));

        IsRecording = true;
        _logger.LogInformation("Recording started → {Dir}", dir);
    }

    public void Stop()
    {
        if (!IsRecording) return;
        IsRecording = false;
        lock (_rawLock) { _rawWriter?.Flush(); _rawWriter?.Dispose(); _rawWriter = null; }
        lock (_infLock) { _inferenceWriter?.Flush(); _inferenceWriter?.Dispose(); _inferenceWriter = null; }
        lock (_evtLock) { _eventWriter?.Flush(); _eventWriter?.Dispose(); _eventWriter = null; }
        _logger.LogInformation("Recording stopped → {Dir}", SessionDirectory);
    }

    /// <summary>Raw EEG sample at the native acquisition rate (lossless).</summary>
    public void RecordRawEeg(EEGSample sample)
    {
        var w = _rawWriter;
        if (!IsRecording || w == null) return;
        lock (_rawLock)
        {
            w.Write(sample.Timestamp.Ticks);
            w.Write(TAG_EEG);
            w.Write((float)(sample.Channels.Length > 0 ? sample.Channels[0] : 0.0));
        }
    }

    /// <summary>Raw vital-sign sample (whenever the pulse sensor reports), lossless.</summary>
    public void RecordRawVital(DateTime ts, double hr, double spo2, double pulse)
    {
        var w = _rawWriter;
        if (!IsRecording || w == null) return;
        lock (_rawLock)
        {
            w.Write(ts.Ticks);
            w.Write(TAG_VITAL);
            w.Write((float)hr);
            w.Write((float)spo2);
            w.Write((float)pulse);
        }
    }

    /// <summary>Raw PPG sample (IR + RED) at the native acquisition rate, lossless — lets us
    /// re-derive / diagnose the pulse waveform offline (PR/PRV/PI).</summary>
    public void RecordRawPpg(DateTime ts, int ir, int red)
    {
        var w = _rawWriter;
        if (!IsRecording || w == null) return;
        lock (_rawLock)
        {
            w.Write(ts.Ticks);
            w.Write(TAG_PPG);
            w.Write(ir);
            w.Write(red);
        }
    }

    /// <summary>Per-second inference result.</summary>
    public void RecordInference(ProcessedEEGResult result)
    {
        var w = _inferenceWriter;
        if (!IsRecording || w == null) return;
        // Keep the per-second file compact: drop the bulky waveform arrays.
        var compact = new
        {
            t = result.Timestamp.ToString("o"),
            qcon = double.IsNaN(result.BIS) ? (double?)null : Math.Round(result.BIS, 1),
            qnox = result.FNox,
            sqi = Math.Round(result.SQI, 1),
            se = result.StateEntropy,
            re = result.ResponseEntropy,
            hr = result.HeartRate,
            hrv = result.HRV_RMSSD,
            spo2 = result.SpO2,
            bands = new { result.DeltaPower, result.ThetaPower, result.AlphaPower, result.BetaPower, result.GammaPower },
            amp_uv = Math.Round(result.EegAmplitudeUv, 1),
            dom_hz = Math.Round(result.EegDominantHz, 1),
            saturated = result.HwIsSaturated,
            valid = result.SignalValid,                 // 是否接好电极(false=悬空噪声,指标无效)
            electrode = result.ElectrodeStatus,
        };
        var line = JsonConvert.SerializeObject(compact);
        lock (_infLock) { w.WriteLine(line); }
    }

    /// <summary>Clinical event annotation (用药 / 电刀 / 喉镜 / 缝合 / 翻身 …).</summary>
    public void RecordEvent(EventItem ev)
    {
        var w = _eventWriter;
        if (!IsRecording || w == null) return;
        var line = JsonConvert.SerializeObject(new
        {
            t = DateTime.Now.ToString("o"),
            ev.Time, ev.Category, ev.Name, ev.Dose, ev.Operator, ev.Note,
        });
        lock (_evtLock) { w.WriteLine(line); }
    }

    public void Dispose() => Stop();
}
