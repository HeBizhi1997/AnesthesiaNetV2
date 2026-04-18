using EEGMonitor.Models;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System.IO;

namespace EEGMonitor.Services.Recording;

/// <summary>
/// Persists raw EEG samples (binary float32), processed results (JSONL), and events (JSON).
/// Session directory layout:
///   Sessions/{SessionId}/
///     session.json          – metadata
///     raw_eeg.bin           – packed float32 samples (timestamp:int64, ch0..chN:float32)
///     vitals.bin            – packed float32 (timestamp:int64, spo2, hr, pulse)
///     processed.jsonl       – one JSON per processed epoch
///     events.json           – list of ClinicalEvent
/// </summary>
public sealed class RecordingService : IRecordingService
{
    private readonly ILogger<RecordingService> _logger;
    private readonly string _baseDir;

    private BinaryWriter? _rawWriter;
    private BinaryWriter? _vitalsWriter;
    private StreamWriter? _processedWriter;

    public bool IsRecording { get; private set; }
    public RecordingSession? CurrentSession { get; private set; }

    public RecordingService(ILogger<RecordingService> logger)
    {
        _logger = logger;
        _baseDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "EEGMonitor", "Sessions");
        Directory.CreateDirectory(_baseDir);
    }

    public RecordingSession StartSession(string patientId, string surgeryType, string @operator = "")
    {
        if (IsRecording) StopSessionAsync().GetAwaiter().GetResult();

        var session = new RecordingSession
        {
            PatientId = patientId,
            SurgeryType = surgeryType,
            Operator = @operator,
            StartTime = DateTime.Now,
        };

        var dir = Path.Combine(_baseDir, session.SessionId.ToString());
        Directory.CreateDirectory(dir);
        session.RecordingDirectory = dir;
        CurrentSession = session;

        _rawWriter = new BinaryWriter(File.Open(Path.Combine(dir, "raw_eeg.bin"), FileMode.Create));
        _vitalsWriter = new BinaryWriter(File.Open(Path.Combine(dir, "vitals.bin"), FileMode.Create));
        _processedWriter = new StreamWriter(Path.Combine(dir, "processed.jsonl"), append: false);

        SaveSessionMetadata();
        IsRecording = true;
        _logger.LogInformation("Recording session started: {SessionId} Patient={PatientId}", session.SessionId, patientId);
        return session;
    }

    public async Task StopSessionAsync()
    {
        if (!IsRecording || CurrentSession == null) return;
        IsRecording = false;
        CurrentSession.EndTime = DateTime.Now;

        _rawWriter?.Flush(); _rawWriter?.Close(); _rawWriter = null;
        _vitalsWriter?.Flush(); _vitalsWriter?.Close(); _vitalsWriter = null;
        await (_processedWriter?.FlushAsync() ?? Task.CompletedTask);
        _processedWriter?.Close(); _processedWriter = null;

        SaveSessionMetadata();
        _logger.LogInformation("Recording session stopped: {SessionId} Duration={Duration}",
            CurrentSession.SessionId, CurrentSession.Duration);
    }

    public Task RecordRawSampleAsync(EEGSample sample)
    {
        if (!IsRecording || _rawWriter == null) return Task.CompletedTask;
        lock (_rawWriter)
        {
            _rawWriter.Write(sample.Timestamp.ToBinary());
            foreach (var ch in sample.Channels) _rawWriter.Write((float)ch);
        }
        if (sample.SpO2.HasValue && _vitalsWriter != null)
        {
            lock (_vitalsWriter)
            {
                _vitalsWriter.Write(sample.Timestamp.ToBinary());
                _vitalsWriter.Write((float)(sample.SpO2 ?? 0));
                _vitalsWriter.Write((float)(sample.HeartRate ?? 0));
                _vitalsWriter.Write((float)(sample.PulseWaveValue ?? 0));
            }
        }
        return Task.CompletedTask;
    }

    public async Task RecordResultAsync(ProcessedEEGResult result)
    {
        if (!IsRecording || _processedWriter == null) return;
        var line = JsonConvert.SerializeObject(result);
        await _processedWriter.WriteLineAsync(line);
    }

    public async Task RecordEventAsync(ClinicalEvent clinicalEvent)
    {
        if (CurrentSession == null) return;
        CurrentSession.Events.Add(clinicalEvent);
        SaveSessionMetadata();
        await Task.CompletedTask;
    }

    private void SaveSessionMetadata()
    {
        if (CurrentSession == null) return;
        var path = Path.Combine(CurrentSession.RecordingDirectory, "session.json");
        File.WriteAllText(path, JsonConvert.SerializeObject(CurrentSession, Formatting.Indented));
    }

    public IEnumerable<string> GetRecordedSessions() =>
        Directory.EnumerateDirectories(_baseDir)
                 .Select(d => Path.GetFileName(d))
                 .OrderByDescending(s => s);
}
