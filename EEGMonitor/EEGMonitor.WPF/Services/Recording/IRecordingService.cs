using EEGMonitor.Models;

namespace EEGMonitor.Services.Recording;

public interface IRecordingService
{
    bool IsRecording { get; }
    RecordingSession? CurrentSession { get; }

    RecordingSession StartSession(string patientId, string surgeryType, string @operator = "");
    Task StopSessionAsync();
    Task RecordRawSampleAsync(EEGSample sample);
    Task RecordResultAsync(ProcessedEEGResult result);
    Task RecordEventAsync(ClinicalEvent clinicalEvent);
    IEnumerable<string> GetRecordedSessions();
}
