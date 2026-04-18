using EEGMonitor.Models;

namespace EEGMonitor.Services.Processing;

public interface IEEGProcessingClient
{
    Task<ProcessedEEGResult> ProcessChunkAsync(EEGDataChunk chunk, CancellationToken ct = default);
    Task<bool> PingAsync(CancellationToken ct = default);
}
