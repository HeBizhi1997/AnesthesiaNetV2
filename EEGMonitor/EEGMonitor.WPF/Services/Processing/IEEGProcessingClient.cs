using EEGMonitor.Models;

namespace EEGMonitor.Services.Processing;

public interface IEEGProcessingClient
{
    Task<ProcessedEEGResult> ProcessChunkAsync(EEGDataChunk chunk, CancellationToken ct = default);
    Task<bool> PingAsync(CancellationToken ct = default);

    /// <summary>
    /// Calls POST /reset on the Python service.
    /// Clears the rolling BIS buffer, GRU hidden state, entropy history, and fNox state.
    /// Must be called at the start of every new monitoring session.
    /// </summary>
    Task ResetSessionAsync(CancellationToken ct = default);
}
