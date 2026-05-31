using Ads1299Monitor.Models;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System.Net.Http;

namespace Ads1299Monitor.Services;

/// <summary>HTTP client to the Python FastAPI inference service (/process, /reset, /health).</summary>
public sealed class EEGProcessingClient
{
    private readonly HttpClient _http;
    private readonly ILogger<EEGProcessingClient> _logger;

    public EEGProcessingClient(HttpClient http, ILogger<EEGProcessingClient> logger)
    {
        _http = http;
        _logger = logger;
    }

    public async Task<bool> PingAsync(CancellationToken ct = default)
    {
        try { return (await _http.GetAsync("health", ct)).IsSuccessStatusCode; }
        catch { return false; }
    }

    public async Task ResetSessionAsync(CancellationToken ct = default)
    {
        try { await _http.PostAsync("reset", null, ct); }
        catch (Exception ex) { _logger.LogWarning(ex, "Failed to reset session"); }
    }

    public async Task<ProcessedEEGResult> ProcessChunkAsync(EEGDataChunk chunk, CancellationToken ct = default)
    {
        var request = new
        {
            sample_rate = chunk.SampleRate,
            channel_count = chunk.ChannelCount,
            start_time = chunk.StartTime.ToString("o"),
            eeg_data = chunk.Samples.Select(s => s.Channels).ToArray(),
            spo2 = chunk.Samples.LastOrDefault()?.SpO2,
            heart_rate = chunk.Samples.LastOrDefault()?.HeartRate,
            pulse_wave = chunk.Samples.Select(s => s.PulseWaveValue ?? 0.0).ToArray(),
        };
        var json = JsonConvert.SerializeObject(request);
        var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

        try
        {
            var response = await _http.PostAsync("process", content, ct);
            response.EnsureSuccessStatusCode();
            var body = await response.Content.ReadAsStringAsync(ct);
            var result = JsonConvert.DeserializeObject<ProcessedEEGResult>(body)!;
            result.Timestamp = chunk.StartTime;
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Inference service unavailable — stub result");
            return new ProcessedEEGResult { Timestamp = chunk.StartTime, BIS = double.NaN, SQI = 0 };
        }
    }
}
