using EEGMonitor.Models;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System.Net.Http;

namespace EEGMonitor.Services.Processing;

/// <summary>
/// HTTP client that sends EEG data chunks to the Python FastAPI processing service
/// and deserializes the processed results (band waves, power ratios, BIS, DSA, etc.).
/// </summary>
public sealed class EEGProcessingClient : IEEGProcessingClient
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
        try
        {
            var resp = await _http.GetAsync("health", ct);
            return resp.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task ResetSessionAsync(CancellationToken ct = default)
    {
        try
        {
            await _http.PostAsync("reset", null, ct);
            _logger.LogInformation("Processing session state reset");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to reset processing session (service may be offline)");
        }
    }

    public async Task<ProcessedEEGResult> ProcessChunkAsync(EEGDataChunk chunk, CancellationToken ct = default)
    {
        // Build request DTO
        var request = new
        {
            sample_rate = chunk.SampleRate,
            channel_count = chunk.ChannelCount,
            start_time = chunk.StartTime.ToString("o"),
            // Shape: (n_samples, n_channels)
            eeg_data = chunk.Samples.Select(s => s.Channels).ToArray(),
            spo2 = chunk.Samples.LastOrDefault()?.SpO2,
            heart_rate = chunk.Samples.LastOrDefault()?.HeartRate,
            pulse_wave = chunk.Samples.Select(s => s.PulseWaveValue ?? 0.0).ToArray(),
            device_band_powers_db = chunk.DeviceBandPowersDb,
        };

        var json = JsonConvert.SerializeObject(request);
        var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

        ProcessedEEGResult result;
        try
        {
            var response = await _http.PostAsync("process", content, ct);
            response.EnsureSuccessStatusCode();
            var responseJson = await response.Content.ReadAsStringAsync(ct);
            result = DeserializeResult(responseJson, chunk.StartTime);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Processing service unavailable – returning stub result");
            result = BuildStubResult(chunk);
        }

        return result;
    }

    private static ProcessedEEGResult DeserializeResult(string json, DateTime timestamp)
    {
        var result = JsonConvert.DeserializeObject<ProcessedEEGResult>(json)!;
        result.Timestamp = timestamp;
        return result;
    }

    private static ProcessedEEGResult BuildStubResult(EEGDataChunk chunk) => new()
    {
        Timestamp = chunk.StartTime,
        BIS = double.NaN,
        SQI = 0,
    };
}
