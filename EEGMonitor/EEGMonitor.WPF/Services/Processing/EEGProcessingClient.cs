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
        dynamic d = JsonConvert.DeserializeObject<dynamic>(json)!;
        var result = new ProcessedEEGResult
        {
            Timestamp = timestamp,
            BIS = (double)(d.bis ?? double.NaN),
            QNox = null,
            SPI = null,
            SQI = (double)(d.sqi ?? 0.0),
            StateEntropy = (double?)(d.se),
            ResponseEntropy = (double?)(d.re),
            DeltaWave = ToDoubleArray(d.delta_wave),
            ThetaWave = ToDoubleArray(d.theta_wave),
            AlphaWave = ToDoubleArray(d.alpha_wave),
            BetaWave = ToDoubleArray(d.beta_wave),
            GammaWave = ToDoubleArray(d.gamma_wave),
            RawEEG = ToDoubleArray(d.raw_eeg),
            DeltaPower = (double)(d.delta_power ?? 0.0),
            ThetaPower = (double)(d.theta_power ?? 0.0),
            AlphaPower = (double)(d.alpha_power ?? 0.0),
            BetaPower = (double)(d.beta_power ?? 0.0),
            GammaPower = (double)(d.gamma_power ?? 0.0),
            DSAFrequencies = ToDoubleArray(d.dsa_frequencies),
            DSATimes = ToDoubleArray(d.dsa_times),
            DSAMatrix = To2DArray(d.dsa_matrix),
            HeartRate = (double?)(d.heart_rate),
            HRV_RMSSD = (double?)(d.hrv_rmssd),
            PulseWave = ToDoubleArray(d.pulse_wave),
            SpO2 = (double?)(d.spo2),
        };
        return result;
    }

    private static double[] ToDoubleArray(dynamic? arr)
    {
        if (arr == null) return Array.Empty<double>();
        var list = new List<double>();
        foreach (var v in arr) list.Add((double)v);
        return list.ToArray();
    }

    private static double[,] To2DArray(dynamic? matrix)
    {
        if (matrix == null) return new double[0, 0];
        var rows = new List<double[]>();
        foreach (var row in matrix) rows.Add(ToDoubleArray(row));
        if (rows.Count == 0) return new double[0, 0];
        var result = new double[rows.Count, rows[0].Length];
        for (int i = 0; i < rows.Count; i++)
            for (int j = 0; j < rows[i].Length; j++)
                result[i, j] = rows[i][j];
        return result;
    }

    private static ProcessedEEGResult BuildStubResult(EEGDataChunk chunk) => new()
    {
        Timestamp = chunk.StartTime,
        BIS = double.NaN,
        SQI = 0,
    };
}
