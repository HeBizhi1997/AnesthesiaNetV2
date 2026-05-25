using System.Text.Json;
using EEGMonitor.AnesthesiaSleep.Models;

namespace EEGMonitor.AnesthesiaSleep.Infrastructure.ReportGeneration;

public class ReportGenerator
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public async Task<string> GenerateJsonAsync(SleepReport report, string outputPath)
    {
        var json = JsonSerializer.Serialize(report, JsonOptions);
        await System.IO.File.WriteAllTextAsync(outputPath, json);
        return outputPath;
    }

    public string GenerateJsonString(SleepReport report)
    {
        return JsonSerializer.Serialize(report, JsonOptions);
    }

    public string GenerateCsvSummary(SleepReport report)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Metric,Value");
        var s = report.Statistics;
        sb.AppendLine($"PatientID,{report.PatientId}");
        sb.AppendLine($"RecordingStart,{report.RecordingStart:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"RecordingEnd,{report.RecordingEnd:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"TST_min,{s.TSTMinutes:F1}");
        sb.AppendLine($"SE_pct,{s.SEPercent:F1}");
        sb.AppendLine($"SOL_min,{s.SOLMinutes:F1}");
        sb.AppendLine($"WASO_min,{s.WASOMinutes:F1}");
        sb.AppendLine($"N1_pct,{s.N1Percent:F1}");
        sb.AppendLine($"N2_pct,{s.N2Percent:F1}");
        sb.AppendLine($"N3_pct,{s.N3Percent:F1}");
        sb.AppendLine($"REM_pct,{s.REMPercent:F1}");
        sb.AppendLine($"REM_latency_min,{s.REMLatencyMinutes:F1}");
        sb.AppendLine($"ArousalIndex,{s.ArousalIndex:F1}");
        sb.AppendLine($"ODI3,{report.ODI3:F1}");
        sb.AppendLine($"ODI4,{report.ODI4:F1}");
        sb.AppendLine($"MeanSpO2,{report.MeanSpO2:F1}");
        sb.AppendLine($"MinSpO2,{report.MinSpO2:F1}");
        sb.AppendLine($"SpO2Below90_pct,{report.SpO2Below90Percent:F1}");
        sb.AppendLine($"TotalEpochs,{s.TotalEpochs}");
        sb.AppendLine($"ArtifactEpochs,{s.ArtifactEpochs}");
        return sb.ToString();
    }
}
