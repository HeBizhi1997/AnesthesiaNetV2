using Serilog;
using Serilog.Events;
using Serilog.Formatting.Compact;
using System.IO;

namespace EEGMonitor.Infrastructure.Logging;

public static class SerilogSetup
{
    public static void Configure()
    {
        var logsDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "EEGMonitor", "Logs");

        Directory.CreateDirectory(logsDir);

        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .MinimumLevel.Override("Microsoft", LogEventLevel.Warning)
            .MinimumLevel.Override("System", LogEventLevel.Warning)
            .Enrich.FromLogContext()
            .Enrich.WithThreadId()
            .Enrich.WithProperty("Application", "EEGMonitor")
            // Structured JSON log (for analysis)
            .WriteTo.File(
                new CompactJsonFormatter(),
                Path.Combine(logsDir, "eegmonitor-.jsonl"),
                rollingInterval: RollingInterval.Day,
                retainedFileCountLimit: 30,
                fileSizeLimitBytes: 100 * 1024 * 1024)
            // Human-readable log
            .WriteTo.File(
                Path.Combine(logsDir, "eegmonitor-.log"),
                rollingInterval: RollingInterval.Day,
                retainedFileCountLimit: 7,
                outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff} [{Level:u3}] [{ThreadId}] {Message:lj}{NewLine}{Exception}")
            // Debug console
            .WriteTo.Console(
                outputTemplate: "{Timestamp:HH:mm:ss} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
            .CreateLogger();
    }
}
