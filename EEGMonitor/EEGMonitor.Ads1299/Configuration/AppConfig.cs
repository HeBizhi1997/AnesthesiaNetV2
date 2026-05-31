using Microsoft.Extensions.Configuration;
using System.IO;

namespace Ads1299Monitor.Configuration;

/// <summary>Strongly-typed view over appsettings.json. Loaded once at startup.</summary>
public sealed class AppConfig
{
    public SerialConfig Serial { get; init; } = new();
    public PythonConfig Python { get; init; } = new();
    public RecordingConfig Recording { get; init; } = new();

    public static AppConfig Load()
    {
        var baseDir = AppContext.BaseDirectory;
        var cfg = new ConfigurationBuilder()
            .SetBasePath(baseDir)
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: false)
            .Build();

        var app = new AppConfig();
        cfg.Bind(app);
        return app;
    }
}

public sealed class SerialConfig
{
    public string EegPort { get; set; } = "COM6";
    public int EegBaud { get; set; } = 230400;
    public int EegSampleRate { get; set; } = 250;
    public byte EegGain { get; set; } = 12;
    public bool EegDifferential { get; set; } = true;
    public bool PulseEnabled { get; set; } = false;
    public string PulsePort { get; set; } = "COM5";
}

public sealed class PythonConfig
{
    public string BaseUrl { get; set; } = "http://localhost:8765/";
}

public sealed class RecordingConfig
{
    public bool Enabled { get; set; } = true;
    public string OutputDirectory { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                     "Ads1299Monitor", "Recordings");
}
