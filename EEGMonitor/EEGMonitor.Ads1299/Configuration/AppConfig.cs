using Microsoft.Extensions.Configuration;
using System.IO;

namespace Ads1299Monitor.Configuration;

/// <summary>Strongly-typed view over appsettings.json. Loaded once at startup.</summary>
public sealed class AppConfig
{
    public SerialConfig Serial { get; init; } = new();
    public PythonConfig Python { get; init; } = new();
    public RecordingConfig Recording { get; init; } = new();
    public PatientConfig Patient { get; init; } = new();

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

    // PPG / 血氧指夹模组 (CH340) — 提供 SpO₂、PR(脉率)、PRV(脉率变异性)、PI(灌注指数)
    public bool PpgEnabled { get; set; } = true;
    public string PpgPort { get; set; } = "COM7";
    public int PpgBaud { get; set; } = 57600;
}

public sealed class PythonConfig
{
    public string BaseUrl { get; set; } = "http://localhost:8765/";
    /// <summary>When true, the app launches EEGProcessingService/main.py itself if /health is offline.</summary>
    public bool AutoStart { get; set; } = true;
    /// <summary>Optional explicit folder containing main.py. Empty = auto-locate by walking the tree.</summary>
    public string ServiceDirectory { get; set; } = "";
}

public sealed class PatientConfig
{
    public string Name { get; set; } = "张三";
    public string Gender { get; set; } = "男";
    public int Age { get; set; } = 56;
    public string AdmissionNo { get; set; } = "--";
    public string OperatingRoom { get; set; } = "--";
    public string SurgeryName { get; set; } = "--";
    public string AnesthesiaMethod { get; set; } = "全身麻醉";
}

public sealed class RecordingConfig
{
    public bool Enabled { get; set; } = true;
    public string OutputDirectory { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                     "Ads1299Monitor", "Recordings");
}
