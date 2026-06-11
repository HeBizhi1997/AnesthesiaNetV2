using System.IO;
using System.Text.Json;

namespace EEGRecorder.Configuration;

/// <summary>Strongly-typed view over appsettings.json (next to the exe). Edit it on the laptop
/// to set COM ports / output folder — no rebuild needed.</summary>
public sealed class RecorderConfig
{
    public SerialConfig Serial { get; set; } = new();
    public RecordingConfig Recording { get; set; } = new();
    public SessionConfig Session { get; set; } = new();

    public static RecorderConfig Load()
    {
        try
        {
            var path = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
            if (File.Exists(path))
            {
                var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                return JsonSerializer.Deserialize<RecorderConfig>(File.ReadAllText(path), opts) ?? new();
            }
        }
        catch { /* fall back to defaults */ }
        return new();
    }
}

public sealed class SerialConfig
{
    public string EegPort { get; set; } = "COM6";
    public int EegBaud { get; set; } = 230400;
    public int EegSampleRate { get; set; } = 250;
    public byte EegGain { get; set; } = 12;
    public bool EegDifferential { get; set; } = true;
    public bool PpgEnabled { get; set; } = true;
    public string PpgPort { get; set; } = "COM7";
    public int PpgBaud { get; set; } = 57600;
}

public sealed class RecordingConfig
{
    public string OutputDirectory { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "EEGRecorder");
}

public sealed class SessionConfig
{
    public string Subject { get; set; } = "S001";
    public string Note { get; set; } = "";
}
