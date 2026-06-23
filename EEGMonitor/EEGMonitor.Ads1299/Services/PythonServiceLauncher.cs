using Ads1299Monitor.Configuration;
using Microsoft.Extensions.Logging;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;

namespace Ads1299Monitor.Services;

/// <summary>
/// Ensures the FastAPI inference service (EEGProcessingService/main.py) is running.
/// On startup: pings /health; if offline, locates main.py (config override or by walking
/// the tree upward) and spawns Python. On app exit: kills ONLY the process we started —
/// never an externally-launched service, and never any unrelated python (e.g. training).
/// </summary>
public sealed class PythonServiceLauncher : IDisposable
{
    private static readonly string[] PythonExecutables = { "python", "py", "python3" };

    private readonly AppConfig _cfg;
    private readonly ILogger<PythonServiceLauncher> _logger;
    private readonly string _healthUrl;
    private Process? _process;          // non-null only if WE started it

    public PythonServiceLauncher(AppConfig cfg, ILogger<PythonServiceLauncher> logger)
    {
        _cfg = cfg;
        _logger = logger;
        _healthUrl = new Uri(new Uri(cfg.Python.BaseUrl), "health").ToString();
    }

    /// <summary>True if the service responds; starts it first if needed (and AutoStart is on).</summary>
    public async Task<bool> EnsureRunningAsync(CancellationToken ct = default)
    {
        if (await PingAsync(ct))
        {
            _logger.LogInformation("Inference service already running");
            return true;
        }
        if (!_cfg.Python.AutoStart)
        {
            _logger.LogInformation("Inference service offline and AutoStart disabled");
            return false;
        }

        var serviceDir = ResolveServiceDir();
        if (serviceDir == null)
        {
            _logger.LogWarning("Could not locate EEGProcessingService/main.py — set Python:ServiceDirectory in appsettings.json");
            return false;
        }
        if (!TryStartProcess(serviceDir, ResolvePythonExecutables(serviceDir))) return false;

        for (int i = 0; i < 40; i++)        // poll up to 20 s (model load can take a while)
        {
            await Task.Delay(500, ct);
            if (await PingAsync(ct)) { _logger.LogInformation("Inference service is ready"); return true; }
        }
        _logger.LogWarning("Inference service started but did not respond in time");
        return false;
    }

    // Prefer the project venv interpreter (EEGProcessingService/.venv) — the pinned deps
    // (numpy/scipy/torch) only have wheels up to Python 3.12, so a bare PATH `python`
    // (e.g. 3.14) can't run the service. Fall back to PATH executables if no venv.
    private static string[] ResolvePythonExecutables(string serviceDir)
    {
        var venvPy = Path.Combine(serviceDir, ".venv", "Scripts", "python.exe");
        return File.Exists(venvPy)
            ? new[] { venvPy }.Concat(PythonExecutables).ToArray()
            : PythonExecutables;
    }

    private bool TryStartProcess(string serviceDir, string[] executables)
    {
        foreach (var exe in executables)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = exe, Arguments = "main.py", WorkingDirectory = serviceDir,
                    UseShellExecute = false, CreateNoWindow = true,
                    RedirectStandardOutput = true, RedirectStandardError = true,
                };
                var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
                proc.OutputDataReceived += (_, e) => { if (e.Data != null) _logger.LogDebug("[PySvc] {Line}", e.Data); };
                proc.ErrorDataReceived  += (_, e) => { if (e.Data != null) _logger.LogDebug("[PySvc] {Line}", e.Data); };
                proc.Start();
                proc.BeginOutputReadLine();
                proc.BeginErrorReadLine();
                _process = proc;
                _logger.LogInformation("Started inference service via '{Exe}' (PID {Pid}) in {Dir}", exe, proc.Id, serviceDir);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogDebug("Could not start with '{Exe}': {Msg}", exe, ex.Message);
            }
        }
        _logger.LogWarning("No usable Python executable found on PATH");
        return false;
    }

    private async Task<bool> PingAsync(CancellationToken ct)
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
            return (await http.GetAsync(_healthUrl, ct)).IsSuccessStatusCode;
        }
        catch { return false; }
    }

    private string? ResolveServiceDir()
    {
        // 1. explicit config override
        var cfgDir = _cfg.Python.ServiceDirectory;
        if (!string.IsNullOrWhiteSpace(cfgDir) && File.Exists(Path.Combine(cfgDir, "main.py")))
            return cfgDir;

        // 2. walk up from the executable looking for EEGMonitor/EEGProcessingService/main.py
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, "EEGMonitor", "EEGProcessingService", "main.py");
            if (File.Exists(candidate)) return Path.GetDirectoryName(candidate);
            dir = dir.Parent;
        }
        return null;
    }

    public void Dispose()
    {
        if (_process == null) return;     // we didn't start it → leave external service alone
        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
                _process.WaitForExit(3000);
            }
        }
        catch { /* ignore shutdown errors */ }
        finally { _process.Dispose(); _process = null; }
    }
}
