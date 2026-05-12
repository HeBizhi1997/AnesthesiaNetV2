using Microsoft.Extensions.Logging;
using System.Diagnostics;
using System.IO;
using System.Net.Http;

namespace EEGMonitor.Services.PythonService;

/// <summary>
/// Ensures the FastAPI processing service (main.py) is running.
/// On startup: checks the /health endpoint; if offline, locates main.py by walking
/// the directory tree upward and spawns a new Python process.
/// On WPF exit: kills the process that was started (only if WE started it).
/// </summary>
public sealed class PythonServiceLauncher : IAsyncDisposable
{
    private const string HEALTH_URL = "http://localhost:8765/health";
    private static readonly string[] PYTHON_EXECUTABLES = ["python", "py", "python3"];

    private readonly ILogger<PythonServiceLauncher> _logger;
    private Process? _process;

    public PythonServiceLauncher(ILogger<PythonServiceLauncher> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Checks if the service is already running; if not, locates main.py and starts it.
    /// Returns true once the /health endpoint responds.
    /// </summary>
    public async Task<bool> EnsureRunningAsync(CancellationToken ct = default)
    {
        if (await PingAsync(ct))
        {
            _logger.LogInformation("EEG processing service already running");
            return true;
        }

        var serviceDir = FindServiceDir();
        if (serviceDir == null)
        {
            _logger.LogWarning("Could not locate EEGProcessingService/main.py – start it manually");
            return false;
        }

        if (!TryStartProcess(serviceDir))
            return false;

        // Poll up to 15 seconds for the service to become ready
        for (int i = 0; i < 30; i++)
        {
            await Task.Delay(500, ct);
            if (await PingAsync(ct))
            {
                _logger.LogInformation("EEG processing service is ready");
                return true;
            }
        }

        _logger.LogWarning("EEG processing service started but did not respond within 15 s");
        return false;
    }

    private bool TryStartProcess(string serviceDir)
    {
        foreach (var exe in PYTHON_EXECUTABLES)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = exe,
                    Arguments = "main.py",
                    WorkingDirectory = serviceDir,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                };

                var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
                proc.OutputDataReceived += (_, e) =>
                { if (e.Data != null) _logger.LogDebug("[PySvc] {Line}", e.Data); };
                proc.ErrorDataReceived += (_, e) =>
                { if (e.Data != null) _logger.LogDebug("[PySvc] {Line}", e.Data); };

                proc.Start();
                proc.BeginOutputReadLine();
                proc.BeginErrorReadLine();
                _process = proc;
                _logger.LogInformation("Started EEG processing service with '{Exe}' (PID {Pid})", exe, proc.Id);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogDebug("Could not start with '{Exe}': {Msg}", exe, ex.Message);
            }
        }

        _logger.LogWarning("No usable Python executable found – install Python and ensure it is on PATH");
        return false;
    }

    private static async Task<bool> PingAsync(CancellationToken ct)
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
            var resp = await http.GetAsync(HEALTH_URL, ct);
            return resp.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    /// <summary>Walk up the directory tree to locate EEGProcessingService/main.py.</summary>
    private static string? FindServiceDir()
    {
        var dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, "EEGMonitor", "EEGProcessingService", "main.py");
            if (File.Exists(candidate))
                return Path.GetDirectoryName(candidate);
            dir = dir.Parent;
        }
        return null;
    }

    public async ValueTask DisposeAsync()
    {
        if (_process == null) return;
        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
                await _process.WaitForExitAsync().ConfigureAwait(false);
            }
        }
        catch { /* ignore shutdown errors */ }
        finally
        {
            _process.Dispose();
            _process = null;
        }
    }
}
