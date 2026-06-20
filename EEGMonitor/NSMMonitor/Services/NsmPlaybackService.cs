using System.IO;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using NSMMonitor.Models;

namespace NSMMonitor.Services;

/// <summary>
/// NSM 文件回放服务：读取录制的 .nsm（JSON Lines）文件，
/// 按数据包原始时间间隔重放，实现 <see cref="INsmDataSource"/> 接口，
/// 与真实串口/模拟器共用同一套界面绑定。
/// </summary>
public sealed class NsmPlaybackService : INsmDataSource, IDisposable
{
    private readonly ILogger<NsmPlaybackService> _logger;
    private CancellationTokenSource? _cts;
    private Task? _task;
    private long _played;

    public bool IsConnected { get; private set; }
    public string SourceName => IsConnected ? $"回放 {Path.GetFileName(_filePath)}" : "回放未加载";
    public int SampleRate => 100;
    public long PacketsDecoded => _played;

    private string _filePath = "";

    public event Action<NSMDataPacket>? NSMDataReceived;
    public event Action<string>? StatusChanged;
    public event Action<Exception>? ErrorOccurred;
    public event Action? PlaybackCompleted;

    public NsmPlaybackService(ILogger<NsmPlaybackService> logger) => _logger = logger;

    public IEnumerable<string> GetAvailablePorts() => Array.Empty<string>();

    /// <summary>portName 此处用作回放文件路径。</summary>
    public bool Connect(string portName, int baudRate = 115200)
    {
        if (IsConnected) Disconnect();
        if (string.IsNullOrWhiteSpace(portName) || !File.Exists(portName))
        {
            StatusChanged?.Invoke("回放文件不存在");
            return false;
        }

        _filePath = portName;
        _played = 0;
        _cts = new CancellationTokenSource();
        IsConnected = true;
        _task = Task.Run(() => ReplayLoop(_cts.Token));
        _logger.LogInformation("开始回放 {File}", portName);
        StatusChanged?.Invoke($"开始回放：{Path.GetFileName(portName)}");
        return true;
    }

    private async Task ReplayLoop(CancellationToken ct)
    {
        try
        {
            var packets = new List<NSMDataPacket>();
            foreach (var line in File.ReadLines(_filePath))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var p = JsonSerializer.Deserialize<NSMDataPacket>(line);
                if (p != null) packets.Add(p);
            }

            if (packets.Count == 0)
            {
                StatusChanged?.Invoke("回放文件为空");
                Finish();
                return;
            }

            DateTime? prev = null;
            foreach (var pkt in packets)
            {
                ct.ThrowIfCancellationRequested();

                if (prev.HasValue)
                {
                    double deltaMs = (pkt.LocalTimestamp - prev.Value).TotalMilliseconds;
                    int delay = (int)Math.Clamp(deltaMs, 50, 2000);
                    await Task.Delay(delay, ct);
                }
                prev = pkt.LocalTimestamp;

                NSMDataReceived?.Invoke(pkt);
                _played++;
                if (_played == 1 || _played % 16 == 0)
                    StatusChanged?.Invoke($"回放中：{_played}/{packets.Count} 包");
            }

            StatusChanged?.Invoke($"回放结束：共 {packets.Count} 包");
            Finish();
        }
        catch (OperationCanceledException)
        {
            // 正常停止
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "回放出错");
            ErrorOccurred?.Invoke(ex);
            Finish();
        }
    }

    private void Finish()
    {
        IsConnected = false;
        PlaybackCompleted?.Invoke();
    }

    public void Disconnect()
    {
        try { _cts?.Cancel(); } catch { /* ignore */ }
        IsConnected = false;
        _logger.LogInformation("回放已停止");
        StatusChanged?.Invoke("回放已停止");
    }

    public void Dispose()
    {
        Disconnect();
        _cts?.Dispose();
    }
}
