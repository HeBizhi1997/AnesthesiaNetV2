using System.IO;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using NSMMonitor.Models;

namespace NSMMonitor.Services;

/// <summary>
/// NSM 数据录制服务：将接收到的数据包逐行写入 JSON Lines 文件（.nsm），
/// 供日后通过 <see cref="NsmPlaybackService"/> 按原始时序回放。
/// </summary>
public sealed class NsmRecordingService : IDisposable
{
    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = false };

    private readonly ILogger<NsmRecordingService> _logger;
    private readonly object _lock = new();
    private StreamWriter? _writer;

    public bool IsRecording { get; private set; }
    public string? CurrentFile { get; private set; }
    public long Count { get; private set; }

    public NsmRecordingService(ILogger<NsmRecordingService> logger) => _logger = logger;

    public void Start(string filePath)
    {
        lock (_lock)
        {
            Stop();
            var dir = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            _writer = new StreamWriter(filePath, append: false) { AutoFlush = false };
            IsRecording = true;
            CurrentFile = filePath;
            Count = 0;
            _logger.LogInformation("开始录制到 {File}", filePath);
        }
    }

    public void Write(NSMDataPacket packet)
    {
        lock (_lock)
        {
            if (!IsRecording || _writer == null) return;
            _writer.WriteLine(JsonSerializer.Serialize(packet, JsonOpts));
            Count++;
            if (Count % 16 == 0) _writer.Flush();
        }
    }

    public void Stop()
    {
        lock (_lock)
        {
            if (_writer == null) { IsRecording = false; return; }
            _writer.Flush();
            _writer.Dispose();
            _writer = null;
            IsRecording = false;
            _logger.LogInformation("停止录制：{File}，共 {Count} 包", CurrentFile, Count);
        }
    }

    public void Dispose() => Stop();
}
