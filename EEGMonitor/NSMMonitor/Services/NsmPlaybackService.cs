using System.IO;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using NSMMonitor.Models;

namespace NSMMonitor.Services;

/// <summary>
/// NSM 文件回放服务：读取 .nsm 录制文件，按数据包原始时间间隔重放，
/// 实现 <see cref="INsmDataSource"/> 接口，与真实串口/模拟器共用同一套界面绑定。
/// <para>支持两种 .nsm 格式，按首字节自动识别：</para>
/// <list type="bullet">
///   <item>本机录制格式：JSON Lines（每行一个 <see cref="NSMDataPacket"/>，首字符 '{'）。</item>
///   <item>设备采集格式：128 字节定长二进制记录（首字节 0x80），
///         即协议帧前 126 字节（帧头…NOX）+ 2 字节 CRC，详见《NSM 设备通讯协议文档》。
///         此格式不含频带功率 / EOG / SEF95 字段。</item>
/// </list>
/// </summary>
public sealed class NsmPlaybackService : INsmDataSource, IDisposable
{
    // ── 设备二进制记录格式（128 字节）──
    private const byte FRAME_HEADER = 0x80;
    private const int RECORD_SIZE = 128;
    private const int REC_EEG_OFFSET = 22;
    private const int REC_EEG_SAMPLES = 100;
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
            var packets = IsBinaryRecordFile(_filePath)
                ? ReadBinaryRecords(_filePath)
                : ReadJsonLines(_filePath);

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

    // ─────────────────────────── 格式识别与读取 ───────────────────────────

    /// <summary>按首个有效字节判别：0x80 为设备二进制记录，否则按 JSON Lines 处理。</summary>
    private static bool IsBinaryRecordFile(string path)
    {
        try
        {
            using var fs = File.OpenRead(path);
            int first = fs.ReadByte();
            return first == FRAME_HEADER;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>读取本机录制的 JSON Lines 格式。</summary>
    private static List<NSMDataPacket> ReadJsonLines(string path)
    {
        var packets = new List<NSMDataPacket>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var p = JsonSerializer.Deserialize<NSMDataPacket>(line);
            if (p != null) packets.Add(p);
        }
        return packets;
    }

    /// <summary>读取设备采集的 128 字节定长二进制记录。</summary>
    private List<NSMDataPacket> ReadBinaryRecords(string path)
    {
        var bytes = File.ReadAllBytes(path);
        int recordCount = bytes.Length / RECORD_SIZE;
        var packets = new List<NSMDataPacket>(recordCount);
        for (int i = 0; i < recordCount; i++)
        {
            int off = i * RECORD_SIZE;
            if (bytes[off] != FRAME_HEADER) continue;   // 跳过未对齐/损坏记录
            try { packets.Add(ParseRecord(bytes, off)); }
            catch (Exception ex) { _logger.LogWarning(ex, "解析 NSM 记录 #{Index} 出错", i); }
        }
        return packets;
    }

    /// <summary>解析单条 128 字节记录，偏移定义与协议帧前段一致（帧头…NOX）。</summary>
    private static NSMDataPacket ParseRecord(byte[] buf, int off)
    {
        // 设备时间为大端 Unix 秒（Sec4@6 为高位字节）
        uint deviceTime = ((uint)buf[off + 6] << 24) | ((uint)buf[off + 7] << 16)
                        | ((uint)buf[off + 8] << 8) | buf[off + 9];
        byte blockStatus = buf[off + 10];

        var eeg = new double[REC_EEG_SAMPLES];
        for (int i = 0; i < REC_EEG_SAMPLES; i++)
            eeg[i] = (sbyte)buf[off + REC_EEG_OFFSET + i];

        var localTime = deviceTime > 0
            ? DateTimeOffset.FromUnixTimeSeconds(deviceTime).ToLocalTime().DateTime
            : DateTime.Now;

        return new NSMDataPacket
        {
            LocalTimestamp = localTime,
            DeviceTimeSec = deviceTime,
            ElectrodeAlarm   = (blockStatus & (1 << 1)) != 0,
            ImpedanceHigh    = (blockStatus & (1 << 3)) != 0,
            ElectrodeInvalid = (blockStatus & (1 << 7)) != 0,
            BlackImpedance = buf[off + 16],
            WhiteImpedance = buf[off + 17],
            CSI = buf[off + 13],
            BS  = buf[off + 14],
            SQI = buf[off + 15],
            EMG = buf[off + 18],
            NOX = buf[off + 125],
            EventNumber = buf[off + 11],
            EventType = (NSMEventType)buf[off + 12],
            AlarmHigh = buf[off + 20],
            AlarmLow  = buf[off + 21],
            EEGSamplesUv = eeg,
            // 128 字节记录格式不含频带功率 / EOG / SEF95
        };
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
