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
///   <item>设备完整帧格式：355 字节定长协议帧（首字节 0x80，长度字段=351），
///         含全部字段（频带功率 / EOG / SEF95），详见《NSM 设备通讯协议文档》。</item>
///   <item>设备紧凑记录格式：128 字节定长记录（首字节 0x80），
///         即协议帧前 126 字节（帧头…NOX）+ 2 字节 CRC，不含频带功率 / EOG / SEF95。</item>
/// </list>
/// </summary>
public sealed class NsmPlaybackService : INsmDataSource, IDisposable
{
    // ── 设备二进制记录格式 ──
    private const byte FRAME_HEADER = 0x80;
    private const int RECORD_SIZE = 128;          // 紧凑记录（仅 EEG + 指标，至 NOX）
    private const int FULL_FRAME_SIZE = 355;       // 完整协议帧（含频带功率 / EOG / SEF95）
    private const int PAYLOAD_LENGTH = 351;        // 完整帧长度字段值（帧头后 2 字节）
    private const int REC_EEG_OFFSET = 22;
    private const int REC_EEG_SAMPLES = 100;
    private readonly ILogger<NsmPlaybackService> _logger;
    private CancellationTokenSource? _cts;
    private Task? _task;
    private long _played;

    private List<NSMDataPacket> _packets = new();
    private int _position;
    private volatile int _seekRequest = -1;
    private double _speed = 1.0;
    private DateTime _firstTs;

    public bool IsConnected { get; private set; }
    public string SourceName => IsConnected ? $"回放 {Path.GetFileName(_filePath)}" : "回放未加载";
    public int SampleRate => 100;
    public long PacketsDecoded => _played;

    /// <summary>回放倍率（0.1–64×），可在回放过程中实时调整。</summary>
    public double Speed { get => _speed; set => _speed = Math.Clamp(value, 0.1, 64); }

    private string _filePath = "";

    public event Action<NSMDataPacket>? NSMDataReceived;
    public event Action<string>? StatusChanged;
    public event Action<Exception>? ErrorOccurred;
    public event Action? PlaybackCompleted;
    /// <summary>回放进度更新（当前帧索引 / 总帧数 / 已播时长 / 总时长）。</summary>
    public event Action<PlaybackProgress>? ProgressChanged;
    /// <summary>用户拖动进度后已跳转：传出全部数据帧与新位置索引，界面据此重建累积图表（趋势 / DSA / 事件）。</summary>
    public event Action<IReadOnlyList<NSMDataPacket>, int>? Seeked;

    /// <summary>请求跳转到指定帧索引（线程安全，回放循环将在下一步处理）。</summary>
    public void SeekToIndex(int index)
    {
        if (_packets.Count == 0) return;
        _seekRequest = Math.Clamp(index, 0, _packets.Count - 1);
    }

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
        _position = 0;
        _seekRequest = -1;
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
            _packets = IsBinaryRecordFile(_filePath)
                ? ReadBinaryRecords(_filePath)
                : ReadJsonLines(_filePath);

            if (_packets.Count == 0)
            {
                StatusChanged?.Invoke("回放文件为空");
                Finish();
                return;
            }

            _firstTs = _packets[0].LocalTimestamp;
            TimeSpan duration = _packets[^1].LocalTimestamp - _firstTs;

            int i = 0;
            DateTime? prev = null;
            while (i < _packets.Count)
            {
                ct.ThrowIfCancellationRequested();

                // 处理拖动跳转
                int seek = _seekRequest;
                if (seek >= 0)
                {
                    _seekRequest = -1;
                    i = seek;
                    prev = null;
                    Seeked?.Invoke(_packets, i);   // 由界面重建 [0, i) 的累积图表，随后正常推送第 i 帧
                }

                var pkt = _packets[i];
                if (prev.HasValue)
                {
                    double deltaMs = (pkt.LocalTimestamp - prev.Value).TotalMilliseconds;
                    double baseDelay = Math.Clamp(deltaMs, 50, 2000);
                    int delay = (int)Math.Max(1, baseDelay / _speed);
                    if (await WaitOrSeek(delay, ct)) continue;   // 等待期间收到跳转 → 重新求值
                }
                prev = pkt.LocalTimestamp;

                NSMDataReceived?.Invoke(pkt);
                _position = i;
                _played++;
                ProgressChanged?.Invoke(new PlaybackProgress(
                    i, _packets.Count, pkt.LocalTimestamp - _firstTs, duration));
                if (_played == 1 || _played % 16 == 0)
                    StatusChanged?.Invoke($"回放中：{i + 1}/{_packets.Count} 包");
                i++;
            }

            StatusChanged?.Invoke($"回放结束：共 {_packets.Count} 包");
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

    /// <summary>分片等待，使拖动跳转能在 ~40ms 内被响应；返回 true 表示等待期间收到跳转请求。</summary>
    private async Task<bool> WaitOrSeek(int delayMs, CancellationToken ct)
    {
        int waited = 0;
        while (waited < delayMs)
        {
            if (_seekRequest >= 0) return true;
            int step = Math.Min(40, delayMs - waited);
            await Task.Delay(step, ct);
            waited += step;
        }
        return _seekRequest >= 0;
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

    /// <summary>
    /// 读取设备采集的定长二进制记录。按帧头后 2 字节长度字段区分：
    /// 等于 351 为完整 355 字节协议帧（含频带功率 / EOG / SEF95），否则按 128 字节紧凑记录处理。
    /// </summary>
    private List<NSMDataPacket> ReadBinaryRecords(string path)
    {
        var bytes = File.ReadAllBytes(path);
        int lengthField = bytes.Length >= 3 ? bytes[1] | (bytes[2] << 8) : 0;
        bool full = lengthField == PAYLOAD_LENGTH;
        int size = full ? FULL_FRAME_SIZE : RECORD_SIZE;

        int recordCount = bytes.Length / size;
        var packets = new List<NSMDataPacket>(recordCount);
        for (int i = 0; i < recordCount; i++)
        {
            int off = i * size;
            if (bytes[off] != FRAME_HEADER) continue;   // 跳过未对齐/损坏记录
            try { packets.Add(full ? ParseFullFrame(bytes, off) : ParseRecord(bytes, off)); }
            catch (Exception ex) { _logger.LogWarning(ex, "解析 NSM 记录 #{Index} 出错", i); }
        }
        return packets;
    }

    /// <summary>解析完整 355 字节协议帧：在紧凑记录字段基础上补充频带功率 / EOG / SEF95 / DSA。</summary>
    private static NSMDataPacket ParseFullFrame(byte[] buf, int off)
    {
        // DSA：偏移 132 起 44 字节（无符号强度）。电极报警/失效帧（状态位 0x82）按设备约定清零。
        var dsa = new byte[44];
        if ((buf[off + 10] & 0x82) == 0)
            Array.Copy(buf, off + 132, dsa, 0, 44);

        return ParseRecord(buf, off) with
        {
            DeltaPowerDb = (sbyte)buf[off + 126],
            ThetaPowerDb = (sbyte)buf[off + 127],
            AlphaPowerDb = (sbyte)buf[off + 128],
            BetaPowerDb  = (sbyte)buf[off + 129],
            GammaPowerDb = (sbyte)buf[off + 130],
            EOG   = buf[off + 131],
            SEF95 = buf[off + 176],
            Dsa = dsa,
        };
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

/// <summary>回放进度快照。</summary>
public readonly record struct PlaybackProgress(int Index, int Total, TimeSpan Elapsed, TimeSpan Duration);
