using Microsoft.Extensions.Logging;
using NSMMonitor.Configuration;
using NSMMonitor.Models;
using NSMMonitor.Services.Ppg;
using Ports = System.IO.Ports;
using Timer = System.Timers.Timer;

namespace NSMMonitor.Services;

/// <summary>
/// AFE4490 指夹血氧模组串口服务（协议见仓库根目录 serial-protocol.md，V1.1）。
///   - CH340 USB 虚拟串口，57600 8N1，无流控，单向推送
///   - 17 字节定长帧 @ 125 Hz：0A FA | 0A 00 02 | IR(int32 LE) | RED(int32 LE) | SpO₂ HR | 00 0B
///   - 只解帧、不算指标；SpO₂ / 脉率由 <see cref="Ppg.PulseOximetryProcessor"/> 计算
///
/// 帧同步严格按协议 §6.1 实现（校验 5 个固定字节，假同步字只丢 2 字节重搜）——
/// 负载是任意二进制，同步字 0A FA 在其中偶现的概率不可忽略，只匹配帧头不验帧尾会解出垃圾。
/// </summary>
public sealed class PpgSerialService : IPpgDataSource, IDisposable
{
    /// <summary>协议 §2：DTR 复位后 bootloader 2 s + AFE4490 上电初始化 2.5 s，故 5 秒内无数据不算异常。</summary>
    private static readonly TimeSpan NoDataTimeout = TimeSpan.FromSeconds(5);

    private readonly ILogger<PpgSerialService> _logger;
    private readonly PpgConfig _cfg;
    private readonly PpgFrameParser _parser = new();
    private readonly object _bufLock = new();

    private Ports.SerialPort? _port;
    private Timer? _healthTimer;
    private DateTime _openedAt;
    private DateTime _lastTickAt;
    private long _framesAtLastTick;
    private double _frameRate;
    private double _peakFrameRate;
    private DateTime _lastRateWarnAt;
    private bool _noDataWarned, _strictHintShown;

    // 计数镜像出来用 Volatile 读写，避免 UI 线程为了取一个数去抢 _bufLock（串口线程正持有它解析）
    private long _framesDecoded, _badFrames;

    public bool IsConnected => _port?.IsOpen ?? false;
    public string SourceName => IsConnected ? $"{_port!.PortName}@{_port.BaudRate}" : "未连接";
    public double FrameRate => Volatile.Read(ref _frameRate);
    public long FramesDecoded => Interlocked.Read(ref _framesDecoded);
    public long BadFrames => Interlocked.Read(ref _badFrames);

    public event Action<PpgFrame>? FrameReceived;
    public event Action<string>? StatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public PpgSerialService(ILogger<PpgSerialService> logger, NsmConfig config)
    {
        _logger = logger;
        _cfg = config.Ppg;
    }

    public IEnumerable<string> GetAvailablePorts() => Ports.SerialPort.GetPortNames().OrderBy(p => p);

    public bool Connect(string portName, int baudRate = 57600)
    {
        if (IsConnected) Disconnect();

        lock (_bufLock)
        {
            _parser.Reset();
            _parser.Strict = _cfg.StrictFrameCheck;
        }
        _framesAtLastTick = 0;
        _frameRate = 0;
        _peakFrameRate = 0;
        _framesDecoded = _badFrames = 0;
        _noDataWarned = _strictHintShown = false;
        _openedAt = _lastTickAt = DateTime.UtcNow;
        _lastRateWarnAt = DateTime.MinValue;

        try
        {
            _port = new Ports.SerialPort(portName, baudRate, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadBufferSize = 16384,
                ReadTimeout = 500,
                WriteTimeout = 500,
                // 厂商抓包与 scripts/ppg_demo.py 均以 DTR/RTS 拉低打开；置高会让模组静默。
                DtrEnable = false,
                RtsEnable = false,
            };
            _port.DataReceived += OnDataReceived;
            _port.ErrorReceived += OnErrorReceived;
            _port.Open();
            _port.DiscardInBuffer();

            _healthTimer = new Timer(1000) { AutoReset = true };
            _healthTimer.Elapsed += (_, _) => CheckLinkHealth();
            _healthTimer.Start();

            _logger.LogInformation("PPG 串口 {Port} 已打开 @ {Baud}", portName, baudRate);
            StatusChanged?.Invoke($"血氧已连接 {portName}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "打开 PPG 串口 {Port} 失败", portName);
            ErrorOccurred?.Invoke(ex);
            return false;
        }
    }

    /// <summary>
    /// 断开。CH340 的 <c>SerialPort.Close()</c> 在驱动层可能长时间不返回（拔线、缓冲未排空等），
    /// 因此放到后台线程关闭并设上限等待：最坏情况是端口延迟释放（下次连接报错，可见且可恢复），
    /// 而不是把 UI 线程永久钉死。
    /// </summary>
    public void Disconnect()
    {
        if (_healthTimer != null)
        {
            _healthTimer.Stop();
            _healthTimer.Dispose();
            _healthTimer = null;
        }

        var port = Interlocked.Exchange(ref _port, null);
        if (port == null) return;

        // 先摘事件：之后不会再有新的回调进来
        port.DataReceived -= OnDataReceived;
        port.ErrorReceived -= OnErrorReceived;

        var closing = Task.Run(() =>
        {
            try { if (port.IsOpen) port.Close(); } catch (Exception ex) { _logger.LogWarning(ex, "关闭 PPG 串口时出错"); }
            try { port.Dispose(); } catch { /* Dispose 也可能抛，无所谓了 */ }
        });
        if (!closing.Wait(TimeSpan.FromSeconds(2)))
            _logger.LogWarning("PPG 串口关闭超过 2 秒未返回，转入后台等待（端口可能延迟释放）");

        lock (_bufLock) _parser.Reset();
        _framesDecoded = _badFrames = 0;
        _frameRate = 0;

        _logger.LogInformation("PPG 串口已断开");
        StatusChanged?.Invoke("血氧已断开");
    }

    private void OnDataReceived(object sender, Ports.SerialDataReceivedEventArgs e)
    {
        var port = _port;
        if (port == null || !port.IsOpen) return;
        try
        {
            int avail = port.BytesToRead;
            if (avail <= 0) return;
            var chunk = new byte[avail];
            int read = port.Read(chunk, 0, avail);

            IReadOnlyList<PpgFrame> frames;
            lock (_bufLock)
            {
                frames = _parser.Feed(chunk.AsSpan(0, read));
                Interlocked.Exchange(ref _framesDecoded, _parser.FramesDecoded);
                Interlocked.Exchange(ref _badFrames, _parser.BadFrames);
            }

            // 事件在锁外触发：下游要刷 UI / 落盘，不能占着缓冲锁。
            for (int i = 0; i < frames.Count; i++) FrameReceived?.Invoke(frames[i]);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "读取 PPG 串口数据出错");
            ErrorOccurred?.Invoke(ex);
        }
    }

    private void OnErrorReceived(object sender, Ports.SerialErrorReceivedEventArgs e) =>
        _logger.LogWarning("PPG 串口错误：{Error}", e.EventType);

    /// <summary>每秒一次的链路健康检查，判据见协议 §6.3。</summary>
    private void CheckLinkHealth()
    {
        long total = FramesDecoded;
        // 除以实际间隔而非假定 1 秒：System.Timers.Timer 会漂移，
        // 直接拿计数当速率会让 §6.3 的 125±2 判据失真。
        var now = DateTime.UtcNow;
        double dt = (now - _lastTickAt).TotalSeconds;
        if (dt > 0.2) Volatile.Write(ref _frameRate, (total - _framesAtLastTick) / dt);
        _lastTickAt = now;
        _framesAtLastTick = total;

        // 开口 5 秒后仍无任何字节 → 固件未运行或端口错误
        if (total == 0 && !_noDataWarned && DateTime.UtcNow - _openedAt > NoDataTimeout)
        {
            _noDataWarned = true;
            long bad = BadFrames;
            if (bad > 0 && _cfg.StrictFrameCheck && !_strictHintShown)
            {
                // 收到了字节、同步字也对上了，但一帧都没通过 → 极可能是偏移 15 的固件差异
                _strictHintShown = true;
                var msg = $"血氧：收到数据但 {bad} 帧全部校验失败。疑似固件差异（偏移 15 非 0x00），" +
                          "可在 appsettings.json 中设 Ppg.StrictFrameCheck = false 后重连。";
                _logger.LogWarning("{Msg}", msg);
                StatusChanged?.Invoke(msg);
            }
            else
            {
                _logger.LogWarning("PPG 打开 5 秒仍无有效帧（端口 {Port}）", _port?.PortName);
                StatusChanged?.Invoke("血氧：5 秒无有效数据，请检查端口与接线");
            }
            return;
        }

        // 丢帧判据用「相对自己的基线」而不是协议标称的 125。
        // 实测这块板子稳定跑在 ~87 帧/秒（12 秒 1046 帧，0 坏帧），拿 125 当门限会每秒误报一次。
        double rate = Volatile.Read(ref _frameRate);
        if (rate > _peakFrameRate) _peakFrameRate = rate;

        if (_peakFrameRate >= 20 && rate > 0 && rate < _peakFrameRate * 0.7
            && (now - _lastRateWarnAt).TotalSeconds >= 10)
        {
            _lastRateWarnAt = now;
            StatusChanged?.Invoke($"血氧：帧率下降 {rate:0}/{_peakFrameRate:0} 帧/秒，可能丢帧");
        }
    }

    public void Dispose() => Disconnect();
}
