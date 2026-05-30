using EEGMonitor.Models;
using Microsoft.Extensions.Logging;
using Ports = System.IO.Ports;

namespace EEGMonitor.Services.SerialPort;

/// <summary>
/// ADS1299 EEG acquisition board — 2026-05 framed command protocol
/// (doc/EEG-ads1299-通信协议 20260529.docx).
///
/// Frame layout (8-byte overhead + N data):
///   [0]        Header      请求 0xA5 / 响应 0xAA
///   [1..2]     Length      uint16 LE = total frame length (= 8 + N)
///   [3]        Address     0x00 广播 / 0x01-0xFE 设备 / 0xFF 主机
///   [4]        Command     bit7 = 写(1)/读(0)，bit6:0 = 命令码 0x00-0x7F
///   [5]        HdrCheck    XOR(Length[0], Length[1], Address, Command)
///   [6..6+N-1] Data        变长（命令决定）
///   [6+N]      DataCheck   XOR(所有 Data 字节)（N=0 时为 0x00）
///   [7+N]      Tail        请求 0x5A / 响应 0x55
///
/// Commands:
///   0x00 R   固件版本    0x01 R   硬件版本    0x02 W 重启
///   0x03 R/W 连接状态(0=断开 1=已连接 2=保活)
///   0x04 R/W 开始/停止(0=停止 1=开始)
///   0x05 R/W 采样参数 [rate:u16][gain][chMask][inputSwitch][single(0)/diff(1)]
///   0x06     采样数据    float32 µV × (ch × n)（本板单通道 → n 个 float）
///
/// 本板为单通道、默认差分模式。数据已是 µV（float32），WPF 不再做滤波，
/// 滤波由 Python 服务统一负责（与训练预处理一致，避免双重滤波）。
///
/// 默认参数（协议文档）：波特率 230400，采样率 500，放大倍数 12。
/// </summary>
public sealed class SerialPortService : ISerialPortService
{
    // ── Frame constants ────────────────────────────────────────────────────────
    private const byte REQ_HEADER = 0xA5;   // host → device
    private const byte RSP_HEADER = 0xAA;   // device → host
    private const byte REQ_TAIL   = 0x5A;
    private const byte RSP_TAIL   = 0x55;
    private const byte ADDR_BROADCAST = 0x00;
    private const int  FRAME_OVERHEAD = 8;  // header+len(2)+addr+cmd+hdrChk+dataChk+tail
    private const int  MAX_FRAME = 4096;    // sanity bound for a single data packet

    // Command codes (low 7 bits)
    private const byte CMD_FW_VERSION    = 0x00;
    private const byte CMD_HW_VERSION    = 0x01;
    private const byte CMD_REBOOT        = 0x02;
    private const byte CMD_CONN_STATUS   = 0x03;
    private const byte CMD_START_STOP    = 0x04;
    private const byte CMD_SAMPLE_PARAMS = 0x05;
    private const byte CMD_SAMPLE_DATA   = 0x06;
    private const byte WRITE_BIT = 0x80;

    // Connection-status data values
    private const byte CONN_DISCONNECT = 0x00;
    private const byte CONN_CONNECTED  = 0x01;
    private const byte CONN_KEEPALIVE  = 0x02;

    private const int DEFAULT_BAUD = 230400;

    // ── Acquisition configuration (set before Connect; sensible defaults) ──────
    /// <summary>
    /// 有效每通道采样率(Hz)。2026-05 板实测固定 ~250 Hz/通道（忽略 rate 请求，ACK 恒回 500）。
    /// 此值用于 pipeline 重采样与时间戳回填，必须等于设备真实速率，默认 250。
    /// </summary>
    public int SampleRate { get; set; } = 250;
    /// <summary>放大倍数 1/2/4/6/8/12/24（默认 12）。</summary>
    public byte Gain { get; set; } = 12;
    /// <summary>true=差分模式(0x01)，false=单端(0x00)。本板默认差分。</summary>
    public bool Differential { get; set; } = true;
    /// <summary>通道输入开关 0x00 电极 / 0x01 内部短路 / 0x02 测试信号。</summary>
    public byte InputSwitch { get; set; } = 0x00;
    /// <summary>通道使能掩码 bit7:0 = CH8:1。单通道板默认仅 CH1 (0x01)。</summary>
    public byte ChannelMask { get; set; } = 0x01;

    private readonly ILogger<SerialPortService> _logger;
    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(65536);
    private readonly object _bufLock = new();
    private System.Timers.Timer? _keepAliveTimer;

    private long _totalBytesIn;
    private long _framesDecoded;
    private long _dataFramesDecoded;
    private long _samplesEmitted;

    // Channels the device actually streams. VERIFIED on the 2026-05 board (COM6):
    // the firmware ALWAYS streams 8 floats/sample interleaved and IGNORES both the
    // chMask and the sample-rate request — only column 0 (CH1) carries the differential
    // electrode signal; columns 1–7 are exactly 0.0. So this is FIXED at 8 and must NOT
    // be derived from the params-ACK chMask (echoing 0x01 there reset it to 1, which made
    // the de-interleave treat all 8 floats as samples → 7-zero-stuffing → 50 Hz mains
    // folded to a bogus ~31 Hz artifact instead of clean EEG).
    private const int STREAM_CHANNELS = 8;
    private int _streamChannels = STREAM_CHANNELS;
    /// <summary>Index (within the interleaved stream) of the electrode channel to keep. CH1 = 0.</summary>
    public int KeepChannelIndex { get; set; } = 0;

    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? string.Empty;
    public int BaudRate => _port?.BaudRate ?? 0;
    public long TotalBytesIn => _totalBytesIn;
    public long FramesDecoded => _framesDecoded;

    public event Action<EEGSample>? SampleReceived;
    public event Action<string>? ConnectionStatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public SerialPortService(ILogger<SerialPortService> logger)
    {
        _logger = logger;
    }

    public IEnumerable<string> GetAvailablePorts() =>
        Ports.SerialPort.GetPortNames().OrderBy(p => p);

    /// <param name="channelCount">忽略 — 本板固定单通道（CH1）。保留以兼容接口。</param>
    /// <param name="sampleRate">若为受支持值(250/500/1000/2000)则采用，否则用默认 500。</param>
    public bool Connect(string portName, int baudRate = DEFAULT_BAUD, int channelCount = 1, int sampleRate = 500)
    {
        if (IsConnected) Disconnect();

        lock (_bufLock) _buffer.Clear();
        _totalBytesIn = 0;
        _framesDecoded = 0;
        _dataFramesDecoded = 0;
        _samplesEmitted = 0;
        _streamChannels = STREAM_CHANNELS;   // board always streams 8 interleaved (CH0 = electrode)
        if (sampleRate is 250 or 500 or 1000 or 2000) SampleRate = sampleRate;

        try
        {
            _port = new Ports.SerialPort(portName, baudRate, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadBufferSize = 65536,
                WriteBufferSize = 4096,
                ReadTimeout = 500,
                WriteTimeout = 500,
                DtrEnable = true,
                RtsEnable = true,
            };
            _port.DataReceived += OnDataReceived;
            _port.ErrorReceived += OnErrorReceived;
            _port.Open();

            // Handshake: announce connection → push acquisition params → start sampling.
            SendFrame(CMD_CONN_STATUS, write: true, data: new[] { CONN_CONNECTED });
            SendSampleParams();
            SendFrame(CMD_START_STOP, write: true, data: new[] { (byte)0x01 });

            // 1 Hz keep-alive (protocol §2.2.3: 定时1秒发送一次)
            _keepAliveTimer = new System.Timers.Timer(1000) { AutoReset = true };
            _keepAliveTimer.Elapsed += (_, _) =>
            {
                try { SendFrame(CMD_CONN_STATUS, write: true, data: new[] { CONN_KEEPALIVE }); }
                catch { /* port may be closing */ }
            };
            _keepAliveTimer.Start();

            _logger.LogInformation(
                "ADS1299 port {Port}@{Baud} opened — rate={Rate}Hz gain={Gain} {Mode} chMask=0x{Mask:X2}",
                portName, baudRate, SampleRate, Gain, Differential ? "差分" : "单端", ChannelMask);
            ConnectionStatusChanged?.Invoke($"Connected: {portName}@{baudRate}  {SampleRate}Hz {(Differential ? "差分" : "单端")}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to open ADS1299 serial port {Port}", portName);
            ErrorOccurred?.Invoke(ex);
            return false;
        }
    }

    public void Disconnect()
    {
        if (_port == null) return;

        _keepAliveTimer?.Stop();
        _keepAliveTimer?.Dispose();
        _keepAliveTimer = null;

        // Best-effort graceful stop: stop sampling, announce disconnect.
        try
        {
            if (_port.IsOpen)
            {
                SendFrame(CMD_START_STOP, write: true, data: new[] { (byte)0x00 });
                SendFrame(CMD_CONN_STATUS, write: true, data: new[] { CONN_DISCONNECT });
            }
        }
        catch { /* ignore — closing anyway */ }

        _port.DataReceived -= OnDataReceived;
        _port.ErrorReceived -= OnErrorReceived;
        if (_port.IsOpen) _port.Close();
        _port.Dispose();
        _port = null;
        lock (_bufLock) _buffer.Clear();
        _logger.LogInformation("ADS1299 serial port disconnected");
        ConnectionStatusChanged?.Invoke("Disconnected");
    }

    // ── TX ─────────────────────────────────────────────────────────────────────

    private void SendSampleParams()
    {
        var data = new byte[6];
        data[0] = (byte)(SampleRate & 0xFF);          // rate LE
        data[1] = (byte)((SampleRate >> 8) & 0xFF);
        data[2] = Gain;
        data[3] = ChannelMask;                         // bit7:0 = CH8:1
        data[4] = InputSwitch;                         // 0=电极 1=短路 2=测试
        data[5] = (byte)(Differential ? 0x01 : 0x00);  // 0=单端 1=差分
        SendFrame(CMD_SAMPLE_PARAMS, write: true, data: data);
    }

    private void SendFrame(byte cmd, bool write, byte[]? data)
    {
        if (_port is not { IsOpen: true }) return;
        var frame = BuildFrame(cmd, write, data);
        lock (_bufLock)   // serialise writes (keep-alive timer vs connect thread)
        {
            _port.Write(frame, 0, frame.Length);
        }
    }

    /// <summary>Build a request frame (header 0xA5, tail 0x5A) with XOR checks.</summary>
    private static byte[] BuildFrame(byte cmd, bool write, byte[]? data)
    {
        int n = data?.Length ?? 0;
        int total = FRAME_OVERHEAD + n;
        var f = new byte[total];
        f[0] = REQ_HEADER;
        f[1] = (byte)(total & 0xFF);
        f[2] = (byte)((total >> 8) & 0xFF);
        f[3] = ADDR_BROADCAST;
        f[4] = (byte)(write ? (cmd | WRITE_BIT) : cmd);
        f[5] = (byte)(f[1] ^ f[2] ^ f[3] ^ f[4]);   // header check
        byte dataCheck = 0;
        for (int i = 0; i < n; i++)
        {
            f[6 + i] = data![i];
            dataCheck ^= data[i];
        }
        f[6 + n] = dataCheck;
        f[7 + n] = REQ_TAIL;
        return f;
    }

    // ── RX ─────────────────────────────────────────────────────────────────────

    private void OnDataReceived(object sender, Ports.SerialDataReceivedEventArgs e)
    {
        if (_port == null || !_port.IsOpen) return;
        try
        {
            int available = _port.BytesToRead;
            if (available <= 0) return;
            var buf = new byte[available];
            int read = _port.Read(buf, 0, available);

            lock (_bufLock)
            {
                _totalBytesIn += read;
                for (int i = 0; i < read; i++) _buffer.Add(buf[i]);
                ProcessBuffer();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error reading ADS1299 serial data");
            ErrorOccurred?.Invoke(ex);
        }
    }

    private void OnErrorReceived(object sender, Ports.SerialErrorReceivedEventArgs e)
    {
        _logger.LogWarning("ADS1299 serial port error: {Error}", e.EventType);
    }

    private void ProcessBuffer()
    {
        while (true)
        {
            // 1. Sync to a response header 0xAA.
            int hdr = -1;
            for (int i = 0; i < _buffer.Count; i++)
                if (_buffer[i] == RSP_HEADER) { hdr = i; break; }

            if (hdr < 0)
            {
                if (_buffer.Count > MAX_FRAME) _buffer.Clear();   // no header in a large buffer → junk
                return;
            }
            if (hdr > 0) _buffer.RemoveRange(0, hdr);

            // 2. Need length field.
            if (_buffer.Count < 3) return;
            int total = _buffer[1] | (_buffer[2] << 8);
            if (total < FRAME_OVERHEAD || total > MAX_FRAME)
            {
                _buffer.RemoveAt(0);   // implausible length → false header
                continue;
            }

            // 3. Wait for the full frame.
            if (_buffer.Count < total) return;

            // 4. Validate tail; if wrong this wasn't a real frame boundary.
            if (_buffer[total - 1] != RSP_TAIL)
            {
                _buffer.RemoveAt(0);
                continue;
            }

            byte cmd = _buffer[4];
            byte hdrChk = (byte)(_buffer[1] ^ _buffer[2] ^ _buffer[3] ^ _buffer[4]);
            bool hdrOk = hdrChk == _buffer[5];   // doc's example values are inconsistent → tolerate

            int n = total - FRAME_OVERHEAD;
            var data = new byte[n];
            byte dataChk = 0;
            for (int i = 0; i < n; i++) { data[i] = _buffer[6 + i]; dataChk ^= data[i]; }
            bool dataOk = dataChk == _buffer[6 + n];

            _buffer.RemoveRange(0, total);
            _framesDecoded++;

            try
            {
                DispatchFrame((byte)(cmd & 0x7F), data, hdrOk && dataOk);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error handling ADS1299 frame cmd=0x{Cmd:X2}", cmd);
            }
        }
    }

    private void DispatchFrame(byte cmd, byte[] data, bool checksumOk)
    {
        switch (cmd)
        {
            case CMD_SAMPLE_DATA:
                _dataFramesDecoded++;
                EmitSamples(data, checksumOk);
                if (_dataFramesDecoded == 1)
                    _logger.LogInformation(
                        "ADS1299 first data frame decoded: {N} bytes, {Ch} stream channels, keep CH{Keep}, chkOk={Ok}",
                        data.Length, _streamChannels, KeepChannelIndex, checksumOk);
                else if (_dataFramesDecoded % 200 == 0)
                    _logger.LogInformation("ADS1299 streaming: {Frames} data frames, {Samples} samples emitted",
                        _dataFramesDecoded, _samplesEmitted);
                if (_dataFramesDecoded == 1 || (_dataFramesDecoded % 16) == 0)
                    ConnectionStatusChanged?.Invoke(
                        $"接收中: {PortName}  {_dataFramesDecoded} 帧  {_totalBytesIn / 1024} KB");
                break;

            case CMD_FW_VERSION:
            case CMD_HW_VERSION:
                _logger.LogInformation("ADS1299 {Kind}: {Ver}",
                    cmd == CMD_FW_VERSION ? "firmware" : "hardware",
                    System.Text.Encoding.ASCII.GetString(data).TrimEnd('\0'));
                break;

            case CMD_SAMPLE_PARAMS:
                // ACK echoes the config the device claims to apply:
                // [rate_lo, rate_hi, gain, chMask, inputSwitch, single/diff].
                // NOTE: this firmware lies — it ignores both chMask and rate and always
                // streams 8 floats/sample @ ~250 Hz/ch (verified on COM6). So we do NOT
                // derive _streamChannels from the ACK chMask (that reset it to 1 and broke
                // the de-interleave). _streamChannels stays pinned at STREAM_CHANNELS=8.
                if (data.Length >= 4)
                {
                    int ackCh = System.Numerics.BitOperations.PopCount((uint)data[3]);
                    int appliedRate = data.Length >= 2 ? (data[0] | (data[1] << 8)) : SampleRate;
                    _logger.LogInformation(
                        "ADS1299 params ACK: rate={Rate} gain={Gain} chMask=0x{Mask:X2} (ACK says {Ack}ch; " +
                        "using fixed {Use}ch — board always streams 8) diff={Diff}",
                        appliedRate, data.Length >= 3 ? data[2] : 0, data[3], ackCh, _streamChannels,
                        data.Length >= 6 ? data[5] : 0);
                }
                break;

            case CMD_CONN_STATUS:
            case CMD_START_STOP:
                // Device acknowledgements — nothing to do.
                break;

            default:
                _logger.LogDebug("ADS1299 unhandled frame cmd=0x{Cmd:X2} ({N} bytes)", cmd, data.Length);
                break;
        }
    }

    /// <summary>
    /// Parse a 0x86 data frame: float32 little-endian µV samples, channel-interleaved
    /// as [ch0_s0, ch1_s0, … chK_s0, ch0_s1, …] where K = _streamChannels (from params ACK).
    /// We de-interleave and emit ONE electrode channel (KeepChannelIndex, default CH1=0) as a
    /// single-channel <see cref="EEGSample"/>. The Python service owns all filtering.
    /// </summary>
    private void EmitSamples(byte[] data, bool checksumOk)
    {
        if (!checksumOk)
        {
            _logger.LogDebug("ADS1299 data frame failed checksum — dropped ({N} bytes)", data.Length);
            return;
        }
        int nFloats = data.Length / 4;
        int nCh = Math.Max(1, _streamChannels);
        int keep = Math.Clamp(KeepChannelIndex, 0, nCh - 1);
        int nSamples = nFloats / nCh;   // samples per channel in this frame
        if (nSamples == 0) return;

        var now = DateTime.Now;
        double periodMs = 1000.0 / Math.Max(1, SampleRate);
        var baseTime = now.AddMilliseconds(-(nSamples - 1) * periodMs);

        for (int s = 0; s < nSamples; s++)
        {
            float uv = BitConverter.ToSingle(data, (s * nCh + keep) * 4);
            if (float.IsNaN(uv) || float.IsInfinity(uv)) uv = 0f;
            var ts = baseTime.AddMilliseconds(s * periodMs);
            SampleReceived?.Invoke(new EEGSample(ts, new double[] { uv }));
        }
        _samplesEmitted += nSamples;
    }

    public void Dispose() => Disconnect();
}
