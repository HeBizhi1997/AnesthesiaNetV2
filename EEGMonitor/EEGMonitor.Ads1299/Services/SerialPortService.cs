using Ads1299Monitor.Models;
using Microsoft.Extensions.Logging;
using Ports = System.IO.Ports;

namespace Ads1299Monitor.Services;

/// <summary>
/// ADS1299 EEG acquisition board — 2026-05 framed command protocol.
///
/// Frame layout (8-byte overhead + N data):
///   [0]        Header      请求 0xA5 / 响应 0xAA
///   [1..2]     Length      uint16 LE = total frame length (= 8 + N)
///   [3]        Address     0x00 广播
///   [4]        Command     bit7 = 写(1)/读(0)，bit6:0 = 命令码
///   [5]        HdrCheck    XOR(Length[0], Length[1], Address, Command)
///   [6..]      Data        变长
///   [6+N]      DataCheck   XOR(Data)
///   [7+N]      Tail        请求 0x5A / 响应 0x55
///
/// VERIFIED on the 2026-05 board (COM6): the firmware ALWAYS streams 8 floats/sample
/// interleaved and IGNORES both chMask and the rate request — only column 0 (CH1) carries
/// the differential electrode signal; columns 1–7 are exactly 0.0, and the per-channel rate
/// is ~250 Hz. We de-interleave and keep CH0. Python owns all filtering.
/// </summary>
public sealed class SerialPortService : IDisposable
{
    private const byte REQ_HEADER = 0xA5;
    private const byte RSP_HEADER = 0xAA;
    private const byte REQ_TAIL = 0x5A;
    private const byte RSP_TAIL = 0x55;
    private const byte ADDR_BROADCAST = 0x00;
    private const int FRAME_OVERHEAD = 8;
    private const int MAX_FRAME = 4096;

    private const byte CMD_FW_VERSION = 0x00;
    private const byte CMD_HW_VERSION = 0x01;
    private const byte CMD_CONN_STATUS = 0x03;
    private const byte CMD_START_STOP = 0x04;
    private const byte CMD_SAMPLE_PARAMS = 0x05;
    private const byte CMD_SAMPLE_DATA = 0x06;
    private const byte WRITE_BIT = 0x80;

    private const byte CONN_DISCONNECT = 0x00;
    private const byte CONN_CONNECTED = 0x01;
    private const byte CONN_KEEPALIVE = 0x02;

    // Board always streams 8 interleaved floats/sample; CH0 carries the electrode.
    private const int STREAM_CHANNELS = 8;

    public int SampleRate { get; set; } = 250;
    public byte Gain { get; set; } = 12;
    public bool Differential { get; set; } = true;
    public byte InputSwitch { get; set; } = 0x00;
    public byte ChannelMask { get; set; } = 0x01;   // requested; board ignores it
    public int KeepChannelIndex { get; set; } = 0;

    private readonly ILogger<SerialPortService> _logger;
    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(65536);
    private readonly object _bufLock = new();
    private System.Timers.Timer? _keepAliveTimer;

    private long _totalBytesIn;
    private long _framesDecoded;
    private long _dataFramesDecoded;
    private long _samplesEmitted;
    private bool _loggedFirstBytes;
    private readonly int _streamChannels = STREAM_CHANNELS;

    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? string.Empty;

    public event Action<EEGSample>? SampleReceived;
    public event Action<string>? ConnectionStatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public SerialPortService(ILogger<SerialPortService> logger) => _logger = logger;

    public bool Connect(string portName, int baudRate, int sampleRate)
    {
        if (IsConnected) Disconnect();
        lock (_bufLock) _buffer.Clear();
        _totalBytesIn = _framesDecoded = _dataFramesDecoded = _samplesEmitted = 0;
        _loggedFirstBytes = false;
        if (sampleRate is 250 or 500 or 1000 or 2000) SampleRate = sampleRate;

        try
        {
            _port = new Ports.SerialPort(portName, baudRate, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadBufferSize = 65536,
                WriteBufferSize = 4096,
                ReadTimeout = 500,
                WriteTimeout = 500,
                // Vendor LK-M1299 opens with DTR/RTS LOW. Asserting them (true) can hold the
                // board MCU in reset on this hardware → no stream. Match the vendor: both false.
                DtrEnable = false,
                RtsEnable = false,
            };
            _port.DataReceived += OnDataReceived;
            _port.ErrorReceived += OnErrorReceived;
            _port.Open();

            // Vendor start sequence (decompiled): STOP → PARAMS → (settle) → START.
            SendFrame(CMD_START_STOP, true, new[] { (byte)0x00 });    // stop any prior streaming
            System.Threading.Thread.Sleep(120);
            SendSampleParams();                                        // configure rate/gain/ch/mode
            System.Threading.Thread.Sleep(250);                        // let the device apply + ACK
            SendFrame(CMD_START_STOP, true, new[] { (byte)0x01 });    // start streaming

            _keepAliveTimer = new System.Timers.Timer(1000) { AutoReset = true };
            _keepAliveTimer.Elapsed += (_, _) =>
            {
                try { SendFrame(CMD_CONN_STATUS, true, new[] { CONN_KEEPALIVE }); } catch { }
                // Link diagnostic: shows whether ANY bytes/frames are arriving from the port.
                _logger.LogInformation("ADS1299 link: {Bytes}B in · {Frames} frames · {Data} data frames",
                    _totalBytesIn, _framesDecoded, _dataFramesDecoded);
            };
            _keepAliveTimer.Start();

            _logger.LogInformation("ADS1299 {Port}@{Baud} opened — rate={Rate}Hz gain={Gain} {Mode}",
                portName, baudRate, SampleRate, Gain, Differential ? "差分" : "单端");
            ConnectionStatusChanged?.Invoke($"已连接 {portName}@{baudRate} {SampleRate}Hz");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to open ADS1299 port {Port}", portName);
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
        try
        {
            if (_port.IsOpen)
            {
                SendFrame(CMD_START_STOP, true, new[] { (byte)0x00 });
                SendFrame(CMD_CONN_STATUS, true, new[] { CONN_DISCONNECT });
            }
        }
        catch { }
        _port.DataReceived -= OnDataReceived;
        _port.ErrorReceived -= OnErrorReceived;
        if (_port.IsOpen) _port.Close();
        _port.Dispose();
        _port = null;
        lock (_bufLock) _buffer.Clear();
        ConnectionStatusChanged?.Invoke("已断开");
    }

    private void SendSampleParams()
    {
        var data = new byte[6];
        data[0] = (byte)(SampleRate & 0xFF);
        data[1] = (byte)((SampleRate >> 8) & 0xFF);
        data[2] = Gain;
        data[3] = ChannelMask;
        data[4] = InputSwitch;
        data[5] = (byte)(Differential ? 0x01 : 0x00);
        SendFrame(CMD_SAMPLE_PARAMS, true, data);
    }

    private void SendFrame(byte cmd, bool write, byte[]? data)
    {
        if (_port is not { IsOpen: true }) return;
        var frame = BuildFrame(cmd, write, data);
        lock (_bufLock) _port.Write(frame, 0, frame.Length);
    }

    private static byte[] BuildFrame(byte cmd, bool write, byte[]? data)
    {
        int n = data?.Length ?? 0;
        int total = FRAME_OVERHEAD + n;
        var f = new byte[total];
        f[0] = REQ_HEADER;
        f[1] = (byte)(total & 0xFF);
        f[2] = (byte)((total >> 8) & 0xFF);
        f[3] = ADDR_BROADCAST;
        // R/W bit polarity matches the ACTUAL firmware (decompiled from vendor LK-M1299):
        // WRITE = command with bit7 CLEAR, READ = command with bit7 SET. This is the OPPOSITE
        // of what the protocol .docx states. Using the doc's polarity made every START/PARAMS
        // go out as a READ → the board never started streaming from a cold state.
        f[4] = (byte)(write ? (cmd & 0x7F) : (cmd | WRITE_BIT));
        f[5] = (byte)(f[1] ^ f[2] ^ f[3] ^ f[4]);
        byte dc = 0;
        for (int i = 0; i < n; i++) { f[6 + i] = data![i]; dc ^= data[i]; }
        f[6 + n] = dc;
        f[7 + n] = REQ_TAIL;
        return f;
    }

    private void OnDataReceived(object sender, Ports.SerialDataReceivedEventArgs e)
    {
        if (_port == null || !_port.IsOpen) return;
        try
        {
            int available = _port.BytesToRead;
            if (available <= 0) return;
            var buf = new byte[available];
            int read = _port.Read(buf, 0, available);
            if (!_loggedFirstBytes && read > 0)
            {
                _loggedFirstBytes = true;
                var hex = BitConverter.ToString(buf, 0, Math.Min(read, 24));
                _logger.LogInformation("ADS1299 first bytes ({Read}B): {Hex}", read, hex);
            }
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

    private void OnErrorReceived(object sender, Ports.SerialErrorReceivedEventArgs e) =>
        _logger.LogWarning("ADS1299 serial error: {Error}", e.EventType);

    private void ProcessBuffer()
    {
        while (true)
        {
            int hdr = -1;
            for (int i = 0; i < _buffer.Count; i++)
                if (_buffer[i] == RSP_HEADER) { hdr = i; break; }
            if (hdr < 0)
            {
                if (_buffer.Count > MAX_FRAME) _buffer.Clear();
                return;
            }
            if (hdr > 0) _buffer.RemoveRange(0, hdr);
            if (_buffer.Count < 3) return;

            int total = _buffer[1] | (_buffer[2] << 8);
            if (total < FRAME_OVERHEAD || total > MAX_FRAME) { _buffer.RemoveAt(0); continue; }
            if (_buffer.Count < total) return;
            if (_buffer[total - 1] != RSP_TAIL) { _buffer.RemoveAt(0); continue; }

            byte cmd = _buffer[4];
            byte hdrChk = (byte)(_buffer[1] ^ _buffer[2] ^ _buffer[3] ^ _buffer[4]);
            bool hdrOk = hdrChk == _buffer[5];
            int n = total - FRAME_OVERHEAD;
            var data = new byte[n];
            byte dataChk = 0;
            for (int i = 0; i < n; i++) { data[i] = _buffer[6 + i]; dataChk ^= data[i]; }
            bool dataOk = dataChk == _buffer[6 + n];

            _buffer.RemoveRange(0, total);
            _framesDecoded++;
            try { DispatchFrame((byte)(cmd & 0x7F), data, hdrOk && dataOk); }
            catch (Exception ex) { _logger.LogWarning(ex, "Error handling frame 0x{Cmd:X2}", cmd); }
        }
    }

    private void DispatchFrame(byte cmd, byte[] data, bool checksumOk)
    {
        switch (cmd)
        {
            case CMD_SAMPLE_DATA:
                _dataFramesDecoded++;
                EmitSamples(data, checksumOk);
                if (_dataFramesDecoded == 1 || (_dataFramesDecoded % 16) == 0)
                    ConnectionStatusChanged?.Invoke(
                        $"接收中 {PortName} {_dataFramesDecoded}帧 {_totalBytesIn / 1024}KB");
                break;
            case CMD_FW_VERSION:
            case CMD_HW_VERSION:
                _logger.LogInformation("ADS1299 {Kind}: {Ver}",
                    cmd == CMD_FW_VERSION ? "fw" : "hw",
                    System.Text.Encoding.ASCII.GetString(data).TrimEnd('\0'));
                break;
            default:
                break;
        }
    }

    private void EmitSamples(byte[] data, bool checksumOk)
    {
        if (!checksumOk) return;
        int nFloats = data.Length / 4;
        int nCh = Math.Max(1, _streamChannels);
        int keep = Math.Clamp(KeepChannelIndex, 0, nCh - 1);
        int nSamples = nFloats / nCh;
        if (nSamples == 0) return;

        var now = DateTime.Now;
        double periodMs = 1000.0 / Math.Max(1, SampleRate);
        var baseTime = now.AddMilliseconds(-(nSamples - 1) * periodMs);

        for (int s = 0; s < nSamples; s++)
        {
            float uv = BitConverter.ToSingle(data, (s * nCh + keep) * 4);
            if (float.IsNaN(uv) || float.IsInfinity(uv)) uv = 0f;
            var ts = baseTime.AddMilliseconds(s * periodMs);
            SampleReceived?.Invoke(new EEGSample(ts, new[] { (double)uv }));
        }
        _samplesEmitted += nSamples;
    }

    public void Dispose() => Disconnect();
}
