using Ports = System.IO.Ports;

namespace EEGRecorder.Services;

/// <summary>
/// ADS1299 EEG acquisition board — framed command protocol (verified on the 2026-05 board, COM6).
///
/// Frame (8-byte overhead + N data):
///   [0] Header 0xA5/req 0xAA/rsp | [1..2] Length uint16 LE (=8+N) | [3] Address 0x00
///   [4] Command (bit7 = 写/读) | [5] HdrCheck XOR | [6..] Data | [6+N] DataCheck XOR | [7+N] Tail 0x5A/0x55
///
/// Firmware quirks (verified): always streams 8 interleaved float32/sample, ONLY column 0 carries the
/// electrode signal (cols 1–7 = 0), ignores chMask + the rate request, ~250 Hz/ch. R/W bit polarity is
/// INVERTED vs the protocol doc (write = bit7 CLEAR). DTR/RTS must be LOW or the MCU stays in reset.
/// Startup: STOP → PARAMS → (settle) → START. Raw µV is emitted untouched (this app only records).
/// </summary>
public sealed class Ads1299SerialService : IDisposable
{
    private const byte REQ_HEADER = 0xA5;
    private const byte RSP_HEADER = 0xAA;
    private const byte REQ_TAIL = 0x5A;
    private const byte RSP_TAIL = 0x55;
    private const byte ADDR_BROADCAST = 0x00;
    private const int FRAME_OVERHEAD = 8;
    private const int MAX_FRAME = 4096;

    private const byte CMD_CONN_STATUS = 0x03;
    private const byte CMD_START_STOP = 0x04;
    private const byte CMD_SAMPLE_PARAMS = 0x05;
    private const byte CMD_SAMPLE_DATA = 0x06;
    private const byte WRITE_BIT = 0x80;

    private const byte CONN_DISCONNECT = 0x00;
    private const byte CONN_KEEPALIVE = 0x02;

    private const int STREAM_CHANNELS = 8;   // board always interleaves 8 floats; CH0 is the electrode
    private const int KEEP = 0;

    public int SampleRate { get; private set; } = 250;
    public byte Gain { get; set; } = 12;
    public bool Differential { get; set; } = true;
    public byte ChannelMask { get; set; } = 0x01;

    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(65536);
    private readonly object _bufLock = new();
    private System.Timers.Timer? _keepAlive;

    public long TotalBytes { get; private set; }
    public long DataFrames { get; private set; }
    public long SamplesEmitted { get; private set; }

    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? "";

    /// <summary>A batch of CH0 µV samples decoded from one data frame (native order, lossless).</summary>
    public event Action<float[]>? EegBatch;
    public event Action<string>? Status;

    public bool Connect(string portName, int baudRate, int sampleRate)
    {
        if (IsConnected) Disconnect();
        lock (_bufLock) _buffer.Clear();
        TotalBytes = DataFrames = SamplesEmitted = 0;
        if (sampleRate is 250 or 500 or 1000 or 2000) SampleRate = sampleRate;

        try
        {
            _port = new Ports.SerialPort(portName, baudRate, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadBufferSize = 65536,
                WriteBufferSize = 4096,
                ReadTimeout = 500,
                WriteTimeout = 500,
                DtrEnable = false,   // vendor opens with DTR/RTS LOW; asserting them holds the MCU in reset
                RtsEnable = false,
            };
            _port.DataReceived += OnDataReceived;
            _port.Open();

            SendFrame(CMD_START_STOP, true, new[] { (byte)0x00 });   // stop any prior streaming
            Thread.Sleep(120);
            SendSampleParams();                                       // rate / gain / channel / mode
            Thread.Sleep(250);
            SendFrame(CMD_START_STOP, true, new[] { (byte)0x01 });   // start streaming

            _keepAlive = new System.Timers.Timer(1000) { AutoReset = true };
            _keepAlive.Elapsed += (_, _) => { try { SendFrame(CMD_CONN_STATUS, true, new[] { CONN_KEEPALIVE }); } catch { } };
            _keepAlive.Start();

            Status?.Invoke($"EEG 已连接 {portName}@{baudRate} {SampleRate}Hz");
            return true;
        }
        catch (Exception ex)
        {
            Status?.Invoke($"EEG 打开失败 {portName}: {ex.Message}");
            return false;
        }
    }

    public void Disconnect()
    {
        if (_port == null) return;
        _keepAlive?.Stop(); _keepAlive?.Dispose(); _keepAlive = null;
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
        try { if (_port.IsOpen) _port.Close(); } catch { }
        _port.Dispose();
        _port = null;
        lock (_bufLock) _buffer.Clear();
        Status?.Invoke("EEG 已断开");
    }

    private void SendSampleParams()
    {
        var data = new byte[6];
        data[0] = (byte)(SampleRate & 0xFF);
        data[1] = (byte)((SampleRate >> 8) & 0xFF);
        data[2] = Gain;
        data[3] = ChannelMask;
        data[4] = 0x00;                                  // input switch (normal)
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
        // INVERTED vs doc: write = bit7 CLEAR, read = bit7 SET (matches actual firmware).
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
        var port = _port;
        if (port == null || !port.IsOpen) return;
        try
        {
            int available = port.BytesToRead;
            if (available <= 0) return;
            var buf = new byte[available];
            int read = port.Read(buf, 0, available);
            List<float[]>? batches = null;
            lock (_bufLock)
            {
                TotalBytes += read;
                for (int i = 0; i < read; i++) _buffer.Add(buf[i]);
                batches = ProcessBuffer();
            }
            if (batches != null)
                foreach (var b in batches) EegBatch?.Invoke(b);
        }
        catch (Exception ex) { Status?.Invoke($"EEG 读取错误: {ex.Message}"); }
    }

    private List<float[]>? ProcessBuffer()
    {
        List<float[]>? outBatches = null;
        while (true)
        {
            int hdr = -1;
            for (int i = 0; i < _buffer.Count; i++)
                if (_buffer[i] == RSP_HEADER) { hdr = i; break; }
            if (hdr < 0) { if (_buffer.Count > MAX_FRAME) _buffer.Clear(); break; }
            if (hdr > 0) _buffer.RemoveRange(0, hdr);
            if (_buffer.Count < 3) break;

            int total = _buffer[1] | (_buffer[2] << 8);
            if (total < FRAME_OVERHEAD || total > MAX_FRAME) { _buffer.RemoveAt(0); continue; }
            if (_buffer.Count < total) break;
            if (_buffer[total - 1] != RSP_TAIL) { _buffer.RemoveAt(0); continue; }

            byte cmd = (byte)(_buffer[4] & 0x7F);
            byte hdrChk = (byte)(_buffer[1] ^ _buffer[2] ^ _buffer[3] ^ _buffer[4]);
            bool hdrOk = hdrChk == _buffer[5];
            int n = total - FRAME_OVERHEAD;
            var data = new byte[n];
            byte dataChk = 0;
            for (int i = 0; i < n; i++) { data[i] = _buffer[6 + i]; dataChk ^= data[i]; }
            bool ok = hdrOk && dataChk == _buffer[6 + n];

            _buffer.RemoveRange(0, total);

            if (cmd == CMD_SAMPLE_DATA && ok)
            {
                var batch = Decode(data);
                if (batch.Length > 0) (outBatches ??= new()).Add(batch);
            }
        }
        return outBatches;
    }

    private float[] Decode(byte[] data)
    {
        int nFloats = data.Length / 4;
        int nSamples = nFloats / STREAM_CHANNELS;
        if (nSamples == 0) return Array.Empty<float>();
        var ch0 = new float[nSamples];
        for (int s = 0; s < nSamples; s++)
        {
            float uv = BitConverter.ToSingle(data, (s * STREAM_CHANNELS + KEEP) * 4);
            if (float.IsNaN(uv) || float.IsInfinity(uv)) uv = 0f;
            ch0[s] = uv;
        }
        DataFrames++;
        SamplesEmitted += nSamples;
        return ch0;
    }

    public void Dispose() => Disconnect();
}
