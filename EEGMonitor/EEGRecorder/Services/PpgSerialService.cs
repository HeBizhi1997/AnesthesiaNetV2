using Ports = System.IO.Ports;

namespace EEGRecorder.Services;

/// <summary>One PPG sample as pushed by the finger-clip module.</summary>
public readonly record struct PpgFrame(long Ticks, int Ir, int Red, byte Spo2, byte Hr);

/// <summary>
/// PPG / 血氧 指夹模组 (CH340, 默认 COM7, 57600 8N1, 被动推送 ~125 Hz).
///
/// 帧 (17 字节): 0A FA | 0A 00 02 | IR(int32 LE) | RED(int32 LE) | SpO2 | HR | Status | 0B
/// Opens and streams with no handshake (DTR/RTS LOW, matching the vendor capture). This app records
/// the raw IR/RED + device SpO2/HR losslessly — no derived metrics.
/// </summary>
public sealed class PpgSerialService : IDisposable
{
    private const int FRAME = 17;
    private const byte H0 = 0x0A, H1 = 0xFA, TAIL = 0x0B;

    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(2048);
    private readonly object _lock = new();

    public long FramesIn { get; private set; }
    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? "";

    public event Action<IReadOnlyList<PpgFrame>>? PpgBatch;
    public event Action<string>? Status;

    public bool Connect(string portName, int baud = 57600)
    {
        if (IsConnected) Disconnect();
        lock (_lock) { _buffer.Clear(); FramesIn = 0; }
        try
        {
            _port = new Ports.SerialPort(portName, baud, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadTimeout = 500,
                WriteTimeout = 500,
                ReadBufferSize = 16384,
                DtrEnable = false,
                RtsEnable = false,
            };
            _port.DataReceived += OnDataReceived;
            _port.Open();
            _port.DiscardInBuffer();
            Status?.Invoke($"PPG 已连接 {portName}@{baud}");
            return true;
        }
        catch (Exception ex)
        {
            Status?.Invoke($"PPG 打开失败 {portName}: {ex.Message}");
            return false;
        }
    }

    public void Disconnect()
    {
        if (_port == null) return;
        _port.DataReceived -= OnDataReceived;
        try { if (_port.IsOpen) _port.Close(); } catch { }
        _port.Dispose();
        _port = null;
        lock (_lock) _buffer.Clear();
        Status?.Invoke("PPG 已断开");
    }

    private void OnDataReceived(object sender, Ports.SerialDataReceivedEventArgs e)
    {
        var port = _port;
        if (port == null || !port.IsOpen) return;
        try
        {
            int avail = port.BytesToRead;
            if (avail <= 0) return;
            var buf = new byte[avail];
            int read = port.Read(buf, 0, avail);
            List<PpgFrame>? frames = null;
            lock (_lock)
            {
                for (int i = 0; i < read; i++) _buffer.Add(buf[i]);
                frames = ProcessBuffer();
            }
            if (frames is { Count: > 0 }) PpgBatch?.Invoke(frames);
        }
        catch (Exception ex) { Status?.Invoke($"PPG 读取错误: {ex.Message}"); }
    }

    private List<PpgFrame>? ProcessBuffer()
    {
        List<PpgFrame>? frames = null;
        int i = 0, n = _buffer.Count;
        long now = DateTime.Now.Ticks;
        while (i + FRAME <= n)
        {
            if (_buffer[i] == H0 && _buffer[i + 1] == H1 && _buffer[i + 16] == TAIL)
            {
                int ir  = _buffer[i + 5] | (_buffer[i + 6] << 8) | (_buffer[i + 7] << 16) | (_buffer[i + 8] << 24);
                int red = _buffer[i + 9] | (_buffer[i + 10] << 8) | (_buffer[i + 11] << 16) | (_buffer[i + 12] << 24);
                byte spo2 = _buffer[i + 13];
                byte hr = _buffer[i + 14];
                (frames ??= new()).Add(new PpgFrame(now, ir, red, spo2, hr));
                FramesIn++;
                i += FRAME;
            }
            else i++;
        }
        if (i > 0) _buffer.RemoveRange(0, i);
        if (_buffer.Count > 4096) _buffer.Clear();
        return frames;
    }

    public void Dispose() => Disconnect();
}
