using EEGMonitor.Models;
using Microsoft.Extensions.Logging;
using Ports = System.IO.Ports;

namespace EEGMonitor.Services.SerialPort;

/// <summary>
/// Reads raw EEG from the ADS1299-based device using the 516-byte binary frame protocol:
///   [0]      Header   0x2A
///   [1]      PktNum   0x01..0x04 (cyclic)
///   [2..257] CH1      64 × float32 LE
///   [258..513] CH2    64 × float32 LE
///   [514..515] Tail   0x40 0x40
///
/// ADC conversion: µV = raw_float32 × 4.5 / (24 × 8388607) × 1e6 ≈ raw × 0.02235
/// Byte order: BitConverter.ToSingle assumes system-native (Windows/.NET = LittleEndian).
///   ADS1299 default output is LE; if firmware is reconfigured to BE, swap bytes first.
/// Filter chain per channel: 50 Hz notch (Q=30) → 0.5 Hz HPF → 95 Hz LPF
/// </summary>
public sealed class SerialPortService : ISerialPortService
{
    private const int FRAME_SIZE = 516;
    private const byte FRAME_HEADER = 0x2A;
    private const byte FRAME_TAIL = 0x40;
    private const double UV_SCALE = 4.5 / (24.0 * 8_388_607.0) * 1_000_000.0; // ≈ 0.02235 µV/LSB
    private const int SAMPLE_RATE = 256;
    private const int SAMPLES_PER_PACKET = 64;
    private const int CHANNEL_COUNT = 2;

    private readonly ILogger<SerialPortService> _logger;
    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(65536);
    private readonly byte[] _frameTemp = new byte[FRAME_SIZE];
    private readonly object _bufLock = new();

    // [channel, filterStage]: 0=notch, 1=hpf, 2=lpf
    private readonly BiquadFilter[,] _filters = new BiquadFilter[CHANNEL_COUNT, 3];

    // Diagnostics counters (visible in status)
    private long _totalBytesIn;
    private long _framesDecoded;

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
        InitFilters();
    }

    private void InitFilters()
    {
        for (int ch = 0; ch < CHANNEL_COUNT; ch++)
        {
            _filters[ch, 0] = BiquadFilter.Notch(50.0, SAMPLE_RATE, q: 30.0);
            _filters[ch, 1] = BiquadFilter.Highpass(0.5, SAMPLE_RATE);
            _filters[ch, 2] = BiquadFilter.Lowpass(95.0, SAMPLE_RATE);
        }
    }

    public IEnumerable<string> GetAvailablePorts() =>
        Ports.SerialPort.GetPortNames().OrderBy(p => p);

    public bool Connect(string portName, int baudRate = 115200, int channelCount = 2, int sampleRate = 256)
    {
        if (IsConnected) Disconnect();

        _buffer.Clear();
        ResetFilters();

        _totalBytesIn = 0;
        _framesDecoded = 0;

        try
        {
            _port = new Ports.SerialPort(portName, baudRate, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadBufferSize = 65536,
                ReadTimeout = 500,
                WriteTimeout = 500,
                DtrEnable = true,   // Required by most ADS1299 USB-UART boards
                RtsEnable = true,
            };
            _port.DataReceived += OnDataReceived;
            _port.ErrorReceived += OnErrorReceived;
            _port.Open();

            _logger.LogInformation("EEG serial port {Port} opened at {Baud} baud", portName, baudRate);
            ConnectionStatusChanged?.Invoke($"Connected: {portName}@{baudRate}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to open EEG serial port {Port}", portName);
            ErrorOccurred?.Invoke(ex);
            return false;
        }
    }

    public void Disconnect()
    {
        if (_port == null) return;
        _port.DataReceived -= OnDataReceived;
        _port.ErrorReceived -= OnErrorReceived;
        if (_port.IsOpen) _port.Close();
        _port.Dispose();
        _port = null;
        _buffer.Clear();
        _logger.LogInformation("EEG serial port disconnected");
        ConnectionStatusChanged?.Invoke("Disconnected");
    }

    private void OnDataReceived(object sender, Ports.SerialDataReceivedEventArgs e)
    {
        if (_port == null || !_port.IsOpen) return;
        try
        {
            int available = _port.BytesToRead;
            var buf = new byte[available];
            _port.Read(buf, 0, available);

            lock (_bufLock)
            {
                _totalBytesIn += available;
                _buffer.AddRange(buf);
                ProcessBuffer();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error reading EEG serial data");
            ErrorOccurred?.Invoke(ex);
        }
    }

    private void OnErrorReceived(object sender, Ports.SerialErrorReceivedEventArgs e)
    {
        _logger.LogWarning("EEG serial port error: {Error}", e.EventType);
    }

    // Sliding-window frame sync per protocol §1.5
    private void ProcessBuffer()
    {
        while (true)
        {
            // Find 0x2A header
            int headerIdx = -1;
            for (int i = 0; i < _buffer.Count; i++)
            {
                if (_buffer[i] == FRAME_HEADER) { headerIdx = i; break; }
            }

            if (headerIdx < 0)
            {
                _buffer.Clear();
                return;
            }

            if (headerIdx > 0)
                _buffer.RemoveRange(0, headerIdx);

            // Wait for a full frame
            if (_buffer.Count < FRAME_SIZE)
                return;

            // Validate PktNum ∈ {0x01..0x04}
            byte pktNum = _buffer[1];
            if (pktNum < 0x01 || pktNum > 0x04)
            {
                _buffer.RemoveAt(0);
                continue;
            }

            // Validate tail 0x40 0x40
            if (_buffer[514] != FRAME_TAIL || _buffer[515] != FRAME_TAIL)
            {
                _buffer.RemoveAt(0);
                continue;
            }

            // Valid frame: copy to temp buffer and emit
            for (int i = 0; i < FRAME_SIZE; i++)
                _frameTemp[i] = _buffer[i];

            EmitSamples();
            _buffer.RemoveRange(0, FRAME_SIZE);
            _framesDecoded++;

            // Report streaming status every 16 frames (every 4 seconds)
            if (_framesDecoded == 1 || (_framesDecoded % 16) == 0)
            {
                ConnectionStatusChanged?.Invoke(
                    $"接收中: {PortName}  {_framesDecoded} 帧  {_totalBytesIn / 1024} KB");
            }
        }
    }

    private void EmitSamples()
    {
        var now = DateTime.Now;
        // Back-date samples: packet covers the last 64/256 Hz = 250 ms
        var baseTime = now.AddMilliseconds(-(SAMPLES_PER_PACKET - 1) * 1000.0 / SAMPLE_RATE);

        for (int i = 0; i < SAMPLES_PER_PACKET; i++)
        {
            var channels = new double[CHANNEL_COUNT];
            for (int ch = 0; ch < CHANNEL_COUNT; ch++)
            {
                int offset = 2 + ch * 256 + i * 4;
                float raw = BitConverter.ToSingle(_frameTemp, offset);
                double uv = raw * UV_SCALE;
                uv = _filters[ch, 0].Process(uv); // 50 Hz notch
                uv = _filters[ch, 1].Process(uv); // 0.5 Hz HPF
                uv = _filters[ch, 2].Process(uv); // 95 Hz LPF
                channels[ch] = uv;
            }

            var timestamp = baseTime.AddMilliseconds(i * 1000.0 / SAMPLE_RATE);
            SampleReceived?.Invoke(new EEGSample(timestamp, channels));
        }
    }

    private void ResetFilters()
    {
        for (int ch = 0; ch < CHANNEL_COUNT; ch++)
        {
            _filters[ch, 0].Reset();
            _filters[ch, 1].Reset();
            _filters[ch, 2].Reset();
        }
    }

    public void Dispose() => Disconnect();
}

/// <summary>
/// Direct Form II Transposed 2nd-order IIR biquad.
/// H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
/// </summary>
internal sealed class BiquadFilter
{
    private readonly double _b0, _b1, _b2, _a1, _a2;
    private double _s1, _s2;

    private BiquadFilter(double b0, double b1, double b2, double a1, double a2)
    {
        _b0 = b0; _b1 = b1; _b2 = b2;
        _a1 = a1; _a2 = a2;
    }

    public double Process(double x)
    {
        double y = _b0 * x + _s1;
        _s1 = _b1 * x - _a1 * y + _s2;
        _s2 = _b2 * x - _a2 * y;
        return y;
    }

    public void Reset() { _s1 = _s2 = 0.0; }

    /// <summary>2nd-order Butterworth low-pass (bilinear transform).</summary>
    public static BiquadFilter Lowpass(double fc, double fs)
    {
        double K = Math.Tan(Math.PI * fc / fs);
        double K2 = K * K;
        double norm = 1.0 + Math.Sqrt(2.0) * K + K2;
        return new BiquadFilter(
            b0: K2 / norm,
            b1: 2.0 * K2 / norm,
            b2: K2 / norm,
            a1: 2.0 * (K2 - 1.0) / norm,
            a2: (1.0 - Math.Sqrt(2.0) * K + K2) / norm);
    }

    /// <summary>2nd-order Butterworth high-pass (bilinear transform).</summary>
    public static BiquadFilter Highpass(double fc, double fs)
    {
        double K = Math.Tan(Math.PI * fc / fs);
        double K2 = K * K;
        double norm = 1.0 + Math.Sqrt(2.0) * K + K2;
        return new BiquadFilter(
            b0: 1.0 / norm,
            b1: -2.0 / norm,
            b2: 1.0 / norm,
            a1: 2.0 * (K2 - 1.0) / norm,
            a2: (1.0 - Math.Sqrt(2.0) * K + K2) / norm);
    }

    /// <summary>IIR notch filter at f0 Hz with quality factor q.</summary>
    public static BiquadFilter Notch(double f0, double fs, double q)
    {
        double w0 = 2.0 * Math.PI * f0 / fs;
        double alpha = Math.Sin(w0) / (2.0 * q);
        double cosw0 = Math.Cos(w0);
        double a0 = 1.0 + alpha;
        return new BiquadFilter(
            b0: 1.0 / a0,
            b1: -2.0 * cosw0 / a0,
            b2: 1.0 / a0,
            a1: -2.0 * cosw0 / a0,
            a2: (1.0 - alpha) / a0);
    }
}
