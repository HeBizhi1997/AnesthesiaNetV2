using Microsoft.Extensions.Logging;
using Ports = System.IO.Ports;

namespace EEGMonitor.Services.SerialPort;

/// <summary>
/// Reads BPM from the HKG-07D pulse sensor using its 5-byte passive push protocol:
///   [0] 0xF0   frame header byte 1
///   [1] 0xC0   frame header byte 2
///   [2] MLH    BPM high byte
///   [3] MLL    BPM low byte
///   [4] CKSUM  (0xF0 + 0xC0 + MLH + MLL) & 0xFF
///
/// BPM = (MLH &lt;&lt; 8) | MLL.  BPM == 0 means no signal.
/// The sensor pushes a frame automatically every ~9 seconds at 19200 baud.
/// </summary>
public sealed class PulseSerialService : IPulseSerialService
{
    private const int BAUD_RATE = 19200;
    private const int FRAME_SIZE = 5;
    private const byte HEADER0 = 0xF0;
    private const byte HEADER1 = 0xC0;

    private readonly ILogger<PulseSerialService> _logger;
    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(64);

    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? string.Empty;

    public event Action<int>? BpmReceived;
    public event Action<string>? ConnectionStatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public PulseSerialService(ILogger<PulseSerialService> logger)
    {
        _logger = logger;
    }

    public bool Connect(string portName)
    {
        if (IsConnected) Disconnect();
        _buffer.Clear();

        try
        {
            _port = new Ports.SerialPort(portName, BAUD_RATE, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadTimeout = 500,
                WriteTimeout = 500,
            };
            _port.DataReceived += OnDataReceived;
            _port.ErrorReceived += OnErrorReceived;
            _port.Open();

            _logger.LogInformation("Pulse serial port {Port} opened at {Baud} baud", portName, BAUD_RATE);
            ConnectionStatusChanged?.Invoke($"脉搏已连接: {portName}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to open pulse serial port {Port}", portName);
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
        _logger.LogInformation("Pulse serial port disconnected");
        ConnectionStatusChanged?.Invoke("脉搏已断开");
    }

    private void OnDataReceived(object sender, Ports.SerialDataReceivedEventArgs e)
    {
        if (_port == null || !_port.IsOpen) return;
        try
        {
            int available = _port.BytesToRead;
            var buf = new byte[available];
            _port.Read(buf, 0, available);
            _buffer.AddRange(buf);
            ProcessBuffer();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error reading pulse serial data");
            ErrorOccurred?.Invoke(ex);
        }
    }

    private void OnErrorReceived(object sender, Ports.SerialErrorReceivedEventArgs e)
    {
        _logger.LogWarning("Pulse serial port error: {Error}", e.EventType);
    }

    private void ProcessBuffer()
    {
        while (_buffer.Count >= FRAME_SIZE)
        {
            // Find 0xF0 header byte
            int headerIdx = -1;
            for (int i = 0; i < _buffer.Count; i++)
            {
                if (_buffer[i] == HEADER0) { headerIdx = i; break; }
            }

            if (headerIdx < 0)
            {
                _buffer.Clear();
                return;
            }

            if (headerIdx > 0)
                _buffer.RemoveRange(0, headerIdx);

            if (_buffer.Count < FRAME_SIZE)
                return;

            // Validate second header byte
            if (_buffer[1] != HEADER1)
            {
                _buffer.RemoveAt(0);
                continue;
            }

            byte mlh = _buffer[2];
            byte mll = _buffer[3];
            byte cksum = _buffer[4];

            byte expected = (byte)((HEADER0 + HEADER1 + mlh + mll) & 0xFF);
            if (cksum != expected)
            {
                _buffer.RemoveAt(0);
                continue;
            }

            int bpm = (mlh << 8) | mll;
            _buffer.RemoveRange(0, FRAME_SIZE);

            _logger.LogDebug("Pulse BPM received: {BPM}", bpm);
            BpmReceived?.Invoke(bpm);
        }
    }

    public void Dispose() => Disconnect();
}
