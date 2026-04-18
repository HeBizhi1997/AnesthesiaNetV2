using EEGMonitor.Models;
using Microsoft.Extensions.Logging;
using Ports = System.IO.Ports;

namespace EEGMonitor.Services.SerialPort;

/// <summary>
/// Reads raw EEG and vitals from a serial device.
/// Supports a simple binary packet format:
///   Header: 0xAA 0xBB
///   Channel count (1 byte)
///   N×float32 (channel voltages µV)
///   Optional: SpO2 (float32), HR (float32), Pulse (float32)
///   Checksum: XOR of all data bytes
/// Also handles OpenBCI-style ASCII CSV for easy integration.
/// </summary>
public sealed class SerialPortService : ISerialPortService
{
    private readonly ILogger<SerialPortService> _logger;
    private Ports.SerialPort? _port;
    private readonly byte[] _receiveBuffer = new byte[4096];
    private int _bufferPos;
    private int _channelCount = 4;
    private int _sampleRate = 256;

    private enum ParseMode { Binary, AsciiCSV }
    private ParseMode _parseMode = ParseMode.AsciiCSV; // default: easy ASCII

    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? string.Empty;
    public int BaudRate => _port?.BaudRate ?? 0;

    public event Action<EEGSample>? SampleReceived;
    public event Action<string>? ConnectionStatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public SerialPortService(ILogger<SerialPortService> logger)
    {
        _logger = logger;
    }

    public IEnumerable<string> GetAvailablePorts() =>
        Ports.SerialPort.GetPortNames().OrderBy(p => p);

    public bool Connect(string portName, int baudRate, int channelCount = 4, int sampleRate = 256)
    {
        if (IsConnected) Disconnect();

        _channelCount = channelCount;
        _sampleRate = sampleRate;
        _bufferPos = 0;

        try
        {
            _port = new Ports.SerialPort(portName, baudRate, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadTimeout = 500,
                WriteTimeout = 500,
                DtrEnable = true,
            };
            _port.DataReceived += OnDataReceived;
            _port.ErrorReceived += OnErrorReceived;
            _port.Open();

            _logger.LogInformation("Serial port {Port} opened at {Baud} baud", portName, baudRate);
            ConnectionStatusChanged?.Invoke($"Connected: {portName}@{baudRate}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to open serial port {Port}", portName);
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
        _logger.LogInformation("Serial port disconnected");
        ConnectionStatusChanged?.Invoke("Disconnected");
    }

    private void OnDataReceived(object sender, Ports.SerialDataReceivedEventArgs e)
    {
        if (_port == null || !_port.IsOpen) return;
        try
        {
            var available = _port.BytesToRead;
            var buf = new byte[available];
            _port.Read(buf, 0, available);
            ProcessIncoming(buf);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error reading serial data");
            ErrorOccurred?.Invoke(ex);
        }
    }

    private void OnErrorReceived(object sender, Ports.SerialErrorReceivedEventArgs e)
    {
        _logger.LogWarning("Serial port error: {Error}", e.EventType);
    }

    private readonly List<byte> _lineBuffer = new();

    private void ProcessIncoming(byte[] data)
    {
        // ASCII CSV mode: parse newline-terminated lines
        // Expected format: timestamp_ms,ch0,ch1,...,chN[,spo2,hr,pulse]
        // or simplified:   ch0,ch1,...,chN
        if (_parseMode == ParseMode.AsciiCSV)
        {
            foreach (var b in data)
            {
                if (b == '\n')
                {
                    var line = System.Text.Encoding.ASCII.GetString(_lineBuffer.ToArray()).Trim();
                    _lineBuffer.Clear();
                    if (!string.IsNullOrEmpty(line) && line[0] != '#')
                        ParseAsciiLine(line);
                }
                else
                {
                    _lineBuffer.Add(b);
                }
            }
        }
        else
        {
            ParseBinaryStream(data);
        }
    }

    private void ParseAsciiLine(string line)
    {
        var parts = line.Split(',');
        if (parts.Length < _channelCount) return;

        int offset = 0;
        // If first token is a large integer, treat it as timestamp
        DateTime timestamp;
        if (long.TryParse(parts[0], out var ms) && ms > 1_000_000)
        {
            timestamp = DateTimeOffset.FromUnixTimeMilliseconds(ms).LocalDateTime;
            offset = 1;
        }
        else
        {
            timestamp = DateTime.Now;
        }

        var channels = new double[_channelCount];
        for (int i = 0; i < _channelCount; i++)
        {
            if (!double.TryParse(parts[offset + i],
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,
                out channels[i]))
                channels[i] = 0;
        }

        double? spo2 = null, hr = null, pulse = null;
        int vitalsStart = offset + _channelCount;
        if (parts.Length > vitalsStart && double.TryParse(parts[vitalsStart], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var s)) spo2 = s;
        if (parts.Length > vitalsStart + 1 && double.TryParse(parts[vitalsStart + 1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var h)) hr = h;
        if (parts.Length > vitalsStart + 2 && double.TryParse(parts[vitalsStart + 2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var p)) pulse = p;

        SampleReceived?.Invoke(new EEGSample(timestamp, channels, spo2, hr, pulse));
    }

    private void ParseBinaryStream(byte[] data)
    {
        // Binary packet: 0xAA 0xBB | channelCount | float32*N | [spo2 float32 | hr float32 | pulse float32] | checksum
        const int HEADER0 = 0xAA, HEADER1 = 0xBB;
        for (int i = 0; i < data.Length; i++)
        {
            _receiveBuffer[_bufferPos++] = data[i];
            if (_bufferPos >= 2
                && _receiveBuffer[_bufferPos - 2] == HEADER0
                && _receiveBuffer[_bufferPos - 1] == HEADER1)
            {
                _bufferPos = 2;
            }
            int payloadLen = 1 + _channelCount * 4 + 3 * 4 + 1;
            if (_bufferPos == 2 + payloadLen)
            {
                TryParseBinaryPacket(_receiveBuffer.AsSpan(0, 2 + payloadLen));
                _bufferPos = 0;
            }
        }
    }

    private void TryParseBinaryPacket(Span<byte> packet)
    {
        int n = packet[2];
        if (n != _channelCount) return;
        var channels = new double[n];
        int pos = 3;
        for (int i = 0; i < n; i++)
        {
            channels[i] = BitConverter.ToSingle(packet.Slice(pos, 4));
            pos += 4;
        }
        var spo2 = (double)BitConverter.ToSingle(packet.Slice(pos, 4)); pos += 4;
        var hr = (double)BitConverter.ToSingle(packet.Slice(pos, 4)); pos += 4;
        var pulse = (double)BitConverter.ToSingle(packet.Slice(pos, 4));
        SampleReceived?.Invoke(new EEGSample(DateTime.Now, channels, spo2, hr, pulse));
    }

    public void Dispose() => Disconnect();
}
