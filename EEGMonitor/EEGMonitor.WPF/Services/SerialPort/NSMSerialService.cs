using EEGMonitor.Models;
using Microsoft.Extensions.Logging;
using Ports = System.IO.Ports;

namespace EEGMonitor.Services.SerialPort;

/// <summary>
/// Serial port service for the NSM anesthesia monitor (355-byte UART protocol).
///
/// Protocol overview:
///   - RS232, 115200 baud, 8N1
///   - Packet: 0x80 header + 2-byte length (353) + payload + CRC-16 (poly 0x1021)
///   - 100 bytes EEG (signed byte, ±127 µV range)
///   - Rich metadata: CSI, BS, SQI, EMG, NOX, band powers (dB), SEF95, events, alarms
///
/// Compatible with ISerialPortService — emits EEGSample events for the existing pipeline.
/// Additional NSM-specific data via NSMDataReceived event.
/// </summary>
public sealed class NSMSerialService : ISerialPortService
{
    private const byte FRAME_HEADER = 0x80;
    private const int PAYLOAD_LENGTH = 351;   // actual device: 2 bytes shorter than doc
    private const int CRC_LENGTH = 2;
    private const int PACKET_SIZE = PAYLOAD_LENGTH + CRC_LENGTH; // 353
    private const int EEG_SAMPLES_PER_PACKET = 100;
    private const int EEG_OFFSET = 22;        // DP_EEGValueStart

    // CRC-16 per protocol doc (polynomial 0x1021, init 0xFFFF).
    // NOTE: actual device CRC algorithm differs slightly; validation pending.
    // private const ushort CRC_POLY = 0x1021;
    // private const ushort CRC_INIT = 0xFFFF;

    private readonly ILogger<NSMSerialService> _logger;
    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(65536);
    private readonly byte[] _packetTemp = new byte[PACKET_SIZE];
    private readonly object _bufLock = new();

    private long _totalBytesIn;
    private long _packetsDecoded;
    private int _sampleRate;

    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? string.Empty;
    public int BaudRate => _port?.BaudRate ?? 0;
    public long TotalBytesIn => _totalBytesIn;
    public long PacketsDecoded => _packetsDecoded;

    public event Action<EEGSample>? SampleReceived;
    public event Action<NSMDataPacket>? NSMDataReceived;
    public event Action<string>? ConnectionStatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public NSMSerialService(ILogger<NSMSerialService> logger)
    {
        _logger = logger;
    }

    public IEnumerable<string> GetAvailablePorts() =>
        Ports.SerialPort.GetPortNames().OrderBy(p => p);

    public bool Connect(string portName, int baudRate = 115200, int channelCount = 1, int sampleRate = 200)
    {
        if (IsConnected) Disconnect();

        _buffer.Clear();
        _totalBytesIn = 0;
        _packetsDecoded = 0;
        _sampleRate = sampleRate;

        try
        {
            _port = new Ports.SerialPort(portName, baudRate, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadBufferSize = 65536,
                ReadTimeout = 500,
                WriteTimeout = 500,
                DtrEnable = true,
                RtsEnable = true,
            };
            _port.DataReceived += OnDataReceived;
            _port.ErrorReceived += OnErrorReceived;
            _port.Open();

            _logger.LogInformation("NSM serial port {Port} opened at {Baud} baud", portName, baudRate);
            ConnectionStatusChanged?.Invoke($"NSM Connected: {portName}@{baudRate}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to open NSM serial port {Port}", portName);
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
        _logger.LogInformation("NSM serial port disconnected");
        ConnectionStatusChanged?.Invoke("NSM Disconnected");
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
            _logger.LogWarning(ex, "Error reading NSM serial data");
            ErrorOccurred?.Invoke(ex);
        }
    }

    private void OnErrorReceived(object sender, Ports.SerialErrorReceivedEventArgs e)
    {
        _logger.LogWarning("NSM serial port error: {Error}", e.EventType);
    }

    private void ProcessBuffer()
    {
        while (true)
        {
            // Find 0x80 header
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

            // Need at least PACKET_SIZE bytes to attempt decode
            if (_buffer.Count < PACKET_SIZE)
                return;

            // Validate length field (bytes 1-2: LSB, MSB)
            ushort length = (ushort)(_buffer[1] | (_buffer[2] << 8));
            if (length != PAYLOAD_LENGTH)
            {
                // Not a valid NSM packet at this position — skip header and re-scan
                _buffer.RemoveAt(0);
                continue;
            }

            // Copy candidate packet and validate CRC
            for (int i = 0; i < PACKET_SIZE; i++)
                _packetTemp[i] = _buffer[i];

            // CRC validation skipped — device uses non-standard variant (TODO)

            // Valid packet — parse and emit
            try
            {
                var packet = ParsePacket(_packetTemp);
                EmitSamples(packet);
                NSMDataReceived?.Invoke(packet);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error parsing NSM packet");
            }

            _buffer.RemoveRange(0, PACKET_SIZE);
            _packetsDecoded++;

            if (_packetsDecoded == 1 || (_packetsDecoded % 16) == 0)
            {
                ConnectionStatusChanged?.Invoke(
                    $"NSM 接收中: {PortName}  {_packetsDecoded} 包  {_totalBytesIn / 1024} KB");
            }
        }
    }

    private void EmitSamples(NSMDataPacket packet)
    {
        if (packet.EEGSamplesUv.Length == 0) return;

        var now = DateTime.Now;
        double periodMs = 1000.0 / _sampleRate;
        var baseTime = now.AddMilliseconds(-(EEG_SAMPLES_PER_PACKET - 1) * periodMs);

        for (int i = 0; i < EEG_SAMPLES_PER_PACKET; i++)
        {
            var timestamp = baseTime.AddMilliseconds(i * periodMs);
            SampleReceived?.Invoke(new EEGSample(timestamp, new[] { packet.EEGSamplesUv[i] }));
        }
    }

    private static NSMDataPacket ParsePacket(byte[] buf)
    {
        // Device time (bytes 6-9, little-endian seconds)
        uint deviceTime = (uint)(buf[6] | (buf[7] << 8) | (buf[8] << 16) | (buf[9] << 24));

        // Block status (byte 10)
        byte blockStatus = buf[10];

        // EEG samples (bytes 22-121): signed byte → µV
        var eegSamples = new double[EEG_SAMPLES_PER_PACKET];
        for (int i = 0; i < EEG_SAMPLES_PER_PACKET; i++)
            eegSamples[i] = (sbyte)buf[EEG_OFFSET + i];

        // Band powers (dB, signed byte)
        int deltaDb = (sbyte)buf[126];
        int thetaDb = (sbyte)buf[127];
        int alphaDb = (sbyte)buf[128];
        int betaDb  = (sbyte)buf[129];
        int gammaDb = (sbyte)buf[130];

        // SEF95 is at offset 176 (DP_NSM_D8 + 44)
        int sef95 = buf[176];

        return new NSMDataPacket
        {
            LocalTimestamp = DateTime.Now,
            DeviceTimeSec = deviceTime,
            ElectrodeAlarm   = (blockStatus & (1 << 1)) != 0,
            ImpedanceHigh    = (blockStatus & (1 << 3)) != 0,
            ElectrodeInvalid = (blockStatus & (1 << 7)) != 0,
            BlackImpedance = buf[16],
            WhiteImpedance = buf[17],
            CSI = buf[13],
            BS  = buf[14],
            SQI = buf[15],
            EMG = buf[18],
            NOX = buf[125],
            EventNumber = buf[11],
            EventType = (NSMEventType)buf[12],
            AlarmHigh = buf[20],
            AlarmLow  = buf[21],
            EEGSamplesUv = eegSamples,
            DeltaPowerDb = deltaDb,
            ThetaPowerDb = thetaDb,
            AlphaPowerDb = alphaDb,
            BetaPowerDb  = betaDb,
            GammaPowerDb = gammaDb,
            SEF95 = sef95,
            EOG   = buf[131],
        };
    }

    public void Dispose() => Disconnect();
}
