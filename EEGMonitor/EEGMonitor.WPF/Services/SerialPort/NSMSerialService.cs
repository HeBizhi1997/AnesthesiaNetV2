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

    // Auto-detection: the NSM device outputs at a fixed internal rate (~100 Hz).
    // We measure inter-packet arrival times to infer the true rate, independent
    // of what the user selected in the UI dropdown.
    private DateTime _lastPacketTime = DateTime.MinValue;
    private readonly List<double> _packetIntervalsMs = new(10);
    private int _detectedSampleRate;
    private bool _rateLocked;

    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? string.Empty;
    public int BaudRate => _port?.BaudRate ?? 0;
    public long TotalBytesIn => _totalBytesIn;
    public long PacketsDecoded => _packetsDecoded;

    /// <summary>
    /// Auto-detected sample rate from packet arrival timing.
    /// Falls back to the configured rate until enough packets have arrived.
    /// </summary>
    public int DetectedSampleRate =>
        _rateLocked ? _detectedSampleRate : _sampleRate;

    /// <summary>ISerialPortService.SampleRate — exposes the auto-detected NSM rate.</summary>
    public int SampleRate => DetectedSampleRate;

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
        _lastPacketTime = DateTime.MinValue;
        _packetIntervalsMs.Clear();
        _detectedSampleRate = sampleRate;
        _rateLocked = false;

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

            // ── Auto-detect actual sample rate from packet timing ──────────
            var now = DateTime.UtcNow;
            if (_lastPacketTime != DateTime.MinValue)
            {
                double intervalMs = (now - _lastPacketTime).TotalMilliseconds;
                if (intervalMs > 100 && intervalMs < 5000)  // sane range
                {
                    _packetIntervalsMs.Add(intervalMs);
                    if (_packetIntervalsMs.Count > 10)
                        _packetIntervalsMs.RemoveAt(0);

                    if (_packetIntervalsMs.Count >= 5 && !_rateLocked)
                    {
                        double avgIntervalMs = _packetIntervalsMs.Average();
                        // 100 samples per packet → sample rate = 100 / interval_s
                        _detectedSampleRate = (int)Math.Round(100000.0 / avgIntervalMs);
                        _detectedSampleRate = Math.Clamp(_detectedSampleRate, 50, 500);
                        _rateLocked = true;

                        if (Math.Abs(_detectedSampleRate - _sampleRate) > 10)
                        {
                            _logger.LogWarning(
                                "NSM rate mismatch: configured={0} Hz, detected={1} Hz. Using detected rate.",
                                _sampleRate, _detectedSampleRate);
                        }
                        else
                        {
                            _logger.LogInformation(
                                "NSM rate confirmed: {0} Hz (configured: {1} Hz)",
                                _detectedSampleRate, _sampleRate);
                        }
                    }
                }
            }
            _lastPacketTime = now;

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
        // 用**有效采样率**(实测锁定后的 DetectedSampleRate;未锁定前回退配置值=100)生成逐样本时间戳,
        // 与分块/Python 端使用的采样率保持一致(避免 100/200 Hz 错配导致时间轴与频谱平移)。
        int effectiveRate = DetectedSampleRate > 0 ? DetectedSampleRate : _sampleRate;
        double periodMs = 1000.0 / effectiveRate;
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
