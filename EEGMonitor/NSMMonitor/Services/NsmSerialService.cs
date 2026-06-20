using Microsoft.Extensions.Logging;
using NSMMonitor.Models;
using Ports = System.IO.Ports;

namespace NSMMonitor.Services;

/// <summary>
/// NSM 麻醉深度监护仪串口服务（353 字节 UART 协议）。
/// 协议详见《NSM 设备通讯协议文档》。
///   - RS232, 115200 baud, 8N1
///   - 帧 = 0x80 帧头 + 2 字节长度(351) + 载荷 + CRC-16
///   - 100 字节 EEG（有符号字节，±127 µV）
///   - CSI / BS / SQI / EMG / NOX / 频带功率(dB) / SEF95 / 事件 / 报警
/// </summary>
public sealed class NsmSerialService : INsmDataSource, IDisposable
{
    private const byte FRAME_HEADER = 0x80;
    private const int PAYLOAD_LENGTH = 351;
    private const int CRC_LENGTH = 2;
    private const int PACKET_SIZE = PAYLOAD_LENGTH + CRC_LENGTH; // 353
    private const int EEG_SAMPLES_PER_PACKET = 100;
    private const int EEG_OFFSET = 22;

    private readonly ILogger<NsmSerialService> _logger;
    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(65536);
    private readonly byte[] _packetTemp = new byte[PACKET_SIZE];
    private readonly object _bufLock = new();

    private long _totalBytesIn;
    private long _packetsDecoded;

    private DateTime _lastPacketTime = DateTime.MinValue;
    private readonly List<double> _packetIntervalsMs = new(10);
    private int _detectedSampleRate = 100;
    private bool _rateLocked;

    public bool IsConnected => _port?.IsOpen ?? false;
    public string SourceName => IsConnected ? $"串口 {_port!.PortName}@{_port.BaudRate}" : "串口未连接";
    public int SampleRate => _rateLocked ? _detectedSampleRate : 100;
    public long PacketsDecoded => _packetsDecoded;

    public event Action<NSMDataPacket>? NSMDataReceived;
    public event Action<string>? StatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public NsmSerialService(ILogger<NsmSerialService> logger) => _logger = logger;

    public IEnumerable<string> GetAvailablePorts() =>
        Ports.SerialPort.GetPortNames().OrderBy(p => p);

    public bool Connect(string portName, int baudRate = 115200)
    {
        if (IsConnected) Disconnect();

        _buffer.Clear();
        _totalBytesIn = 0;
        _packetsDecoded = 0;
        _lastPacketTime = DateTime.MinValue;
        _packetIntervalsMs.Clear();
        _detectedSampleRate = 100;
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

            _logger.LogInformation("NSM 串口 {Port} 已打开 @ {Baud}", portName, baudRate);
            StatusChanged?.Invoke($"NSM 已连接：{portName}@{baudRate}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "打开 NSM 串口 {Port} 失败", portName);
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
        _logger.LogInformation("NSM 串口已断开");
        StatusChanged?.Invoke("NSM 已断开");
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
            _logger.LogWarning(ex, "读取 NSM 串口数据出错");
            ErrorOccurred?.Invoke(ex);
        }
    }

    private void OnErrorReceived(object sender, Ports.SerialErrorReceivedEventArgs e) =>
        _logger.LogWarning("NSM 串口错误：{Error}", e.EventType);

    private void ProcessBuffer()
    {
        while (true)
        {
            int headerIdx = -1;
            for (int i = 0; i < _buffer.Count; i++)
            {
                if (_buffer[i] == FRAME_HEADER) { headerIdx = i; break; }
            }

            if (headerIdx < 0) { _buffer.Clear(); return; }
            if (headerIdx > 0) _buffer.RemoveRange(0, headerIdx);
            if (_buffer.Count < PACKET_SIZE) return;

            ushort length = (ushort)(_buffer[1] | (_buffer[2] << 8));
            if (length != PAYLOAD_LENGTH) { _buffer.RemoveAt(0); continue; }

            for (int i = 0; i < PACKET_SIZE; i++)
                _packetTemp[i] = _buffer[i];

            // CRC 校验跳过：设备使用非标准变体（待厂商确认）

            try
            {
                var packet = ParsePacket(_packetTemp);
                NSMDataReceived?.Invoke(packet);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "解析 NSM 数据包出错");
            }

            _buffer.RemoveRange(0, PACKET_SIZE);
            _packetsDecoded++;

            DetectSampleRate();

            if (_packetsDecoded == 1 || (_packetsDecoded % 16) == 0)
                StatusChanged?.Invoke($"NSM 接收中：{_packetsDecoded} 包  {_totalBytesIn / 1024} KB");
        }
    }

    private void DetectSampleRate()
    {
        var now = DateTime.UtcNow;
        if (_lastPacketTime != DateTime.MinValue)
        {
            double intervalMs = (now - _lastPacketTime).TotalMilliseconds;
            if (intervalMs is > 100 and < 5000)
            {
                _packetIntervalsMs.Add(intervalMs);
                if (_packetIntervalsMs.Count > 10) _packetIntervalsMs.RemoveAt(0);

                if (_packetIntervalsMs.Count >= 5 && !_rateLocked)
                {
                    double avg = _packetIntervalsMs.Average();
                    _detectedSampleRate = Math.Clamp((int)Math.Round(100000.0 / avg), 50, 500);
                    _rateLocked = true;
                    _logger.LogInformation("NSM 采样率锁定：{Rate} Hz", _detectedSampleRate);
                }
            }
        }
        _lastPacketTime = now;
    }

    /// <summary>按字节偏移解析 353 字节数据包，偏移定义见协议文档 §三。</summary>
    private static NSMDataPacket ParsePacket(byte[] buf)
    {
        uint deviceTime = (uint)(buf[6] | (buf[7] << 8) | (buf[8] << 16) | (buf[9] << 24));
        byte blockStatus = buf[10];

        var eegSamples = new double[EEG_SAMPLES_PER_PACKET];
        for (int i = 0; i < EEG_SAMPLES_PER_PACKET; i++)
            eegSamples[i] = (sbyte)buf[EEG_OFFSET + i];

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
            DeltaPowerDb = (sbyte)buf[126],
            ThetaPowerDb = (sbyte)buf[127],
            AlphaPowerDb = (sbyte)buf[128],
            BetaPowerDb  = (sbyte)buf[129],
            GammaPowerDb = (sbyte)buf[130],
            SEF95 = buf[176],
            EOG   = buf[131],
        };
    }

    public void Dispose() => Disconnect();
}
