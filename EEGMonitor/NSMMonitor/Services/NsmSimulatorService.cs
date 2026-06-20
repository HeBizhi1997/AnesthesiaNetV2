using System.Timers;
using Microsoft.Extensions.Logging;
using NSMMonitor.Models;
using Timer = System.Timers.Timer;

namespace NSMMonitor.Services;

/// <summary>
/// NSM 数据模拟器：无真实硬件时生成符合协议的仿真数据包，
/// 每秒推送一帧（含 100 个 EEG 样本），用于演示与界面联调。
/// 指标随时间缓慢漂移并叠加随机扰动，模拟真实麻醉过程趋势。
/// </summary>
public sealed class NsmSimulatorService : INsmDataSource, IDisposable
{
    private const int EEG_SAMPLES_PER_PACKET = 100;
    private const int SIM_SAMPLE_RATE = 100;

    private readonly ILogger<NsmSimulatorService> _logger;
    private readonly Random _rng = new();
    private Timer? _timer;
    private bool _running;
    private long _packets;
    private double _phase;
    private uint _deviceTime;

    // 缓慢漂移的基线指标
    private double _csiBase = 55;
    private double _noxBase = 42;
    private double _bsBase = 0;
    private int _eventCounter;

    public bool IsConnected => _running;
    public string SourceName => "内置模拟器";
    public int SampleRate => SIM_SAMPLE_RATE;
    public long PacketsDecoded => _packets;

    public event Action<NSMDataPacket>? NSMDataReceived;
    public event Action<string>? StatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public NsmSimulatorService(ILogger<NsmSimulatorService> logger) => _logger = logger;

    public IEnumerable<string> GetAvailablePorts() => new[] { "SIM" };

    public bool Connect(string portName = "SIM", int baudRate = 115200)
    {
        if (_running) Disconnect();

        _packets = 0;
        _phase = 0;
        _deviceTime = 0;
        _csiBase = 55;
        _noxBase = 42;
        _bsBase = 0;
        _eventCounter = 0;

        _timer = new Timer(1000) { AutoReset = true };
        _timer.Elapsed += OnTick;
        _running = true;
        _timer.Start();

        _logger.LogInformation("NSM 模拟器已启动");
        StatusChanged?.Invoke("NSM 模拟器已启动");
        return true;
    }

    public void Disconnect()
    {
        _running = false;
        if (_timer != null)
        {
            _timer.Elapsed -= OnTick;
            _timer.Stop();
            _timer.Dispose();
            _timer = null;
        }
        _logger.LogInformation("NSM 模拟器已停止");
        StatusChanged?.Invoke("NSM 模拟器已停止");
    }

    private void OnTick(object? sender, ElapsedEventArgs e)
    {
        if (!_running) return;
        try
        {
            var packet = BuildPacket();
            _packets++;
            _deviceTime++;
            NSMDataReceived?.Invoke(packet);

            if (_packets == 1 || _packets % 16 == 0)
                StatusChanged?.Invoke($"NSM 模拟器：{_packets} 包");
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(ex);
        }
    }

    private NSMDataPacket BuildPacket()
    {
        // 指标缓慢漂移 + 扰动
        _csiBase = Clamp(_csiBase + (_rng.NextDouble() - 0.5) * 4, 38, 72);
        _noxBase = Clamp(_noxBase + (_rng.NextDouble() - 0.5) * 5, 18, 70);
        _bsBase  = Clamp(_bsBase + (_rng.NextDouble() - 0.55) * 2, 0, 20);

        int csi = (int)Math.Round(_csiBase);
        int nox = (int)Math.Round(_noxBase);
        int bs  = (int)Math.Round(_bsBase);
        int sqi = 78 + _rng.Next(0, 22);          // 78-99
        int emg = 20 + _rng.Next(0, 35);          // 20-54

        // EEG 波形：随麻醉深度变化幅度，叠加多频成分 + 噪声
        double amp = 35 + (60 - csi) * 0.8;       // 越深幅度越大
        var eeg = new double[EEG_SAMPLES_PER_PACKET];
        for (int i = 0; i < EEG_SAMPLES_PER_PACKET; i++)
        {
            double t = _phase + i / (double)SIM_SAMPLE_RATE;
            double v = amp * 0.5 * Math.Sin(2 * Math.PI * 2.0 * t)    // δ/θ 主成分
                     + amp * 0.25 * Math.Sin(2 * Math.PI * 10.0 * t)  // α
                     + amp * 0.12 * Math.Sin(2 * Math.PI * 22.0 * t)  // β
                     + (_rng.NextDouble() - 0.5) * 14;                // 噪声
            eeg[i] = Clamp(v, -127, 127);
        }
        _phase += EEG_SAMPLES_PER_PACKET / (double)SIM_SAMPLE_RATE;

        // 频带功率（dB）：随麻醉深度变化，δ 占优
        int delta = (int)Clamp(8 + (60 - csi) * 0.3 + Noise(2), -40, 40);
        int theta = (int)Clamp(2 + Noise(3), -40, 40);
        int alpha = (int)Clamp(-2 + Noise(3), -40, 40);
        int beta  = (int)Clamp(-8 + csi * 0.1 + Noise(3), -40, 40);
        int gamma = (int)Clamp(-15 + Noise(3), -40, 40);

        // 偶发临床事件（约每 20 包一次）
        int eventNumber = 0;
        NSMEventType eventType = NSMEventType.General;
        if (_rng.NextDouble() < 0.05)
        {
            eventNumber = ++_eventCounter;
            eventType = (NSMEventType)_rng.Next(1, 9);
        }

        // 偶发阻抗波动
        int blackImp = _rng.NextDouble() < 0.04 ? 15 : _rng.Next(1, 6);
        int whiteImp = _rng.NextDouble() < 0.04 ? 15 : _rng.Next(1, 6);
        bool impedanceHigh = blackImp >= 15 || whiteImp >= 15;

        return new NSMDataPacket
        {
            LocalTimestamp = DateTime.Now,
            DeviceTimeSec = _deviceTime,
            ElectrodeAlarm = false,
            ImpedanceHigh = impedanceHigh,
            ElectrodeInvalid = false,
            BlackImpedance = blackImp,
            WhiteImpedance = whiteImp,
            CSI = csi,
            BS = bs,
            SQI = sqi,
            EMG = emg,
            NOX = nox,
            EventNumber = eventNumber,
            EventType = eventType,
            AlarmHigh = 60,
            AlarmLow = 40,
            EEGSamplesUv = eeg,
            DeltaPowerDb = delta,
            ThetaPowerDb = theta,
            AlphaPowerDb = alpha,
            BetaPowerDb = beta,
            GammaPowerDb = gamma,
            SEF95 = (int)Clamp(8 + csi * 0.3 + Noise(2), 1, 44),
            EOG = _rng.Next(0, 30),
        };
    }

    private double Noise(double scale) => (_rng.NextDouble() - 0.5) * 2 * scale;
    private static double Clamp(double v, double lo, double hi) => Math.Max(lo, Math.Min(hi, v));

    public void Dispose() => Disconnect();
}
