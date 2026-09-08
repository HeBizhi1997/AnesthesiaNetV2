using Microsoft.Extensions.Logging;
using NSMMonitor.Configuration;
using NSMMonitor.Models;
using Timer = System.Timers.Timer;

namespace NSMMonitor.Services;

/// <summary>
/// PPG 模拟器：无指夹模组时生成合成的 IR / RED 光电容积波，用于演示与界面联调。
///
/// 与真实模组一致的地方：125 Hz、int32 量级、设备侧 SpO₂/HR 字段恒为 0（协议 v1.1 由上位机计算）。
/// 波形用两个高斯叠加近似真实 PPG 的收缩波 + 重搏波，并叠加呼吸性基线漂移与噪声。
/// RED / IR 的交流直流比按 R ≈ 0.545 设定，经默认二次标定曲线约得 SpO₂ ≈ 98%。
///
/// 每约 90 秒模拟一次手术刺激：心率上升 + 脉搏波幅下降（血管收缩），
/// 好让 SPI 有真实的上行反应可看，而不是恒定在 50。
/// </summary>
public sealed class PpgSimulatorService : IPpgDataSource, IDisposable
{
    private const double DcIr = 60_000, DcRed = 45_000;
    private const double AcIrFrac = 0.020;      // 灌注指数约 2%
    private const double TargetR = 0.545;       // → SpO₂ ≈ 98%（默认二次曲线）

    private readonly ILogger<PpgSimulatorService> _logger;
    private readonly int _fs;
    private readonly Random _rng = new();

    private Timer? _timer;
    private bool _running;
    private long _frames;
    private long _framesAtLastTick;
    private double _frameRate;

    private DateTime _startedAt;
    private long _emitted;          // 已发出的样本数，用于按墙钟时间对齐发帧节奏
    private double _beatPhase;      // 当前心搏内的相位 [0,1)
    private double _hr = 72;        // 当前瞬时心率
    private double _hrBase = 72;
    private double _ampScale = 1.0; // 脉搏波幅缩放（血管收缩时下降）
    private double _stimulus;       // 刺激衰减量 [0,1]
    private double _elapsedSec;

    public bool IsConnected => _running;
    public string SourceName => "内置模拟器";
    public double FrameRate => _frameRate;
    public long FramesDecoded => Interlocked.Read(ref _frames);
    public long BadFrames => 0;

    public event Action<PpgFrame>? FrameReceived;
    public event Action<string>? StatusChanged;
    public event Action<Exception>? ErrorOccurred;

    public PpgSimulatorService(ILogger<PpgSimulatorService> logger, NsmConfig config)
    {
        _logger = logger;
        _fs = config.Ppg.SampleRate > 0 ? config.Ppg.SampleRate : 125;
    }

    public IEnumerable<string> GetAvailablePorts() => new[] { "SIM" };

    public bool Connect(string portName = "SIM", int baudRate = 57600)
    {
        if (_running) Disconnect();

        Interlocked.Exchange(ref _frames, 0);
        _framesAtLastTick = 0;
        _frameRate = 0;
        _emitted = 0;
        _beatPhase = 0;
        _hrBase = 72;
        _ampScale = 1.0;
        _stimulus = 0;
        _elapsedSec = 0;
        _startedAt = DateTime.UtcNow;

        // 40 ms 一批（约 5 帧）。发帧数按墙钟时间推算，故平均速率严格等于 fs，不受定时器抖动影响。
        _timer = new Timer(40) { AutoReset = true };
        _timer.Elapsed += OnTick;
        _running = true;
        _timer.Start();

        _logger.LogInformation("PPG 模拟器已启动 @ {Fs} Hz", _fs);
        StatusChanged?.Invoke("血氧模拟器已启动");
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
        _frameRate = 0;
        _logger.LogInformation("PPG 模拟器已停止");
        StatusChanged?.Invoke("血氧模拟器已停止");
    }

    private void OnTick(object? sender, System.Timers.ElapsedEventArgs e)
    {
        if (!_running) return;
        try
        {
            double elapsed = (DateTime.UtcNow - _startedAt).TotalSeconds;
            long due = (long)(elapsed * _fs);
            int n = (int)Math.Min(due - _emitted, _fs);   // 单批上限 1 秒，防休眠唤醒后暴发
            if (n <= 0) return;

            for (int k = 0; k < n; k++)
            {
                var f = NextSample();
                Interlocked.Increment(ref _frames);
                FrameReceived?.Invoke(f);
            }
            _emitted += n;

            // 帧率必须除以实际窗口时长：定时器 40 ms 一跳，跨过 1 秒时窗口实际是 ~1.04 s，
            // 直接拿计数当速率会报出 130 帧/秒这种虚高值。
            double dt = elapsed - _elapsedSec;
            if (dt >= 1.0)
            {
                long total = FramesDecoded;
                _frameRate = (total - _framesAtLastTick) / dt;
                _elapsedSec = elapsed;
                _framesAtLastTick = total;
            }
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(ex);
        }
    }

    private PpgFrame NextSample()
    {
        double t = _emitted / (double)_fs;

        // ── 手术刺激：约每 90 秒来一次，心率↑ + 波幅↓，之后指数衰减回基线 ──
        if (_rng.NextDouble() < 1.0 / (90.0 * _fs)) _stimulus = 1.0;
        _stimulus *= 1.0 - 1.0 / (25.0 * _fs);           // 时间常数约 25 秒

        _hrBase += (_rng.NextDouble() - 0.5) * 0.02;      // 基线缓慢游走
        _hrBase = Math.Clamp(_hrBase, 62, 82);
        _hr = _hrBase + 26 * _stimulus + (_rng.NextDouble() - 0.5) * 1.2;   // 末项 = 搏动间变异，PRV 由此而来
        _ampScale = 1.0 - 0.45 * _stimulus;

        // ── 推进心搏相位 ──
        _beatPhase += _hr / 60.0 / _fs;
        if (_beatPhase >= 1.0) _beatPhase -= 1.0;

        double wave = PulseShape(_beatPhase) * _ampScale;

        // 呼吸性基线漂移（约 0.25 Hz，即 15 次/分）
        double resp = 0.004 * Math.Sin(2 * Math.PI * 0.25 * t);

        double acIr = DcIr * AcIrFrac * wave;
        double acRed = DcRed * AcIrFrac * TargetR * wave;

        double ir = DcIr * (1 + resp) + acIr + Noise(DcIr * 0.00025);
        double red = DcRed * (1 + resp) + acRed + Noise(DcRed * 0.00025);

        // 协议 v1.1：偏移 13/14 恒为 0，SpO₂ 与心率均由上位机计算
        return new PpgFrame(DateTime.Now, (int)Math.Round(ir), (int)Math.Round(red), 0, 0);
    }

    /// <summary>
    /// 单个心搏的归一化波形：收缩波（窄高斯）+ 重搏波（宽而低的高斯），
    /// 去均值后峰峰值约为 1，形状接近真实指端 PPG。
    /// </summary>
    private static double PulseShape(double phase)
    {
        double systolic = Math.Exp(-Math.Pow((phase - 0.24) / 0.105, 2));
        double dicrotic = 0.42 * Math.Exp(-Math.Pow((phase - 0.52) / 0.155, 2));
        return systolic + dicrotic - 0.32;   // 减去近似均值，使交流分量围绕 0 摆动
    }

    private double Noise(double scale) => (_rng.NextDouble() - 0.5) * 2 * scale;

    public void Dispose() => Disconnect();
}
