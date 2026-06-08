using Microsoft.Extensions.Logging;
using Ports = System.IO.Ports;

namespace Ads1299Monitor.Services;

/// <summary>
/// PPG / 血氧 指夹模组 (CH340, 默认 COM7, 57600 8N1, 被动推送 ~125 Hz)。
///
/// 帧 (17 字节):
///   [0]0x0A [1]0xFA | [2]0x0A [3]0x00 [4]0x02 | [5..8]IR(int32 LE) | [9..12]RED(int32 LE)
///   | [13]SpO₂(%) | [14]HR(bpm) | [15]Status | [16]0x0B
///
/// 设备直出 SpO₂ 与 HR；脉率 <b>PR</b>、脉率变异性 <b>PRV</b>(RMSSD) 与灌注指数 <b>PI</b>
/// 由 IR 光体积描记波(带通 0.5–5Hz 基线消除 + 峰检测)实时计算 —— 算法与 scripts/ppg_demo.py 一致。
/// 约每秒聚合一次 <see cref="ReadingReceived"/>。开口即流,无需握手 (匹配厂商抓包:DTR/RTS=false)。
/// </summary>
public sealed class PpgSerialService : IDisposable
{
    public sealed record PpgReading(
        DateTime Time,
        int Spo2,        // 设备直出血氧 %       (0 = 暂无)
        int DeviceHr,    // 设备直出脉率 bpm     (0 = 暂无)
        double Pr,       // 计算脉率 bpm         (0 = 间期不足)
        double Prv,      // 脉率变异性 RMSSD ms  (0 = 间期不足)
        double Pi);      // 灌注指数 %

    private const int FRAME = 17;
    private const byte H0 = 0x0A, H1 = 0xFA, TAIL = 0x0B;

    private readonly ILogger<PpgSerialService> _logger;
    private Ports.SerialPort? _port;
    private readonly List<byte> _buffer = new(2048);
    private readonly object _lock = new();

    // Rolling IR window (~12 s @ 125 Hz) for PR / PRV / PI.
    private readonly double[] _ir = new double[1500];
    private int _irCount;
    private int _samplesSinceAnalyse;
    private int _lastSpo2, _lastHr;

    // Measured sample rate (frames arrive ~107–125 Hz; refine continuously for PR accuracy).
    private double _fs = 125.0;
    private DateTime _t0;
    private long _framesIn;
    private double _prvEma;     // smoothed PRV (kills residual 毛刺 after artifact rejection)

    public bool IsConnected => _port?.IsOpen ?? false;
    public string PortName => _port?.PortName ?? string.Empty;

    public event Action<PpgReading>? ReadingReceived;
    public event Action<string>? ConnectionStatusChanged;
    public event Action<DateTime, int, int>? RawSampleReceived;   // 每帧原始 IR/RED(供无损录制 / 离线诊断)

    public PpgSerialService(ILogger<PpgSerialService> logger) => _logger = logger;

    public bool Connect(string portName, int baud = 57600)
    {
        if (IsConnected) Disconnect();
        lock (_lock)
        {
            _buffer.Clear(); _irCount = 0; _samplesSinceAnalyse = 0;
            _framesIn = 0; _fs = 125.0; _lastSpo2 = _lastHr = 0; _prvEma = 0;
        }
        try
        {
            _port = new Ports.SerialPort(portName, baud, Ports.Parity.None, 8, Ports.StopBits.One)
            {
                ReadTimeout = 500,
                WriteTimeout = 500,
                ReadBufferSize = 16384,
                // Vendor capture (ppg_demo) opens with DTR/RTS LOW; asserting them mutes the module.
                DtrEnable = false,
                RtsEnable = false,
            };
            _port.DataReceived += OnDataReceived;
            _port.Open();
            _port.DiscardInBuffer();
            _logger.LogInformation("PPG {Port}@{Baud} opened", portName, baud);
            ConnectionStatusChanged?.Invoke($"血氧已连接 {portName}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to open PPG port {Port}", portName);
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
        ConnectionStatusChanged?.Invoke("血氧已断开");
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
            var raws = new List<(DateTime ts, int ir, int red)>(read / FRAME + 1);
            PpgReading? emit;
            lock (_lock)
            {
                for (int i = 0; i < read; i++) _buffer.Add(buf[i]);
                emit = ProcessBuffer(raws);
            }
            // Invoke handlers OUTSIDE the lock (recording I/O / UI marshaling).
            if (RawSampleReceived != null)
                foreach (var s in raws) RawSampleReceived(s.ts, s.ir, s.red);
            if (emit != null) ReadingReceived?.Invoke(emit);
        }
        catch (Exception ex) { _logger.LogWarning(ex, "Error reading PPG data"); }
    }

    /// <summary>Consume whole frames (collecting raw IR/RED); returns a reading if an analysis tick fired.</summary>
    private PpgReading? ProcessBuffer(List<(DateTime ts, int ir, int red)> raws)
    {
        PpgReading? result = null;
        int i = 0, n = _buffer.Count;
        while (i + FRAME <= n)
        {
            if (_buffer[i] == H0 && _buffer[i + 1] == H1 && _buffer[i + 16] == TAIL)
            {
                int ir  = _buffer[i + 5] | (_buffer[i + 6] << 8) | (_buffer[i + 7] << 16) | (_buffer[i + 8] << 24);
                int red = _buffer[i + 9] | (_buffer[i + 10] << 8) | (_buffer[i + 11] << 16) | (_buffer[i + 12] << 24);
                int spo2 = _buffer[i + 13];
                int hr = _buffer[i + 14];
                raws.Add((DateTime.Now, ir, red));
                var r = OnFrame(ir, spo2, hr);
                if (r != null) result = r;     // at most one tick per buffer chunk
                i += FRAME;
            }
            else i++;
        }
        if (i > 0) _buffer.RemoveRange(0, i);
        if (_buffer.Count > 4096) _buffer.Clear();
        return result;
    }

    private PpgReading? OnFrame(int ir, int spo2, int hr)
    {
        if (_framesIn == 0) _t0 = DateTime.Now;
        _framesIn++;
        _lastSpo2 = spo2; _lastHr = hr;

        // Continuously refine the true sample rate (linear factor on PR).
        double el = (DateTime.Now - _t0).TotalSeconds;
        if (el > 2.0 && _framesIn > 100)
        {
            double m = _framesIn / el;
            if (m is > 80 and < 200) _fs = m;
        }

        // Push IR into the rolling window.
        if (_irCount < _ir.Length) _ir[_irCount++] = ir;
        else { Array.Copy(_ir, 1, _ir, 0, _ir.Length - 1); _ir[^1] = ir; }

        // Analyse ~once per second, once we have ≥4 s buffered.
        int fsi = Math.Max(1, (int)Math.Round(_fs));
        if (++_samplesSinceAnalyse >= fsi && _irCount >= fsi * 4)
        {
            _samplesSinceAnalyse = 0;
            var (pr, prv, pi) = Analyse();
            return new PpgReading(DateTime.Now, _lastSpo2, _lastHr, pr, prv, pi);
        }
        return null;
    }

    /// <summary>Band-pass IR (baseline removal + smoothing ≈ 0.5–5 Hz), peak-detect → PR/PRV/PI.</summary>
    private (double pr, double prv, double pi) Analyse()
    {
        int n = _irCount;
        double fs = _fs;

        // DC + 1 s moving-average baseline (prefix sums).
        var pref = new double[n + 1];
        double dcMean = 0;
        for (int k = 0; k < n; k++) { pref[k + 1] = pref[k] + _ir[k]; dcMean += _ir[k]; }
        dcMean /= n;

        int wB = Math.Max(3, (int)Math.Round(fs));            // ~1 s high-pass
        var ac = new double[n];
        for (int k = 0; k < n; k++)
        {
            int lo = Math.Max(0, k - wB / 2), hi = Math.Min(n, k + wB / 2 + 1);
            ac[k] = _ir[k] - (pref[hi] - pref[lo]) / (hi - lo);
        }

        // Light low-pass smoothing (~40 ms) to suppress HF noise.
        int wS = Math.Max(1, (int)Math.Round(fs * 0.04));
        if (wS > 1)
        {
            var p2 = new double[n + 1];
            for (int k = 0; k < n; k++) p2[k + 1] = p2[k] + ac[k];
            var sm = new double[n];
            for (int k = 0; k < n; k++)
            {
                int lo = Math.Max(0, k - wS / 2), hi = Math.Min(n, k + wS / 2 + 1);
                sm[k] = (p2[hi] - p2[lo]) / (hi - lo);
            }
            ac = sm;
        }

        // Prominence threshold from std.
        double mean = 0; for (int k = 0; k < n; k++) mean += ac[k]; mean /= n;
        double varr = 0; for (int k = 0; k < n; k++) { double d = ac[k] - mean; varr += d * d; } varr /= n;
        double std = Math.Sqrt(varr);
        double thresh = Math.Max(std * 0.3, 1.0);
        int minDist = Math.Max(1, (int)Math.Round(fs * 0.33));    // ≤ ~180 bpm

        // Local maxima above threshold, enforcing min distance (keep the taller of a close pair).
        var peaks = new List<int>();
        int lastPeak = -minDist;
        for (int k = 1; k < n - 1; k++)
        {
            if (ac[k] > thresh && ac[k] >= ac[k - 1] && ac[k] > ac[k + 1])
            {
                if (k - lastPeak >= minDist) { peaks.Add(k); lastPeak = k; }
                else if (peaks.Count > 0 && ac[k] > ac[peaks[^1]]) { peaks[^1] = k; lastPeak = k; }
            }
        }

        // Inter-beat intervals (physiological 30–200 bpm).
        var ibis = new List<double>(peaks.Count);
        for (int k = 1; k < peaks.Count; k++)
        {
            double ms = (peaks[k] - peaks[k - 1]) / fs * 1000.0;
            if (ms is > 300 and < 2000) ibis.Add(ms);
        }

        // Artifact-robust PR / PRV. A missed or extra beat makes one IBI ≈ 2× (or ½) its
        // neighbours; left in, that single outlier produces a huge RMSSD spike (毛刺). So:
        //   • PR  = mean of IBIs within ±30 % of the median (drop missed/extra beats).
        //   • PRV = RMSSD over successive ACCEPTED pairs whose |Δ| is plausible (<200 ms),
        //           then EMA-smoothed. This is standard HRV/PRV artifact correction.
        double pr = 0, prv = 0;
        if (ibis.Count >= 2)
        {
            var sortedIbi = new List<double>(ibis); sortedIbi.Sort();
            double med = sortedIbi[sortedIbi.Count / 2];
            var accepted = new bool[ibis.Count];
            double sum = 0; int cnt = 0;
            for (int k = 0; k < ibis.Count; k++)
                if (Math.Abs(ibis[k] - med) <= 0.30 * med) { accepted[k] = true; sum += ibis[k]; cnt++; }
            if (cnt > 0) pr = 60000.0 / (sum / cnt);

            double sq = 0; int dn = 0;
            for (int k = 1; k < ibis.Count; k++)
            {
                if (!accepted[k] || !accepted[k - 1]) continue;
                double d = ibis[k] - ibis[k - 1];
                if (Math.Abs(d) < 200) { sq += d * d; dn++; }
            }
            if (dn >= 1)
            {
                double rmssd = Math.Sqrt(sq / dn);
                _prvEma = _prvEma <= 0 ? rmssd : 0.5 * _prvEma + 0.5 * rmssd;
                prv = _prvEma;
            }
        }

        // Perfusion index = AC peak-to-peak (5–95th pct, robust) / DC.
        double pi = 0;
        if (dcMean > 1)
        {
            var sorted = (double[])ac.Clone();
            Array.Sort(sorted);
            double p95 = sorted[Math.Min(n - 1, (int)(n * 0.95))];
            double p05 = sorted[(int)(n * 0.05)];
            pi = (p95 - p05) / dcMean * 100.0;
        }
        return (pr, prv, pi);
    }

    public void Dispose() => Disconnect();
}
