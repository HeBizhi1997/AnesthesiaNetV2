using NSMMonitor.Configuration;
using NSMMonitor.Models;

namespace NSMMonitor.Services.Ppg;

/// <summary>
/// 由 IR / RED 原始光电值算出 SpO₂、脉率 PR、脉率变异性 PRV 与灌注指数 PI。
///
/// 算法与 scripts/ppg_demo.py 一致（并沿用 EEGMonitor.Ads1299 中已验证的伪迹剔除实现）：
///   1. 去基线：减去约 1 秒滑动均值（高通），再做 ~40 ms 平滑（低通），等效 0.5–5 Hz 带通
///   2. 峰检测：阈值 = 0.3×标准差，最小间距 0.33 s（≤180 bpm）
///   3. PR：搏动间期取中位数 ±30% 以内的均值 —— 漏搏会让某个间期变成 2 倍，
///      不剔除的话单个离群值就能把 PR 和 RMSSD 拉飞
///   4. PRV：仅对相邻且都被接受的间期算 RMSSD，再做 EMA 平滑
///   5. PI = 交流峰峰值(5–95 百分位) / 直流 × 100
///   6. R = (AC_red/DC_red)/(AC_ir/DC_ir)，AC 取带通后标准差、DC 取原始均值；再过标定曲线得 SpO₂
///
/// 全部用前缀和实现，单次分析 O(n)，n≈1500。
/// </summary>
public sealed class PulseOximetryProcessor
{
    private const double WindowSeconds = 12.0;   // 分析窗；越长 PR 越稳，但对变化的响应越慢
    private const int WaveDecimation = 8;        // 每 8 个样本推一次波形快照（约 15 Hz），逐点刷 UI 太频繁
    private const int WaveDisplaySamples = 500;  // 显示窗：125 Hz × 4 秒

    private readonly Spo2Config _cfg;
    private readonly int _nominalFs;
    private readonly double _minPerfusion;

    private readonly double[] _ir;
    private readonly double[] _red;
    private int _count;
    private int _sinceAnalyse;
    private int _sinceWave;

    private double _fs;
    private DateTime _t0;
    private DateTime _lastFrameTime = DateTime.Now;
    private long _framesIn;
    private double _prvEma;
    private int _lastDeviceSpo2, _lastDeviceHr;

    /// <summary>约每秒一次的指标输出。</summary>
    public event Action<PpgMetrics>? MetricsReady;
    /// <summary>约 15 Hz 的交流波形快照，供脉搏波显示。</summary>
    public event Action<double[]>? WaveformReady;

    public PulseOximetryProcessor(NsmConfig config)
    {
        _cfg = config.Spo2;
        _nominalFs = config.Ppg.SampleRate > 0 ? config.Ppg.SampleRate : 125;
        _minPerfusion = config.Ppg.MinPerfusionIndex;
        _fs = _nominalFs;
        int cap = (int)Math.Ceiling(_nominalFs * WindowSeconds);
        _ir = new double[cap];
        _red = new double[cap];
    }

    public void Reset()
    {
        _count = _sinceAnalyse = _sinceWave = 0;
        _framesIn = 0;
        _fs = _nominalFs;
        _prvEma = 0;
        _lastDeviceSpo2 = _lastDeviceHr = 0;
    }

    public void Push(PpgFrame f)
    {
        if (_framesIn == 0) _t0 = DateTime.UtcNow;
        _framesIn++;
        _lastDeviceSpo2 = f.DeviceSpo2;
        _lastDeviceHr = f.DeviceHr;
        _lastFrameTime = f.Timestamp;   // 指标时间跟帧走：回放时才能与趋势/血氧同一时间轴

        // 连续修正实测采样率：标称 125 Hz，实际受晶振与 USB 调度影响会有出入，
        // 而 PR 与采样率成正比，不修正会有系统性偏差。
        double elapsed = (DateTime.UtcNow - _t0).TotalSeconds;
        if (elapsed > 2.0 && _framesIn > 100)
        {
            double measured = _framesIn / elapsed;
            if (measured is > 80 and < 200) _fs = measured;
        }

        Append(_ir, f.Ir);
        Append(_red, f.Red);
        if (_count < _ir.Length) _count++;

        int fsi = Math.Max(1, (int)Math.Round(_fs));

        if (++_sinceWave >= WaveDecimation)
        {
            _sinceWave = 0;
            if (_count >= fsi) WaveformReady?.Invoke(RecentAc());
        }

        // 攒够 4 秒再出指标，否则间期太少、PR 抖得厉害
        if (++_sinceAnalyse >= fsi && _count >= fsi * 4)
        {
            _sinceAnalyse = 0;
            MetricsReady?.Invoke(Analyse());
        }
    }

    private void Append(double[] buf, int v)
    {
        if (_count < buf.Length) { buf[_count] = v; return; }
        Array.Copy(buf, 1, buf, 0, buf.Length - 1);
        buf[^1] = v;
    }

    /// <summary>取最近一段 IR 交流分量（去基线后）用于波形显示。</summary>
    private double[] RecentAc()
    {
        int n = Math.Min(_count, Math.Min(WaveDisplaySamples, _ir.Length));
        var seg = new double[n];
        Array.Copy(_ir, _count - n, seg, 0, n);
        return BandPass(seg, _fs);
    }

    private PpgMetrics Analyse()
    {
        int n = _count;
        double fs = _fs;

        var irRaw = new double[n];
        var redRaw = new double[n];
        Array.Copy(_ir, 0, irRaw, 0, n);
        Array.Copy(_red, 0, redRaw, 0, n);

        double dcIr = Mean(irRaw), dcRed = Mean(redRaw);
        double[] acIr = BandPass(irRaw, fs);
        double[] acRed = BandPass(redRaw, fs);

        var (pr, prv) = PulseRate(acIr, fs);

        // 灌注指数：用 5–95 百分位而非最大最小，避免单个尖峰把 PI 拉高
        double pi = 0;
        if (dcIr > 1)
        {
            var sorted = (double[])acIr.Clone();
            Array.Sort(sorted);
            double p95 = sorted[Math.Min(n - 1, (int)(n * 0.95))];
            double p05 = sorted[(int)(n * 0.05)];
            pi = (p95 - p05) / dcIr * 100.0;
        }

        // 比值 R 与 SpO₂
        double r = BeatwiseRatio(irRaw, redRaw, acIr, fs);
        if (!double.IsFinite(r))
        {
            // 逐搏法拿不到足够搏动时，退回整窗统计法（弱灌注下至少还有个数）
            double stdIr = StdDev(acIr), stdRed = StdDev(acRed);
            if (dcIr > 1 && dcRed > 1 && stdIr > 1e-9) r = (stdRed / dcRed) / (stdIr / dcIr);
        }
        double spo2 = double.IsFinite(r) && r > 0 ? ApplyCalibration(r) : double.NaN;

        // 灌注门控：探头脱落 / 佩戴不良时交流分量全是噪声，R 照样算得出，
        // 映射后是个看似合理的低氧值 —— 宁可什么都不显示，也不能给一个会误导处置的数。
        // PI 与 R 仍如实输出，用于判断"为什么没数"。
        bool lowPerfusion = !(pi >= _minPerfusion);   // 写成这样是为了让 NaN 也落进 true
        if (lowPerfusion)
        {
            spo2 = double.NaN;
            pr = double.NaN;
            prv = double.NaN;
            _prvEma = 0;      // 复位平滑，避免恢复灌注后带着脱落期间的陈旧值
        }

        return new PpgMetrics(_lastFrameTime, spo2, pr, prv, pi, r,
                              _lastDeviceSpo2, _lastDeviceHr, lowPerfusion);
    }

    /// <summary>把比值 R 映射为 SpO₂，曲线见 <see cref="Spo2Calibration"/>。</summary>
    private double ApplyCalibration(double r) => Spo2Calibration.FromRatio(r, _cfg);

    /// <summary>
    /// 逐搏法求比值 R，取各搏动的中位数。
    ///
    /// 做法与模组固件 S_spo2_algorithm.cpp 一致：以波谷切分心搏，在每个搏动内
    /// 取原始信号峰值，减去两端波谷之间的线性插值基线得交流分量，直流取搏动内峰值，
    /// 于是 R = (AC_red/DC_red)/(AC_ir/DC_ir)，最后对各搏动取中位数。
    ///
    /// 相比"整窗标准差 / 均值"，中位数对单次体动或漏搏免疫 —— 实测整窗法在手指静止时
    /// SpO₂ 仍会在 89–96% 之间摆动，就是被个别坏搏动带偏的。
    ///
    /// 与固件的唯一差异：固件 S_spo2_algorithm.cpp:168 用红光的峰值下标去取红外的交流分量
    /// （<c>an_x[n_y_dc_max_idx]</c>），这是 Maxim 参考实现里的老 bug，此处各通道各用自己的峰值下标。
    /// </summary>
    private static double BeatwiseRatio(double[] ir, double[] red, double[] acIr, double fs)
    {
        // 波谷 = 取反后的波峰
        var neg = new double[acIr.Length];
        for (int i = 0; i < acIr.Length; i++) neg[i] = -acIr[i];
        var valleys = FindPeaks(neg, fs);
        if (valleys.Count < 3) return double.NaN;      // 至少 2 个完整搏动才谈得上中位数

        var ratios = new List<double>(valleys.Count);
        int minSpan = Math.Max(3, (int)(fs * 0.25));

        for (int k = 0; k + 1 < valleys.Count; k++)
        {
            int a = valleys[k], b = valleys[k + 1];
            if (b - a < minSpan) continue;

            var (acIrBeat, dcIrBeat) = BeatAcDc(ir, a, b);
            var (acRedBeat, dcRedBeat) = BeatAcDc(red, a, b);
            if (acIrBeat <= 0 || acRedBeat <= 0 || dcIrBeat <= 0 || dcRedBeat <= 0) continue;

            double ratio = (acRedBeat / dcRedBeat) / (acIrBeat / dcIrBeat);
            if (ratio is > 0.02 and < 2.0) ratios.Add(ratio);   // 与固件 idx∈(2,184) 的有效区间一致
        }

        if (ratios.Count < 2) return double.NaN;
        ratios.Sort();
        int mid = ratios.Count / 2;
        return ratios.Count % 2 == 0 ? (ratios[mid - 1] + ratios[mid]) / 2 : ratios[mid];
    }

    /// <summary>单个搏动内的交流 / 直流：峰值减去两端波谷的线性插值基线，直流取峰值本身。</summary>
    private static (double ac, double dc) BeatAcDc(double[] x, int a, int b)
    {
        int peak = a;
        double max = x[a];
        for (int i = a; i <= b && i < x.Length; i++)
            if (x[i] > max) { max = x[i]; peak = i; }

        double baseline = x[a] + (x[b] - x[a]) * (peak - a) / (double)(b - a);
        return (max - baseline, max);
    }

    /// <summary>局部极大值检测：阈值 0.3×标准差，最小间距 0.33 秒（≤180 bpm）。</summary>
    private static List<int> FindPeaks(double[] v, double fs)
    {
        int n = v.Length;
        double std = StdDev(v);
        double thresh = Math.Max(std * 0.3, 1.0);
        int minDist = Math.Max(1, (int)Math.Round(fs * 0.33));

        var peaks = new List<int>();
        int lastPeak = -minDist;
        for (int k = 1; k < n - 1; k++)
        {
            if (v[k] <= thresh || v[k] < v[k - 1] || v[k] <= v[k + 1]) continue;
            if (k - lastPeak >= minDist) { peaks.Add(k); lastPeak = k; }
            else if (peaks.Count > 0 && v[k] > v[peaks[^1]]) { peaks[^1] = k; lastPeak = k; }
        }
        return peaks;
    }

    private (double pr, double prv) PulseRate(double[] ac, double fs)
    {
        int n = ac.Length;
        double std = StdDev(ac);
        double thresh = Math.Max(std * 0.3, 1.0);
        int minDist = Math.Max(1, (int)Math.Round(fs * 0.33));   // ≤ ~180 bpm

        // 局部极大值：超过阈值且高于左右邻点；间距过近时保留更高的那个
        var peaks = new List<int>();
        int lastPeak = -minDist;
        for (int k = 1; k < n - 1; k++)
        {
            if (ac[k] <= thresh || ac[k] < ac[k - 1] || ac[k] <= ac[k + 1]) continue;
            if (k - lastPeak >= minDist) { peaks.Add(k); lastPeak = k; }
            else if (peaks.Count > 0 && ac[k] > ac[peaks[^1]]) { peaks[^1] = k; lastPeak = k; }
        }

        var ibis = new List<double>(Math.Max(0, peaks.Count - 1));
        for (int k = 1; k < peaks.Count; k++)
        {
            double ms = (peaks[k] - peaks[k - 1]) / fs * 1000.0;
            if (ms is > 300 and < 2000) ibis.Add(ms);        // 生理范围 30–200 bpm
        }
        if (ibis.Count < 2) return (double.NaN, double.NaN);

        // 漏搏 / 多搏剔除：只保留中位数 ±30% 以内的间期
        var sortedIbi = new List<double>(ibis);
        sortedIbi.Sort();
        double med = sortedIbi[sortedIbi.Count / 2];
        var accepted = new bool[ibis.Count];
        double sum = 0;
        int cnt = 0;
        for (int k = 0; k < ibis.Count; k++)
        {
            if (Math.Abs(ibis[k] - med) > 0.30 * med) continue;
            accepted[k] = true;
            sum += ibis[k];
            cnt++;
        }
        double pr = cnt > 0 ? 60000.0 / (sum / cnt) : double.NaN;

        // RMSSD：仅对相邻且都被接受、且差值合理的间期对求和
        double sq = 0;
        int dn = 0;
        for (int k = 1; k < ibis.Count; k++)
        {
            if (!accepted[k] || !accepted[k - 1]) continue;
            double d = ibis[k] - ibis[k - 1];
            if (Math.Abs(d) < 200) { sq += d * d; dn++; }
        }
        double prv = double.NaN;
        if (dn >= 1)
        {
            double rmssd = Math.Sqrt(sq / dn);
            _prvEma = _prvEma <= 0 ? rmssd : 0.5 * _prvEma + 0.5 * rmssd;
            prv = _prvEma;
        }
        return (pr, prv);
    }

    /// <summary>
    /// 去基线 + 平滑，等效 0.5–5 Hz 带通。高通窗取 1 秒：更宽会漏进呼吸性漂移，
    /// 更窄会削掉心动过缓时的基波。两次滑动平均均以前缀和实现。
    /// </summary>
    private static double[] BandPass(double[] x, double fs)
    {
        int n = x.Length;
        if (n == 0) return Array.Empty<double>();

        var pref = new double[n + 1];
        for (int k = 0; k < n; k++) pref[k + 1] = pref[k] + x[k];

        int wHigh = Math.Max(3, (int)Math.Round(fs));
        var ac = new double[n];
        for (int k = 0; k < n; k++)
        {
            int lo = Math.Max(0, k - wHigh / 2), hi = Math.Min(n, k + wHigh / 2 + 1);
            ac[k] = x[k] - (pref[hi] - pref[lo]) / (hi - lo);
        }

        int wLow = Math.Max(1, (int)Math.Round(fs * 0.04));
        if (wLow <= 1) return ac;

        var pref2 = new double[n + 1];
        for (int k = 0; k < n; k++) pref2[k + 1] = pref2[k] + ac[k];
        var smooth = new double[n];
        for (int k = 0; k < n; k++)
        {
            int lo = Math.Max(0, k - wLow / 2), hi = Math.Min(n, k + wLow / 2 + 1);
            smooth[k] = (pref2[hi] - pref2[lo]) / (hi - lo);
        }
        return smooth;
    }

    private static double Mean(double[] x)
    {
        if (x.Length == 0) return 0;
        double s = 0;
        foreach (var v in x) s += v;
        return s / x.Length;
    }

    private static double StdDev(double[] x)
    {
        if (x.Length == 0) return 0;
        double m = Mean(x), s = 0;
        foreach (var v in x) { double d = v - m; s += d * d; }
        return Math.Sqrt(s / x.Length);
    }
}
