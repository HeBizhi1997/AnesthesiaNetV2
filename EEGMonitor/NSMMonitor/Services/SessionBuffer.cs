using NSMMonitor.Models;

namespace NSMMonitor.Services;

/// <summary>
/// 整场会话的内存缓冲：实时监护与文件回放共用同一套累积逻辑，供「整场报告」一次性成图。
///
/// 数据在 <c>MainViewModel.Apply()</c>（脑电包）与 <c>ApplyPpgMetrics()</c>（血氧指标）里顺手追加，
/// 因此实时与回放自动一致 —— 回放本质上就是把同一条数据流重放一遍。
/// 每次连接调用 <see cref="Reset"/> 清空。
///
/// 内存量级：脑电 100Hz×4h ≈ 144 万个 float ≈ 6 MB；指标每秒一条，几 KB；
/// 脉搏波 125Hz×4h ≈ 180 万个 float ≈ 7 MB。整场几十 MB，可接受。
/// 报告成图时用 <see cref="Downsample"/> 抽稀，不把百万点直接丢给 OxyPlot。
/// </summary>
public sealed class SessionBuffer
{
    private readonly object _lock = new();

    // 时间基准：第一帧的墙钟时间；其余时间点都换算成"距开始的秒数"。
    private DateTime _t0;
    private bool _started;

    public string SourceName { get; private set; } = "";
    public bool IsPlayback { get; private set; }

    // ── 脑电原始波形（整场）：值 + 平行的"手术计时"秒数，两者等长 ──
    // 存真实时间戳而非假定 1/fs 间隔，是为了让脑电与趋势/DSA/血氧共用同一条时间轴（否则事件竖线对不齐）。
    private readonly List<float> _eeg = new(1 << 20);
    private readonly List<float> _eegT = new(1 << 20);
    private double _eegLastT;

    // ── 逐包麻醉指标（约 1 Hz）。Bs = 爆发抑制比 0–100 ──
    public readonly record struct TrendSample(double T, double Csi, double Spi, double Nox, double Fusion, double Sef95, double Bs);
    private readonly List<TrendSample> _trend = new(4096);

    // ── 频带占比（逐包，dB 已转线性归一）──
    public readonly record struct BandSample(double T, double Delta, double Theta, double Alpha, double Beta, double Gamma);
    private readonly List<BandSample> _bands = new(4096);

    // ── DSA 频谱列（整场）──
    private readonly List<byte[]> _dsa = new(4096);

    // ── 血氧指标（约 1 Hz）──
    public readonly record struct VitalSample(double T, double Spo2, double Pr, double Pi);
    private readonly List<VitalSample> _vitals = new(4096);

    // ── 脉搏波（去基线交流分量，抽样后连续拼接）──
    private readonly List<float> _pulse = new(1 << 20);
    private double _pulseSampleRate = 125;

    // ── 事件（时间点 + 标签）──
    public readonly record struct EventMark(double T, string Label);
    private readonly List<EventMark> _events = new();

    public void Reset(string sourceName, bool isPlayback)
    {
        lock (_lock)
        {
            _started = false;
            SourceName = sourceName;
            IsPlayback = isPlayback;
            _eeg.Clear();
            _eegT.Clear();
            _eegLastT = 0;
            _trend.Clear();
            _bands.Clear();
            _dsa.Clear();
            _vitals.Clear();
            _pulse.Clear();
            _events.Clear();
        }
    }

    private double Elapsed(DateTime ts)
    {
        if (!_started) { _t0 = ts; _started = true; }
        return (ts - _t0).TotalSeconds;
    }

    // ─────────────────────────── 追加 ───────────────────────────

    /// <summary>
    /// 追加本包的脑电样本，并把它们等距铺在"上一块结束时间 → 本包 elapsed"这段真实区间上，
    /// 使脑电时间轴与趋势/DSA/血氧一致（修复各面板时间基准不一致导致的事件对不齐）。
    /// </summary>
    public void AddEegSamples(IReadOnlyList<double> samples, DateTime ts)
    {
        int n = samples.Count;
        if (n == 0) return;
        lock (_lock)
        {
            double tEnd = Elapsed(ts);
            double tStart = _eeg.Count == 0 ? Math.Max(0, tEnd - n / 100.0) : _eegLastT;
            if (tEnd <= tStart) tEnd = tStart + n / 100.0;   // 时间戳异常时退回名义间隔，保证单调
            double dt = (tEnd - tStart) / n;
            for (int i = 0; i < n; i++)
            {
                _eeg.Add((float)samples[i]);
                _eegT.Add((float)(tStart + dt * i));
            }
            _eegLastT = tEnd;
        }
    }

    public void AddTrend(DateTime ts, double csi, double spi, double nox, double fusion, double sef95, double bs)
    {
        lock (_lock) _trend.Add(new TrendSample(Elapsed(ts), csi, spi, nox, fusion, sef95, bs));
    }

    public void AddBands(DateTime ts, double d, double t, double a, double b, double g)
    {
        lock (_lock) _bands.Add(new BandSample(Elapsed(ts), d, t, a, b, g));
    }

    public void AddDsa(byte[] column)
    {
        if (column.Length == 0) return;
        lock (_lock) _dsa.Add(column);
    }

    public void AddVital(DateTime ts, double spo2, double pr, double pi)
    {
        lock (_lock) _vitals.Add(new VitalSample(Elapsed(ts), spo2, pr, pi));
    }

    public void AddPulse(IReadOnlyList<double> acSamples, double sampleRate)
    {
        lock (_lock)
        {
            _pulseSampleRate = sampleRate > 0 ? sampleRate : _pulseSampleRate;
            for (int i = 0; i < acSamples.Count; i++) _pulse.Add((float)acSamples[i]);
        }
    }

    /// <summary>追加单个脉搏波交流样本（VM 已按抽稀率喂入，故这里 sampleRate 是抽稀后的等效率）。</summary>
    public void AddPulseSample(double ac, double sampleRate)
    {
        lock (_lock)
        {
            _pulseSampleRate = sampleRate > 0 ? sampleRate : _pulseSampleRate;
            _pulse.Add((float)ac);
        }
    }

    public void AddEvent(DateTime ts, string label)
    {
        lock (_lock) _events.Add(new EventMark(Elapsed(ts), label));
    }

    // ─────────────────────────── 读取快照（成图用）───────────────────────────

    /// <summary>整场概况，供报告头部显示。</summary>
    public readonly record struct Summary(
        string SourceName, bool IsPlayback, double DurationSec,
        int TrendCount, int VitalCount, int EventCount, bool HasPulse, bool HasDsa);

    public Summary GetSummary()
    {
        lock (_lock)
        {
            double dur = 0;
            if (_trend.Count > 0) dur = Math.Max(dur, _trend[^1].T);
            if (_vitals.Count > 0) dur = Math.Max(dur, _vitals[^1].T);
            if (_events.Count > 0) dur = Math.Max(dur, _events[^1].T);
            return new Summary(SourceName, IsPlayback, dur,
                _trend.Count, _vitals.Count, _events.Count, _pulse.Count > 0, _dsa.Count > 0);
        }
    }

    public List<TrendSample> GetTrend() { lock (_lock) return new List<TrendSample>(_trend); }
    public List<VitalSample> GetVitals() { lock (_lock) return new List<VitalSample>(_vitals); }
    public List<EventMark> GetEvents() { lock (_lock) return new List<EventMark>(_events); }

    /// <summary>整场平均频带占比（δ/θ/α/β/γ，和为 100）。</summary>
    public double[] GetAverageBands()
    {
        lock (_lock)
        {
            if (_bands.Count == 0) return new double[5];
            double d = 0, t = 0, a = 0, b = 0, g = 0;
            foreach (var s in _bands) { d += s.Delta; t += s.Theta; a += s.Alpha; b += s.Beta; g += s.Gamma; }
            int n = _bands.Count;
            return new[] { d / n, t / n, a / n, b / n, g / n };
        }
    }

    /// <summary>DSA 频谱矩阵拷贝（列 × 44 频段），可能很宽，报告里按需抽列。</summary>
    public (byte[][] Columns, int Bins) GetDsa()
    {
        lock (_lock)
        {
            int bins = _dsa.Count > 0 ? _dsa[0].Length : 0;
            return (_dsa.ToArray(), bins);
        }
    }

    public double PulseSampleRate { get { lock (_lock) return _pulseSampleRate; } }

    public List<BandSample> GetBands() { lock (_lock) return new List<BandSample>(_bands); }

    // ─────────────────────────── 整场 KPI（临床回顾头部）───────────────────────────

    /// <summary>
    /// 整场关键指标。占比按"逐包计时加权"——每个样本代表到下一样本的时间段，
    /// 故用相邻样本时间差累加，长手术里少数异常包不会被高估。
    /// </summary>
    public readonly record struct Kpi(
        double DurationSec,
        double CsiTargetPct, double CsiDeepPct, double CsiLightPct,   // 靶区40-60 / 过深<40 / 偏浅>60
        double MeanBsr, double SuppressionSec,                        // 平均爆发抑制比、累计抑制时长
        double Spo2Min, int DesatEvents, double DesatSec,             // 血氧最低、低氧(<90)事件数与时长
        double PrMin, double PrMax, double PrMean,
        double SpiTargetPct,                                          // SPI 靶区20-50 占比
        int EventCount);

    public Kpi GetKpi()
    {
        lock (_lock)
        {
            double dur = _trend.Count > 0 ? _trend[^1].T : 0;

            // CSI 占比（逐包计时加权）
            double wTarget = 0, wDeep = 0, wLight = 0, wCsiTotal = 0;
            double bsrSum = 0; int bsrN = 0; double suppSec = 0;
            for (int i = 0; i < _trend.Count; i++)
            {
                double dt = i + 1 < _trend.Count ? _trend[i + 1].T - _trend[i].T : 1.0;
                if (dt <= 0 || dt > 30) dt = 1.0;   // 时间戳异常时退回 1 秒
                double csi = _trend[i].Csi;
                if (double.IsFinite(csi))
                {
                    wCsiTotal += dt;
                    if (csi < 40) wDeep += dt; else if (csi <= 60) wTarget += dt; else wLight += dt;
                }
                double bs = _trend[i].Bs;
                if (double.IsFinite(bs)) { bsrSum += bs; bsrN++; if (bs > 0) suppSec += dt * bs / 100.0; }
            }

            double spiTarget = 0, spiTotal = 0;
            foreach (var t in _trend)
            {
                double dt = 1.0;   // SPI 逐包近似 1 秒
                if (double.IsFinite(t.Spi)) { spiTotal += dt; if (t.Spi is >= 20 and <= 50) spiTarget += dt; }
            }

            // 血氧
            double spo2Min = double.NaN, prMin = double.NaN, prMax = double.NaN, prSum = 0; int prN = 0;
            int desat = 0; double desatSec = 0; bool inDesat = false;
            for (int i = 0; i < _vitals.Count; i++)
            {
                double dt = i + 1 < _vitals.Count ? _vitals[i + 1].T - _vitals[i].T : 1.0;
                if (dt <= 0 || dt > 30) dt = 1.0;
                double s = _vitals[i].Spo2, p = _vitals[i].Pr;
                if (double.IsFinite(s))
                {
                    if (double.IsNaN(spo2Min) || s < spo2Min) spo2Min = s;
                    if (s < 90) { desatSec += dt; if (!inDesat) { desat++; inDesat = true; } } else inDesat = false;
                }
                if (double.IsFinite(p))
                {
                    if (double.IsNaN(prMin) || p < prMin) prMin = p;
                    if (double.IsNaN(prMax) || p > prMax) prMax = p;
                    prSum += p; prN++;
                }
            }

            return new Kpi(dur,
                Pct(wTarget, wCsiTotal), Pct(wDeep, wCsiTotal), Pct(wLight, wCsiTotal),
                bsrN > 0 ? bsrSum / bsrN : double.NaN, suppSec,
                spo2Min, desat, desatSec,
                prMin, prMax, prN > 0 ? prSum / prN : double.NaN,
                Pct(spiTarget, spiTotal),
                _events.Count);
        }
    }

    private static double Pct(double part, double total) => total > 0 ? 100.0 * part / total : double.NaN;

    /// <summary>脑电抽稀成 min/max 包络：X 取自平行时间戳（手术计时秒），与其余面板同轴。</summary>
    public (double[] X, double[] Min, double[] Max) GetEegEnvelope(int maxColumns)
    {
        lock (_lock) return Envelope(_eeg, _eegT, maxColumns);
    }

    /// <summary>脉搏波按抽稀后的等效采样率给时间戳（脉搏波无逐样本时间戳，用等间隔近似即可）。</summary>
    public (double[] X, double[] Min, double[] Max) GetPulseEnvelope(int maxColumns)
    {
        lock (_lock)
        {
            int n = _pulse.Count;
            if (n == 0) return (Array.Empty<double>(), Array.Empty<double>(), Array.Empty<double>());
            var t = new List<float>(n);
            for (int i = 0; i < n; i++) t.Add((float)(i / _pulseSampleRate));
            return Envelope(_pulse, t, maxColumns);
        }
    }

    private static (double[] X, double[] Min, double[] Max) Envelope(List<float> data, List<float> time, int maxColumns)
    {
        int n = data.Count;
        if (n == 0) return (Array.Empty<double>(), Array.Empty<double>(), Array.Empty<double>());
        int cols = Math.Min(maxColumns, n);
        int per = Math.Max(1, n / cols);
        int outN = (n + per - 1) / per;

        var x = new double[outN];
        var lo = new double[outN];
        var hi = new double[outN];
        for (int c = 0; c < outN; c++)
        {
            int a = c * per, b = Math.Min(n, a + per);
            float mn = data[a], mx = data[a];
            for (int i = a + 1; i < b; i++) { float v = data[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
            x[c] = time[Math.Min(time.Count - 1, a + (b - a) / 2)];   // 真实手术计时秒
            lo[c] = mn; hi[c] = mx;
        }
        return (x, lo, hi);
    }

    /// <summary>等距抽稀一维序列到不超过 maxPoints 个点（趋势/血氧折线用，非尖峰信号无需 min/max）。</summary>
    public static List<T> Downsample<T>(IReadOnlyList<T> src, int maxPoints)
    {
        if (src.Count <= maxPoints) return new List<T>(src);
        var outp = new List<T>(maxPoints);
        double step = (double)src.Count / maxPoints;
        for (int i = 0; i < maxPoints; i++) outp.Add(src[(int)(i * step)]);
        return outp;
    }
}
