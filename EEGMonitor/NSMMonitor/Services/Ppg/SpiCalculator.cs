using NSMMonitor.Configuration;

namespace NSMMonitor.Services.Ppg;

/// <summary>
/// SPI（Surgical Pleth Index，手术容积脉搏波指数）—— 由 PPG 导出的伤害感受 / 镇痛平衡指标。
///
///   SPI = 100 − (w_ppga · PPGAnorm + w_hbi · HBInorm)
///
/// 其中 PPGA 为脉搏波幅（此处用灌注指数 PI 代表），HBI 为心搏间期（60000/PR，毫秒）。
/// 两者先各自按滚动百分位秩归一到 0–100，即 SPI 原始文献所称的"直方图变换"。
/// 交感兴奋时血管收缩（波幅↓）且心率上升（间期↓），两个归一值同时下降，SPI 因而上升。
///
/// 出处：Huiku et al., Assessment of surgical stress during general anaesthesia, BJA 2007。
/// GE 未公开其归一化的具体实现，滚动百分位秩是公开文献中的通行近似。
/// 默认权重 0.67/0.33 可在 appsettings.json 调整（文献亦有 0.7/0.3 版本）。
///
/// 临床区间：20–50 为镇痛适当；&gt;50 提示伤害感受上升（镇痛不足）；&lt;20 提示镇痛偏深。
/// </summary>
public sealed class SpiCalculator
{
    /// <summary>历史样本不足时返回中性值，避免开机瞬间给出一个看似确定的极端 SPI。</summary>
    private const int MinSamples = 5;
    private const double NeutralRank = 50.0;

    private readonly SpiConfig _cfg;
    private readonly List<double> _ppgaRoll = new();
    private readonly List<double> _hbiRoll = new();

    public SpiCalculator(NsmConfig config) => _cfg = config.Spi;

    public void Reset()
    {
        _ppgaRoll.Clear();
        _hbiRoll.Clear();
    }

    /// <summary>
    /// 送入一组脉搏波指标，返回 SPI（0–100）。
    /// 灌注指数或脉率无效时返回 NaN —— 宁可显示"--"，也不要用陈旧值凑一个数出来。
    /// </summary>
    public double Update(double perfusionIndex, double pulseRate)
    {
        if (!double.IsFinite(perfusionIndex) || perfusionIndex <= 0) return double.NaN;
        if (!double.IsFinite(pulseRate) || pulseRate <= 0) return double.NaN;

        double ppgaNorm = PushPercentile(_ppgaRoll, perfusionIndex);
        double hbiNorm = PushPercentile(_hbiRoll, 60000.0 / pulseRate);

        double spi = 100.0 - (_cfg.NormalizedPpga * ppgaNorm + _cfg.NormalizedHbi * hbiNorm);
        return Math.Clamp(spi, 0, 100);
    }

    /// <summary>v 在滚动窗口内的百分位秩（0–100）。指标约每秒一个，故窗口长度即秒数。</summary>
    private double PushPercentile(List<double> roll, double v)
    {
        roll.Add(v);
        int cap = Math.Max(MinSamples, _cfg.WindowSeconds);
        while (roll.Count > cap) roll.RemoveAt(0);

        if (roll.Count < MinSamples) return NeutralRank;

        int below = 0;
        for (int i = 0; i < roll.Count; i++) if (roll[i] < v) below++;
        return 100.0 * below / roll.Count;
    }

    public static string ZoneText(double spi) => double.IsNaN(spi)
        ? ""
        : spi > 50 ? "镇痛不足" : spi >= 20 ? "适当" : "镇痛偏深";
}

/// <summary>
/// CSI ⊕ SPI 融合。两者同向——数值越高代表麻醉越浅 / 伤害感受越强——故加权平均语义自洽。
/// 权重来自 appsettings.json 的 Fusion 段，写 0.6/0.4 或 3/2 都可以（内部归一）。
/// </summary>
public static class IndexFusion
{
    /// <summary>任一路无效即返回 NaN：融合值必须两路都在，不能用单路冒充。</summary>
    public static double Combine(double csi, double spi, FusionConfig cfg)
    {
        if (!cfg.Enabled) return double.NaN;
        if (!double.IsFinite(csi) || !double.IsFinite(spi)) return double.NaN;
        return Math.Clamp(cfg.NormalizedCsi * csi + cfg.NormalizedSpi * spi, 0, 100);
    }

    /// <summary>融合值分档沿用 CSI 的临床习惯（40–60 适宜）。</summary>
    public static string ZoneText(double v) => double.IsNaN(v)
        ? ""
        : v < 40 ? "过深" : v <= 60 ? "适宜区间" : v < 80 ? "偏浅" : "清醒风险";
}
