using NSMMonitor.Configuration;

namespace NSMMonitor.Services.Ppg;

/// <summary>
/// 比值 R → SpO₂ 的标定映射。
///
/// 三种模式，默认 <c>table</c>：
///   table     —— 直接照搬模组固件里的 184 项查找表（AFE4490_library/src/S_spo2_algorithm.cpp），
///                这是设备自己用的那条曲线，上位机算出来的数与设备直出值口径一致
///   quadratic —— SpO₂ = −45.060·R² + 30.354·R + 94.845
///                固件源码 S_spo2_algorithm.cpp:196 的注释里给出的等价解析式（用于与查找表对照）
///   linear    —— SpO₂ = 110 − 25·R，scripts/ppg_demo.py 用的粗略近似
///
/// 三者在正常区间内相差 1–2 个百分点。查找表是设备的真实行为，故作默认。
/// 注意：这条曲线来自 Maxim 的通用参考实现，<b>不是对本模组做控制脱氧实验标定的结果</b>，
/// 不能当诊断依据。
/// </summary>
public static class Spo2Calibration
{
    /// <summary>
    /// 固件 S_spo2_algorithm.cpp 中的 uch_spo2_table，索引为 R×100（整数）。
    /// 逐字节照抄，未做任何改动。
    /// </summary>
    private static readonly byte[] Table =
    {
         95,  95,  95,  96,  96,  96,  97,  97,  97,  97,  97,  98,  98,  98,  98,  98,  99,  99,  99,  99,
         99,  99,  99,  99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100,  99,  99,  99,  99,  99,  99,  99,  99,  98,  98,  98,  98,  98,  98,  97,  97,
         97,  97,  96,  96,  96,  96,  95,  95,  95,  94,  94,  94,  93,  93,  93,  92,  92,  92,  91,  91,
         90,  90,  89,  89,  89,  88,  88,  87,  87,  86,  86,  85,  85,  84,  84,  83,  82,  82,  81,  81,
         80,  80,  79,  78,  78,  77,  76,  76,  75,  74,  74,  73,  72,  72,  71,  70,  69,  69,  68,  67,
         66,  66,  65,  64,  63,  62,  62,  61,  60,  59,  58,  57,  56,  56,  55,  54,  53,  52,  51,  50,
         49,  48,  47,  46,  45,  44,  43,  42,  41,  40,  39,  38,  37,  36,  35,  34,  33,  31,  30,  29,
         28,  27,  26,  25,  23,  22,  21,  20,  19,  17,  16,  15,  14,  12,  11,  10,   9,   7,   6,   5,
          3,   2,   1,
    };

    /// <summary>
    /// 把比值 R 映射为 SpO₂ 百分数。R 超出标定范围时返回 NaN
    /// （固件在同样情况下写 −999 / 置无效位，我们对应显示 "--"，不外推）。
    /// </summary>
    public static double FromRatio(double r, Spo2Config cfg)
    {
        if (!double.IsFinite(r) || r <= 0) return double.NaN;

        double v;
        switch ((cfg.Mode ?? "table").Trim().ToLowerInvariant())
        {
            case "linear":
                v = cfg.LinearA - cfg.LinearB * r;
                break;

            case "quadratic":
                v = cfg.QuadA * r * r + cfg.QuadB * r + cfg.QuadC;
                break;

            default:   // table
                // 固件的索引就是 R×100 取整，有效区间 (2, 184)，与 S_spo2_algorithm.cpp:192 一致
                int idx = (int)(r * 100);
                if (idx <= 2 || idx >= Table.Length) return double.NaN;
                v = Table[idx];
                break;
        }
        return Math.Clamp(v, cfg.Min, cfg.Max);
    }

    /// <summary>界面上标注数值来源，避免把厂商通用曲线误当成临床标定值。</summary>
    public static string SourceLabel(Spo2Config cfg) =>
        (cfg.Mode ?? "table").Trim().ToLowerInvariant() switch
        {
            "linear" => "线性近似",
            "quadratic" => "厂商解析式",
            _ => "厂商曲线",
        };
}
