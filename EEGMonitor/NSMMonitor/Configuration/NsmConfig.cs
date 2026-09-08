using System.IO;
using System.Text.Json;

namespace NSMMonitor.Configuration;

/// <summary>
/// appsettings.json 的强类型视图（文件放在 exe 旁边，改完重启即可生效，无需重新编译）。
/// 载入失败时全部回落到默认值，并把原因记在 <see cref="LoadError"/> 里供启动时打日志——
/// 否则用户改坏一个逗号，界面会毫无反应地继续用默认权重。
/// </summary>
public sealed class NsmConfig
{
    public NsmDeviceConfig Nsm { get; set; } = new();
    public PpgConfig Ppg { get; set; } = new();
    public Spo2Config Spo2 { get; set; } = new();
    public SpiConfig Spi { get; set; } = new();
    public FusionConfig Fusion { get; set; } = new();
    public RecordingConfig Recording { get; set; } = new();

    /// <summary>非空表示读配置出错，已回落默认值。</summary>
    public string? LoadError { get; private set; }

    public static NsmConfig Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
        try
        {
            if (!File.Exists(path))
                return new NsmConfig { LoadError = $"未找到 {path}，使用内置默认值" };

            var opts = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                ReadCommentHandling = JsonCommentHandling.Skip,
                AllowTrailingCommas = true,
            };
            return JsonSerializer.Deserialize<NsmConfig>(File.ReadAllText(path), opts) ?? new();
        }
        catch (Exception ex)
        {
            return new NsmConfig { LoadError = $"解析 {path} 失败（已回落默认值）：{ex.Message}" };
        }
    }
}

public sealed class NsmDeviceConfig
{
    /// <summary>脑电串口。留空表示不预选，由界面下拉框选择。</summary>
    public string Port { get; set; } = "";
    public int Baud { get; set; } = 115200;
}

/// <summary>AFE4490 指夹血氧模组，协议见 serial-protocol.md。</summary>
public sealed class PpgConfig
{
    public bool Enabled { get; set; } = true;
    /// <summary>血氧串口。留空表示不预选。</summary>
    public string Port { get; set; } = "";
    public int Baud { get; set; } = 57600;
    /// <summary>标称采样率，实际值由收帧节奏连续测量修正。</summary>
    public int SampleRate { get; set; } = 125;

    /// <summary>
    /// 灌注指数下限（%）。低于此值判为无有效脉动，SpO₂ / 脉搏 / SPI / 融合一律显示 "--"。
    ///
    /// 存在的意义是安全：探头脱落时交流分量全是噪声，比值 R 照样能算出个数，
    /// 经标定曲线映射后会得到一个<b>看起来合理的低氧读数</b>（实测探头脱开时显示过 SpO₂ 79%），
    /// 这比显示 "--" 危险得多 —— 可能触发不必要的临床处置。商用血氧仪在此情形下
    /// 显示"探头脱落"或"低灌注"而不给数，这里沿用同样的做法。
    ///
    /// 0.5% 是商用设备常见下限；置 0 可关闭门控（不建议）。
    /// </summary>
    public double MinPerfusionIndex { get; set; } = 0.5;

    /// <summary>
    /// 严格帧校验：按协议 §6.1 校验全部 5 个固定字节（偏移 2/3/4/15/16）。
    /// 实测本模组固件的帧尾恒为 {0x00, 0x0B}（见 .ino 的 DataPacketFooter），严格校验成立、坏帧为 0。
    /// 若换了把偏移 15 当状态字节用的固件，严格校验会把每帧判坏 —— 服务会在状态栏提示，届时设为 false。
    /// </summary>
    public bool StrictFrameCheck { get; set; } = true;

    /// <summary>
    /// 是否优先采用设备直出的 SpO₂ / 脉率（偏移 13/14），默认 <c>false</c>（用上位机计算值）。
    ///
    /// <b>实测结论：本模组这两个字段恒为 0，根本拿不到值。</b>
    /// 2026-07-19 用 COM8 做过两次 45 秒采集（5172 帧、5624 帧，坏帧均为 0，手指佩戴良好、
    /// 上位机同期能稳定算出 SpO₂ 95–96%），偏移 13/14 全程未出现过任何非零值。
    ///
    /// 对照固件源码，.ino 只在 <c>buffer_count_overflow</c> 为真时才写这两个字节，
    /// 且 <c>spo2 == -999</c>（算法判无效）时写 0；结合实测，可以认为
    /// <c>estimate_spo2</c> 在本模组的信号条件下始终返回 -999。可能的原因：
    /// S_afe44xx.cpp 里 <c>if (dec == 20)</c> 把 125 Hz 抽成约 6 Hz 后再攒 100 点，
    /// 心搏在这条 6 Hz 序列上每周期只剩约 5 个点，波谷检测难以成立；
    /// 且入算法前的 <c>(uint16_t)(IR_data &gt;&gt; 4)</c> 在 IR &gt; 约 1.05×10⁶ 时会溢出回绕。
    /// 未做进一步固件级验证，故仅作推测记录。
    ///
    /// 另外 S_spo2_algorithm.cpp:168 用红光的峰值下标去取红外的交流分量
    /// （<c>an_x[n_y_dc_max_idx]</c>，应为 <c>n_x_dc_max_idx</c>）—— Maxim 参考实现流传已久的 bug，
    /// 我们的逐搏实现各通道各用自己的峰值下标，未沿用该缺陷。
    ///
    /// 保留本开关只为将来换固件时能一键切回设备口径；当前置 true 只会让读数变成 "--"。
    /// </summary>
    public bool PreferDeviceValues { get; set; } = false;
}

/// <summary>
/// SpO₂ 标定曲线。默认 <c>table</c>，即模组固件自带的 184 项查找表
/// （AFE4490_library/src/S_spo2_algorithm.cpp），与设备直出值同口径。
/// <b>该曲线是 Maxim 通用参考实现，并非针对本模组的控制脱氧标定，不能当诊断依据。</b>
/// </summary>
public sealed class Spo2Config
{
    /// <summary>table（默认，厂商查找表）/ quadratic（厂商解析式）/ linear（粗略近似）。</summary>
    public string Mode { get; set; } = "table";

    // SpO₂ = QuadA·R² + QuadB·R + QuadC  —— 固件 S_spo2_algorithm.cpp:196 注释给出的等价式
    public double QuadA { get; set; } = -45.06;
    public double QuadB { get; set; } = 30.354;
    public double QuadC { get; set; } = 94.845;

    // SpO₂ = LinearA − LinearB·R   （与 scripts/ppg_demo.py 一致）
    public double LinearA { get; set; } = 110.0;
    public double LinearB { get; set; } = 25.0;

    public double Min { get; set; } = 50;
    public double Max { get; set; } = 100;
}

/// <summary>
/// SPI（手术容积脉搏波指数）= 100 − (PpgaWeight·PPGAnorm + HbiWeight·HBInorm)。
/// 文献存在 0.7/0.3 与 0.67/0.33 两种系数，故放出来可配。
/// </summary>
public sealed class SpiConfig
{
    public double PpgaWeight { get; set; } = 0.67;
    public double HbiWeight { get; set; } = 0.33;
    /// <summary>百分位归一化的滚动窗口长度（秒）。指标约每秒一个，故等于样本数。</summary>
    public int WindowSeconds { get; set; } = 300;

    private double Total => PpgaWeight + HbiWeight;
    public double NormalizedPpga => Total > 0 ? PpgaWeight / Total : 0.67;
    public double NormalizedHbi => Total > 0 ? HbiWeight / Total : 0.33;
}

/// <summary>录制设置。</summary>
public sealed class RecordingConfig
{
    /// <summary>直接接入真实 NSM 串口时是否自动开始录制（模拟器/回放不录）。默认 true。</summary>
    public bool AutoRecord { get; set; } = true;

    /// <summary>录制目录。留空则用 exe 旁的 Recordings/ 子目录。</summary>
    public string Directory { get; set; } = "";

    /// <summary>是否把 125Hz 原始脉搏波（IR+RED）一并写入 .nsm。接了脉搏仪才有数据，无则该字段为 null。</summary>
    public bool IncludePulseWave { get; set; } = true;
}

/// <summary>CSI ⊕ SPI 融合指数。两者同向（越高＝麻醉越浅／伤害感受越强），故加权平均语义自洽。</summary>
public sealed class FusionConfig
{
    public bool Enabled { get; set; } = true;
    public string DisplayName { get; set; } = "融合指数";
    public double CsiWeight { get; set; } = 0.6;
    public double SpiWeight { get; set; } = 0.4;

    private double Total => CsiWeight + SpiWeight;
    // 权重和不为 1 时自动归一，配置写 3:2 或 60:40 都能算对
    public double NormalizedCsi => Total > 0 ? CsiWeight / Total : 0.5;
    public double NormalizedSpi => Total > 0 ? SpiWeight / Total : 0.5;

    /// <summary>界面上显示的权重说明，如 "0.6·CSI + 0.4·SPI"。</summary>
    public string WeightLabel => $"{NormalizedCsi:0.##}·CSI + {NormalizedSpi:0.##}·SPI";
}
