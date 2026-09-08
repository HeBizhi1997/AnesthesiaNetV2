namespace NSMMonitor.Models;

/// <summary>
/// 解析后的 NSM 麻醉深度监护仪数据包（353 字节 UART 协议）。
/// 字段定义见《NSM 设备通讯协议文档》。
/// </summary>
public record NSMDataPacket
{
    public DateTime LocalTimestamp { get; init; }

    // ── 设备 ──
    public uint DeviceTimeSec { get; init; }

    // ── 电极状态 ──
    public bool ElectrodeAlarm { get; init; }
    public bool ImpedanceHigh { get; init; }
    public bool ElectrodeInvalid { get; init; }
    public int BlackImpedance { get; init; }   // 1-10, 15=过高（电极 3）
    public int WhiteImpedance { get; init; }   // 1-10, 15=过高（电极 1）

    // ── 麻醉指标 ──
    /// <summary>麻醉深度指数（0-99），等效 BIS。0xEE/0xFF = 无效。</summary>
    public int CSI { get; init; }
    /// <summary>爆发抑制比（0-100）。0xFF = 无效。</summary>
    public int BS { get; init; }
    /// <summary>信号质量指数（0-100）。0xFF = 无效。</summary>
    public int SQI { get; init; }
    /// <summary>肌电信号指数（0-100）。0xFF = 无效。</summary>
    public int EMG { get; init; }
    /// <summary>伤害感受/镇痛指数（0-99）。0xFF = 无效。</summary>
    public int NOX { get; init; }

    // ── 临床事件 ──
    public int EventNumber { get; init; }
    public NSMEventType EventType { get; init; }

    // ── 报警阈值 ──
    public int AlarmHigh { get; init; }
    public int AlarmLow { get; init; }

    // ── EEG 波形（100 样本，有符号字节 → µV）──
    public double[] EEGSamplesUv { get; init; } = Array.Empty<double>();

    // ── 频带功率（dB，有符号字节）──
    public int DeltaPowerDb { get; init; }
    public int ThetaPowerDb { get; init; }
    public int AlphaPowerDb { get; init; }
    public int BetaPowerDb { get; init; }
    public int GammaPowerDb { get; init; }

    // ── 派生指标 ──
    public int SEF95 { get; init; }    // 1-44 Hz
    public int EOG { get; init; }      // 0-100

    // ── 密度谱阵列 (DSA) ──
    /// <summary>密度谱阵列：44 个频段（1-44 Hz，每 1 Hz 一格）的功率强度，0-255。
    /// 由设备直接给出（非上位机 FFT 计算），存于完整 355 字节帧偏移 132-175。
    /// 紧凑 128 字节记录格式不含此字段，长度为 0。</summary>
    public byte[] Dsa { get; init; } = Array.Empty<byte>();

    // ── 脉搏波（录制时附加，回放时重建 SpO₂/脉搏/SPI）──
    /// <summary>本包时段的红外脉搏波交流分量（自上一包累积，约 125 个/包），存前已减去 <see cref="IrDc"/>。
    /// 无脉搏仪或老录制文件时为 null。回放用 AC+DC 还原绝对值喂处理器，与实时同一条计算链。</summary>
    public short[]? PulseWaveIr { get; init; }
    /// <summary>对应红光通道交流分量（减去 <see cref="RedDc"/>）。SpO₂ 需 IR+RED 两路才能算。</summary>
    public short[]? PulseWaveRed { get; init; }
    /// <summary>IR 通道本包直流基线（均值）。比值 R = (AC_red/DC_red)/(AC_ir/DC_ir) 需要它。</summary>
    public int IrDc { get; init; }
    /// <summary>RED 通道本包直流基线（均值）。</summary>
    public int RedDc { get; init; }

    // ── 有效性 ──
    public bool CSIValid => CSI != 0xEE && CSI != 0xFF;
    public bool BSValid => BS != 0xFF;
    public bool SQIValid => SQI != 0xFF;
    public bool EMGValid => EMG != 0xFF;
    public bool NOXValid => NOX != 0xFF;
}

public enum NSMEventType : byte
{
    General = 0,
    Induction = 1,
    Intubation = 2,
    Maintenance = 3,
    Surgery = 4,
    Injection = 5,
    Note = 6,
    EndMaintenance = 7,
    Movement = 8,
}
