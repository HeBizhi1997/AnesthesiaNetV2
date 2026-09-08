namespace NSMMonitor.Models;

/// <summary>
/// AFE4490 单帧解析结果（协议 serial-protocol.md §4，17 字节帧）。
/// IR / RED 是 22 位 Σ-Δ ADC 的环境光扣除值，符号扩展为 int32，范围 −2 097 152 ～ +2 097 151。
/// </summary>
public readonly record struct PpgFrame(DateTime Timestamp, int Ir, int Red, byte DeviceSpo2, byte DeviceHr)
{
    private const int FullScale = 2_097_151;

    /// <summary>模拟前端饱和（探头未佩戴 / 环境光过强 / 增益过高），见协议 §4.3。</summary>
    /// <remarks>用 long 取绝对值：帧校验不约束 IR/RED 这 8 个字节，理论上可解出 int.MinValue，
    /// 而 Math.Abs(int.MinValue) 会抛 OverflowException。</remarks>
    public bool IsSaturated => Math.Abs((long)Ir) >= FullScale - 512 || Math.Abs((long)Red) >= FullScale - 512;

    /// <summary>SPI 读取异常（MISO 悬空，恒 −1）或传感器无响应（恒 0），见协议 §4.3。</summary>
    public bool IsSensorFault => (Ir == -1 && Red == -1) || (Ir == 0 && Red == 0);
}

/// <summary>
/// 由 PPG 波形导出的一组生命体征，约每秒产出一次。
/// SpO₂ 与脉率均由上位机计算（协议 v1.1 中设备侧字段恒为 0）；
/// 若模组是老固件、会回填偏移 13/14，则一并带出 <see cref="DeviceSpo2"/> / <see cref="DeviceHr"/> 供交叉核对。
/// </summary>
public sealed record PpgMetrics(
    DateTime Time,
    double Spo2,              // 血氧饱和度 %（NaN = 无法计算或灌注不足）
    double PulseRate,         // 脉率 bpm（NaN = 间期不足或灌注不足）
    double Prv,               // 脉率变异性 RMSSD ms（NaN = 同上）
    double PerfusionIndex,    // 灌注指数 %，无论是否达标都如实给出，便于判断探头佩戴
    double RRatio,            // 比值 R，SpO₂ 的中间量，排障用
    int DeviceSpo2,           // 设备直出血氧（0 = 无）
    int DeviceHr,             // 设备直出脉率（0 = 无）
    bool LowPerfusion);       // true = 灌注低于阈值，上面几个派生值已被置为 NaN
