using NSMMonitor.Models;

namespace NSMMonitor.Services;

/// <summary>
/// PPG 数据源抽象：真实串口与内置模拟器实现相同接口。
/// 只负责产出原始帧，SpO₂ / 脉率 / SPI 的计算由 <see cref="Ppg.PulseOximetryProcessor"/> 承担，
/// 这样模拟器不必重复实现一遍算法。
///
/// 与 <see cref="INsmDataSource"/> 完全独立：两台设备各自开串口、各自连断，互不牵连。
/// </summary>
public interface IPpgDataSource
{
    bool IsConnected { get; }
    string SourceName { get; }

    /// <summary>实测有效帧率（帧/秒）。协议 §6.3：正常 125±2，持续 &lt;120 说明丢帧。</summary>
    double FrameRate { get; }
    long FramesDecoded { get; }
    /// <summary>坏帧计数。协议 §6.3：长期为 0 或极低；持续增长说明波特率不符或线路干扰。</summary>
    long BadFrames { get; }

    event Action<PpgFrame>? FrameReceived;
    event Action<string>? StatusChanged;
    event Action<Exception>? ErrorOccurred;

    IEnumerable<string> GetAvailablePorts();
    bool Connect(string portName, int baudRate = 57600);
    void Disconnect();
}
