using NSMMonitor.Models;

namespace NSMMonitor.Services;

/// <summary>
/// NSM 数据源抽象：真实串口与内置模拟器实现相同接口，
/// 供 ViewModel 统一订阅，无需关心数据来源。
/// </summary>
public interface INsmDataSource
{
    bool IsConnected { get; }
    string SourceName { get; }
    int SampleRate { get; }
    long PacketsDecoded { get; }

    event Action<NSMDataPacket>? NSMDataReceived;
    event Action<string>? StatusChanged;
    event Action<Exception>? ErrorOccurred;

    IEnumerable<string> GetAvailablePorts();
    bool Connect(string portName, int baudRate = 115200);
    void Disconnect();
}
