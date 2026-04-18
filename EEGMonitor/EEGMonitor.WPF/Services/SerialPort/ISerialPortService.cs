using EEGMonitor.Models;

namespace EEGMonitor.Services.SerialPort;

public interface ISerialPortService : IDisposable
{
    bool IsConnected { get; }
    string PortName { get; }
    int BaudRate { get; }

    event Action<EEGSample>? SampleReceived;
    event Action<string>? ConnectionStatusChanged;
    event Action<Exception>? ErrorOccurred;

    bool Connect(string portName, int baudRate, int channelCount = 4, int sampleRate = 256);
    void Disconnect();
    IEnumerable<string> GetAvailablePorts();
}
