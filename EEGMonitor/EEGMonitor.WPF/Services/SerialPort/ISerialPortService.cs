using EEGMonitor.Models;

namespace EEGMonitor.Services.SerialPort;

public interface ISerialPortService : IDisposable
{
    bool IsConnected { get; }
    string PortName { get; }
    int BaudRate { get; }

    /// <summary>Effective EEG sample rate (Hz) of the connected device — drives pipeline chunking / resampling.</summary>
    int SampleRate { get; }

    event Action<EEGSample>? SampleReceived;
    event Action<string>? ConnectionStatusChanged;
    event Action<Exception>? ErrorOccurred;

    bool Connect(string portName, int baudRate, int channelCount = 1, int sampleRate = 500);
    void Disconnect();
    IEnumerable<string> GetAvailablePorts();
}
