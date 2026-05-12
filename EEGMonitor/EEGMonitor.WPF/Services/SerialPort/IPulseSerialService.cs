namespace EEGMonitor.Services.SerialPort;

/// <summary>
/// Interface for the HKG-07D pulse rate sensor (19200 baud, passive push, 5-byte frames).
/// </summary>
public interface IPulseSerialService : IDisposable
{
    bool IsConnected { get; }
    string PortName { get; }

    /// <summary>Fired every ~9 s when the sensor pushes a new BPM. BPM==0 means no signal.</summary>
    event Action<int>? BpmReceived;
    event Action<string>? ConnectionStatusChanged;
    event Action<Exception>? ErrorOccurred;

    bool Connect(string portName);
    void Disconnect();
}
