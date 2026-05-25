using EEGMonitor.AnesthesiaSleep.Models;
using EEGMonitor.Models;

namespace EEGMonitor.AnesthesiaSleep.Infrastructure.Messaging;

public class ResultBroker
{
    private readonly List<Action<ProcessedEEGResult>> _subscribers = new();
    private readonly object _lock = new();

    public event Action<SleepEpoch>? EpochCompleted;

    public IDisposable Subscribe(Action<ProcessedEEGResult> callback)
    {
        lock (_lock) _subscribers.Add(callback);
        return new Unsubscriber(() =>
        {
            lock (_lock) _subscribers.Remove(callback);
        });
    }

    public void Publish(ProcessedEEGResult result)
    {
        Action<ProcessedEEGResult>[] snapshot;
        lock (_lock) snapshot = _subscribers.ToArray();

        foreach (var subscriber in snapshot)
        {
            try { subscriber(result); }
            catch { /* one subscriber failing must not break others */ }
        }
    }

    public void PublishEpoch(SleepEpoch epoch)
    {
        EpochCompleted?.Invoke(epoch);
    }

    private sealed class Unsubscriber : IDisposable
    {
        private readonly Action _onDispose;
        public Unsubscriber(Action onDispose) => _onDispose = onDispose;
        public void Dispose() => _onDispose();
    }
}
