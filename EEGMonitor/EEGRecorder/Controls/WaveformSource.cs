namespace EEGRecorder.Controls;

/// <summary>Lock-free-ish ring buffer the serial threads push into and the WaveformView snapshots on
/// the render tick. A short lock keeps the copy consistent; cost is negligible at these rates.</summary>
public sealed class WaveformSource
{
    private readonly double[] _buf;
    private int _head;          // next write index
    private int _count;
    private readonly object _gate = new();

    public WaveformSource(int capacity) => _buf = new double[capacity];

    public void Push(double v)
    {
        lock (_gate)
        {
            _buf[_head] = v;
            _head = (_head + 1) % _buf.Length;
            if (_count < _buf.Length) _count++;
        }
    }

    public void PushRange(ReadOnlySpan<float> vs)
    {
        lock (_gate)
        {
            foreach (var v in vs)
            {
                _buf[_head] = v;
                _head = (_head + 1) % _buf.Length;
                if (_count < _buf.Length) _count++;
            }
        }
    }

    public void Clear() { lock (_gate) { _head = 0; _count = 0; } }

    /// <summary>Copy the buffered samples into <paramref name="dst"/> in chronological order.
    /// Returns the number copied.</summary>
    public int Snapshot(double[] dst)
    {
        lock (_gate)
        {
            int n = Math.Min(_count, dst.Length);
            int start = (_head - n + _buf.Length) % _buf.Length;
            for (int i = 0; i < n; i++) dst[i] = _buf[(start + i) % _buf.Length];
            return n;
        }
    }

    public int Capacity => _buf.Length;
}
