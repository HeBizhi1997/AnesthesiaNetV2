using Ads1299Monitor.Models;
using Microsoft.Extensions.Logging;
using System.Threading.Channels;

namespace Ads1299Monitor.Services;

/// <summary>
/// Source(serial) → [raw] → 1 s chunking → [chunk] → Python inference → [result] → record + UI.
/// </summary>
public sealed class DataPipeline : IAsyncDisposable
{
    private int ChunkSize => Math.Clamp(DeviceSampleRate, 64, 4096);

    private readonly SerialPortService _serial;
    private readonly EEGProcessingClient _processing;
    private readonly RecordingService _recording;
    private readonly ILogger<DataPipeline> _logger;

    public int DeviceSampleRate { get; set; } = 250;
    public double? CurrentHeartRate { get; set; }

    private readonly Channel<EEGSample> _rawChannel;
    private readonly Channel<EEGDataChunk> _chunkChannel;
    private readonly Channel<ProcessedEEGResult> _resultChannel;

    private readonly CancellationTokenSource _cts = new();
    private readonly List<Task> _tasks = new();
    private bool _started;

    public event Action<ProcessedEEGResult>? ResultAvailable;

    public DataPipeline(SerialPortService serial, EEGProcessingClient processing,
                        RecordingService recording, ILogger<DataPipeline> logger)
    {
        _serial = serial;
        _processing = processing;
        _recording = recording;
        _logger = logger;

        _rawChannel = Channel.CreateBounded<EEGSample>(new BoundedChannelOptions(8192)
        { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true, SingleWriter = false });
        _chunkChannel = Channel.CreateBounded<EEGDataChunk>(new BoundedChannelOptions(32)
        { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true, SingleWriter = true });
        _resultChannel = Channel.CreateUnbounded<ProcessedEEGResult>(new UnboundedChannelOptions
        { SingleReader = true, SingleWriter = true });
    }

    public void Start()
    {
        EnsureStages();
        _serial.SampleReceived += OnSampleReceived;
    }

    public void Stop() => _serial.SampleReceived -= OnSampleReceived;

    private void EnsureStages()
    {
        if (_started) return;
        _started = true;
        _tasks.Add(Task.Run(ChunkingStageAsync));
        _tasks.Add(Task.Run(ProcessingStageAsync));
        _tasks.Add(Task.Run(ResultDispatchStageAsync));
    }

    private void OnSampleReceived(EEGSample sample)
    {
        var hr = CurrentHeartRate;
        if (hr.HasValue && !sample.HeartRate.HasValue)
            sample = sample with { HeartRate = hr };

        _rawChannel.Writer.TryWrite(sample);
        _recording.RecordRawEeg(sample);
    }

    private async Task ChunkingStageAsync()
    {
        var buffer = new List<EEGSample>(512);
        try
        {
            await foreach (var sample in _rawChannel.Reader.ReadAllAsync(_cts.Token))
            {
                buffer.Add(sample);
                if (buffer.Count >= ChunkSize)
                {
                    await _chunkChannel.Writer.WriteAsync(new EEGDataChunk
                    {
                        StartTime = buffer[0].Timestamp,
                        EndTime = buffer[^1].Timestamp,
                        Samples = buffer.ToList(),
                        SampleRate = DeviceSampleRate,
                        ChannelCount = buffer[0].Channels.Length,
                    }, _cts.Token);
                    buffer.Clear();
                }
            }
        }
        catch (OperationCanceledException) { }
        finally { _chunkChannel.Writer.TryComplete(); }
    }

    private async Task ProcessingStageAsync()
    {
        try
        {
            await foreach (var chunk in _chunkChannel.Reader.ReadAllAsync(_cts.Token))
            {
                try
                {
                    var result = await _processing.ProcessChunkAsync(chunk, _cts.Token);
                    await _resultChannel.Writer.WriteAsync(result, _cts.Token);
                }
                catch (Exception ex) { _logger.LogError(ex, "Processing error — skip chunk"); }
            }
        }
        catch (OperationCanceledException) { }
        finally { _resultChannel.Writer.TryComplete(); }
    }

    private async Task ResultDispatchStageAsync()
    {
        try
        {
            await foreach (var result in _resultChannel.Reader.ReadAllAsync(_cts.Token))
            {
                _recording.RecordInference(result);
                ResultAvailable?.Invoke(result);
            }
        }
        catch (OperationCanceledException) { }
    }

    public async ValueTask DisposeAsync()
    {
        _cts.Cancel();
        Stop();
        await Task.WhenAll(_tasks).ContinueWith(_ => { }).ConfigureAwait(false);
        _cts.Dispose();
    }
}
