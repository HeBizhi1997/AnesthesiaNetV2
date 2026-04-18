using EEGMonitor.Models;
using EEGMonitor.Services.Processing;
using EEGMonitor.Services.Recording;
using EEGMonitor.Services.SerialPort;
using Microsoft.Extensions.Logging;
using System.Threading.Channels;

namespace EEGMonitor.Infrastructure.Pipeline;

/// <summary>
/// Multi-stage async pipeline:
///   Source → [raw Channel] → Chunking → [chunk Channel] → Processing → [result Channel] → UI + Recording
///
/// Sources can be:
///   • Real serial port  (Start / Stop)
///   • Vital-file simulator (InjectSample)
/// Both paths converge at the raw Channel so the same stages handle both.
/// </summary>
public sealed class DataPipeline : IAsyncDisposable
{
    private const int CHUNK_SIZE = 256; // 1 second @ 256 Hz

    private readonly ISerialPortService _serial;
    private readonly IEEGProcessingClient _processing;
    private readonly IRecordingService _recording;
    private readonly ILogger<DataPipeline> _logger;

    private readonly Channel<EEGSample> _rawChannel;
    private readonly Channel<EEGDataChunk> _chunkChannel;
    private readonly Channel<ProcessedEEGResult> _resultChannel;

    private readonly CancellationTokenSource _cts = new();
    private readonly List<Task> _tasks = new();
    private bool _stagesStarted;

    public event Action<ProcessedEEGResult>? ResultAvailable;

    public DataPipeline(
        ISerialPortService serial,
        IEEGProcessingClient processing,
        IRecordingService recording,
        ILogger<DataPipeline> logger)
    {
        _serial = serial;
        _processing = processing;
        _recording = recording;
        _logger = logger;

        _rawChannel = Channel.CreateBounded<EEGSample>(new BoundedChannelOptions(4096)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = false,
        });

        _chunkChannel = Channel.CreateBounded<EEGDataChunk>(new BoundedChannelOptions(32)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = true,
        });

        _resultChannel = Channel.CreateUnbounded<ProcessedEEGResult>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true,
        });
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>Start pipeline stages + subscribe to the real serial port.</summary>
    public void Start()
    {
        EnsureStagesStarted();
        _serial.SampleReceived += OnSampleReceived;
        _logger.LogInformation("Data pipeline connected to serial port");
    }

    /// <summary>Detach serial port (stages keep running for simulation injection).</summary>
    public void Stop()
    {
        _serial.SampleReceived -= OnSampleReceived;
    }

    /// <summary>
    /// Inject a sample directly into the pipeline from an external source
    /// (e.g. VitalSimulatorService). Starts stages on first call.
    /// </summary>
    public void InjectSample(EEGSample sample)
    {
        EnsureStagesStarted();
        _rawChannel.Writer.TryWrite(sample);
        _ = _recording.RecordRawSampleAsync(sample);
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private void EnsureStagesStarted()
    {
        if (_stagesStarted) return;
        _stagesStarted = true;
        _tasks.Add(Task.Run(ChunkingStageAsync));
        _tasks.Add(Task.Run(ProcessingStageAsync));
        _tasks.Add(Task.Run(ResultDispatchStageAsync));
        _logger.LogInformation("Pipeline stages started");
    }

    private void OnSampleReceived(EEGSample sample)
    {
        _rawChannel.Writer.TryWrite(sample);
        _ = _recording.RecordRawSampleAsync(sample);
    }

    // Stage 1 – buffer individual samples into 1-second epoch chunks
    private async Task ChunkingStageAsync()
    {
        var buffer = new List<EEGSample>(CHUNK_SIZE);
        try
        {
            await foreach (var sample in _rawChannel.Reader.ReadAllAsync(_cts.Token))
            {
                buffer.Add(sample);
                if (buffer.Count >= CHUNK_SIZE)
                {
                    await _chunkChannel.Writer.WriteAsync(new EEGDataChunk
                    {
                        StartTime = buffer[0].Timestamp,
                        EndTime = buffer[^1].Timestamp,
                        Samples = buffer.ToList(),
                        SampleRate = 256,
                        ChannelCount = buffer[0].Channels.Length,
                    }, _cts.Token);
                    buffer.Clear();
                }
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            _chunkChannel.Writer.TryComplete();
            _logger.LogDebug("Chunking stage finished");
        }
    }

    // Stage 2 – send chunks to Python service, receive processed results
    private async Task ProcessingStageAsync()
    {
        try
        {
            await foreach (var chunk in _chunkChannel.Reader.ReadAllAsync(_cts.Token))
            {
                ProcessedEEGResult result;
                try
                {
                    result = await _processing.ProcessChunkAsync(chunk, _cts.Token);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Processing stage error – skipping chunk");
                    continue;
                }
                await _resultChannel.Writer.WriteAsync(result, _cts.Token);
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            _resultChannel.Writer.TryComplete();
            _logger.LogDebug("Processing stage finished");
        }
    }

    // Stage 3 – dispatch results to recording and UI
    private async Task ResultDispatchStageAsync()
    {
        try
        {
            await foreach (var result in _resultChannel.Reader.ReadAllAsync(_cts.Token))
            {
                await _recording.RecordResultAsync(result);
                ResultAvailable?.Invoke(result);
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            _logger.LogDebug("Result dispatch stage finished");
        }
    }

    public async ValueTask DisposeAsync()
    {
        _cts.Cancel();
        Stop();
        await Task.WhenAll(_tasks).ContinueWith(_ => { }).ConfigureAwait(false);
        _cts.Dispose();
        _logger.LogInformation("Data pipeline disposed");
    }
}
