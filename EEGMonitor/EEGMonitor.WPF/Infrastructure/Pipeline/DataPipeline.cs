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
    // Target one-second epochs. Derived from DeviceSampleRate so chunk duration stays
    // 1 s across devices (NSM 200 Hz, ADS1299 250-2000 Hz) instead of a fixed 256-sample
    // window that would be 0.5 s at 500 Hz. Bounded for safety.
    private int ChunkSize => Math.Clamp(DeviceSampleRate, 64, 4096);

    private readonly ISerialPortService _serial;
    private readonly IEEGProcessingClient _processing;
    private readonly IRecordingService _recording;
    private readonly ILogger<DataPipeline> _logger;

    /// <summary>
    /// Sample rate of the connected EEG device. Set before calling Start().
    /// Default 256 (ADS1299). NSM devices typically use 200.
    /// </summary>
    public int DeviceSampleRate { get; set; } = 256;

    /// <summary>
    /// Latest NSM device-reported band powers (dB). Set by MainViewModel on each
    /// NSMDataReceived event. Attached to chunks for cross-validation in the Python service.
    /// Thread-safe: written by UI thread, read by ChunkingStage.
    /// </summary>
    private Dictionary<string, int>? _latestDeviceBandPowersDb;
    private readonly object _deviceBandLock = new();
    public void SetDeviceBandPowers(Dictionary<string, int>? db)
    {
        lock (_deviceBandLock) { _latestDeviceBandPowersDb = db; }
    }
    private Dictionary<string, int>? GetDeviceBandPowers()
    {
        lock (_deviceBandLock) { return _latestDeviceBandPowersDb; }
    }

    private readonly Channel<EEGSample> _rawChannel;
    private readonly Channel<EEGDataChunk> _chunkChannel;
    private readonly Channel<ProcessedEEGResult> _resultChannel;

    private readonly CancellationTokenSource _cts = new();
    private readonly List<Task> _tasks = new();
    private bool _stagesStarted;

    public event Action<ProcessedEEGResult>? ResultAvailable;

    /// <summary>Latest BPM from the pulse sensor. Set by MainViewModel when BpmReceived fires.</summary>
    public double? CurrentHeartRate { get; set; }

    /// <summary>
    /// Fired once per serial frame (64 samples) with CH1 raw-filtered data.
    /// Delivers immediate visual feedback regardless of whether the Python service is running.
    /// </summary>
    public event Action<double[]>? RawChunkAvailable;

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

        // Raw EEG chart display (same as OnSampleReceived path)
        if (sample.Channels.Length > 0)
        {
            lock (_rawDispLock)
            {
                _rawDispBuf[_rawDispIdx++] = sample.Channels[0];
                if (_rawDispIdx >= 64)
                {
                    _rawDispIdx = 0;
                    RawChunkAvailable?.Invoke((double[])_rawDispBuf.Clone());
                }
            }
        }
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

    private readonly double[] _rawDispBuf = new double[64];
    private readonly object _rawDispLock = new();
    private int _rawDispIdx;

    private void OnSampleReceived(EEGSample sample)
    {
        // Enrich with latest heart rate from pulse sensor if not already present
        var hr = CurrentHeartRate;
        if (hr.HasValue && !sample.HeartRate.HasValue)
            sample = sample with { HeartRate = hr };

        _rawChannel.Writer.TryWrite(sample);
        _ = _recording.RecordRawSampleAsync(sample);

        // Collect CH1 samples for direct raw-EEG display (no Python needed)
        if (sample.Channels.Length > 0)
        {
            lock (_rawDispLock)
            {
                _rawDispBuf[_rawDispIdx++] = sample.Channels[0];
                if (_rawDispIdx >= 64)
                {
                    _rawDispIdx = 0;
                    RawChunkAvailable?.Invoke((double[])_rawDispBuf.Clone());
                }
            }
        }
    }

    // Stage 1 – buffer individual samples into 1-second epoch chunks
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
                        DeviceBandPowersDb = GetDeviceBandPowers(),
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
