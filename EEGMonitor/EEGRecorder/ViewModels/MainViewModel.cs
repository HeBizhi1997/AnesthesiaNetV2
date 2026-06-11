using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EEGRecorder.Configuration;
using EEGRecorder.Controls;
using EEGRecorder.Dsp;
using EEGRecorder.Services;

namespace EEGRecorder.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly RecorderConfig _cfg;
    private readonly Ads1299SerialService _eeg = new();
    private readonly PpgSerialService _ppg = new();
    private readonly RawRecorder _recorder = new();
    private readonly DispatcherTimer _ui;

    private EegDisplayFilter _eegFilter;
    private readonly PpgDisplayFilter _ppgFilter = new(125);
    private readonly WaveformSource _rawRing = new(256);   // raw µV for electrode-quality std

    private long _lastEegTicks, _lastPpgTicks;
    private DateTime _recStarted;
    private volatile byte _devSpo2, _devHr;

    // Live waveforms (bound to WaveformView.Source).
    public WaveformSource EegSource { get; }
    public WaveformSource PpgSource { get; }

    // ── state ──
    [ObservableProperty] private bool _isConnected;
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _connectButtonText = "连接设备";
    [ObservableProperty] private string _recordButtonText = "● 开始录制";
    [ObservableProperty] private string _statusLine = "未连接 — 编辑 appsettings.json 设置串口后点“连接设备”";
    [ObservableProperty] private string _nowText = "";

    // ── live readouts ──
    [ObservableProperty] private string _eegInfo = "EEG: --";
    [ObservableProperty] private string _ppgInfo = "PPG: --";
    [ObservableProperty] private string _deviceHr = "--";
    [ObservableProperty] private string _deviceSpo2 = "--";
    [ObservableProperty] private string _electrodeStatus = "--";
    [ObservableProperty] private bool _electrodeBad;
    [ObservableProperty] private bool _noDataVisible;
    [ObservableProperty] private string _noDataText = "";

    // ── recording ──
    [ObservableProperty] private string _recordElapsed = "00:00:00";
    [ObservableProperty] private string _sessionPath = "--";
    [ObservableProperty] private string _freeSpace = "--";
    [ObservableProperty] private string _eegFileSize = "--";
    [ObservableProperty] private string _ppgFileSize = "--";

    // ── session (editable; written into the folder name + meta) ──
    [ObservableProperty] private string _subject;
    [ObservableProperty] private string _note;
    [ObservableProperty] private string _outputDir;
    [ObservableProperty] private string _portSummary;

    public MainViewModel()
    {
        _cfg = RecorderConfig.Load();
        _eegFilter = new EegDisplayFilter(_cfg.Serial.EegSampleRate);
        EegSource = new WaveformSource(_cfg.Serial.EegSampleRate * 10);   // 10 s window
        PpgSource = new WaveformSource(125 * 8);                          // ~8 s window

        _subject = _cfg.Session.Subject;
        _note = _cfg.Session.Note;
        _outputDir = _cfg.Recording.OutputDirectory;
        _portSummary = $"EEG {_cfg.Serial.EegPort}@{_cfg.Serial.EegBaud} · PPG {(_cfg.Serial.PpgEnabled ? _cfg.Serial.PpgPort : "关")}";

        _eeg.EegBatch += OnEegBatch;
        _eeg.Status += s => Post(() => StatusLine = s);
        _ppg.PpgBatch += OnPpgBatch;
        _ppg.Status += s => Post(() => StatusLine = s);

        _ui = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _ui.Tick += (_, _) => OnUiTick();
        _ui.Start();
    }

    // ── connect / disconnect ──
    [RelayCommand]
    private void ToggleConnect()
    {
        if (IsConnected) Disconnect();
        else Connect();
    }

    private void Connect()
    {
        _eegFilter = new EegDisplayFilter(_cfg.Serial.EegSampleRate);
        _eegFilter.Reset(); _ppgFilter.Reset();
        EegSource.Clear(); PpgSource.Clear(); _rawRing.Clear();
        _lastEegTicks = _lastPpgTicks = 0;

        bool eegOk = _eeg.Connect(_cfg.Serial.EegPort, _cfg.Serial.EegBaud, _cfg.Serial.EegSampleRate);
        bool ppgOk = !_cfg.Serial.PpgEnabled || _ppg.Connect(_cfg.Serial.PpgPort, _cfg.Serial.PpgBaud);

        IsConnected = eegOk || ppgOk;
        ConnectButtonText = IsConnected ? "断开" : "连接设备";
        ToggleRecordCommand.NotifyCanExecuteChanged();
        if (!eegOk) StatusLine = $"EEG {_cfg.Serial.EegPort} 打开失败 — 检查串口/电源";
    }

    private void Disconnect()
    {
        if (IsRecording) StopRecording();
        _eeg.Disconnect();
        if (_ppg.IsConnected) _ppg.Disconnect();
        IsConnected = false;
        ConnectButtonText = "连接设备";
        NoDataVisible = false;
        ToggleRecordCommand.NotifyCanExecuteChanged();
    }

    // ── record / stop ──
    private bool CanRecord() => IsConnected;

    [RelayCommand(CanExecute = nameof(CanRecord))]
    private void ToggleRecord()
    {
        if (IsRecording) StopRecording();
        else StartRecording();
    }

    private void StartRecording()
    {
        // Persist the latest typed subject/note into the folder name + meta.
        _cfg.Session.Subject = string.IsNullOrWhiteSpace(Subject) ? "S001" : Subject.Trim();
        _cfg.Session.Note = Note ?? "";
        var dir = _recorder.Start(_cfg);
        _recStarted = DateTime.Now;
        IsRecording = true;
        RecordButtonText = "■ 停止录制";
        SessionPath = dir;
        StatusLine = $"录制中 → {Path.GetFileName(dir)}";
    }

    private void StopRecording()
    {
        _recorder.Stop();
        IsRecording = false;
        RecordButtonText = "● 开始录制";
        StatusLine = SessionPath == "--" ? "已停止" : $"已保存 → {SessionPath}";
    }

    [RelayCommand]
    private void OpenFolder()
    {
        var target = _recorder.SessionDir ?? OutputDir;
        try { if (Directory.Exists(target)) Process.Start(new ProcessStartInfo("explorer.exe", $"\"{target}\"")); }
        catch { /* ignore */ }
    }

    // ── data (serial threads) ──
    private void OnEegBatch(float[] ch0)
    {
        if (IsRecording) _recorder.WriteEeg(ch0);
        foreach (var v in ch0)
        {
            _rawRing.Push(v);
            EegSource.Push(_eegFilter.Process(v));
        }
        _lastEegTicks = DateTime.Now.Ticks;
    }

    private void OnPpgBatch(IReadOnlyList<PpgFrame> frames)
    {
        if (IsRecording) _recorder.WritePpg(frames);
        foreach (var f in frames) PpgSource.Push(_ppgFilter.Process(f.Ir));
        var last = frames[^1];
        _devSpo2 = last.Spo2; _devHr = last.Hr;
        _lastPpgTicks = DateTime.Now.Ticks;
    }

    // ── UI tick (250 ms) ──
    private void OnUiTick()
    {
        NowText = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

        if (IsConnected)
        {
            EegInfo = $"EEG {_eeg.PortName}  {_eeg.DataFrames} 帧 · {_eeg.TotalBytes / 1024} KB · {Measured(_eeg)} Hz";
            PpgInfo = _cfg.Serial.PpgEnabled
                ? $"PPG {_ppg.PortName}  {_ppg.FramesIn} 帧"
                : "PPG 关闭";
            DeviceHr = _devHr > 0 ? _devHr.ToString() : "--";
            DeviceSpo2 = _devSpo2 > 0 ? _devSpo2.ToString() : "--";
            UpdateElectrode();

            double sinceEeg = (DateTime.Now.Ticks - _lastEegTicks) / (double)TimeSpan.TicksPerSecond;
            bool noData = _eeg.IsConnected && (_lastEegTicks == 0 || sinceEeg > 2.0);
            if (noData != NoDataVisible)
            {
                NoDataVisible = noData;
                NoDataText = noData ? "⚠ EEG 串口已打开但无数据 — 检查采集设备电源 / USB" : "";
            }
        }

        if (IsRecording)
        {
            RecordElapsed = (DateTime.Now - _recStarted).ToString(@"hh\:mm\:ss");
            EegFileSize = $"EEG {_recorder.EegSamples * 4 / 1024.0 / 1024.0:0.0} MB ({_recorder.EegSamples} 采样)";
            PpgFileSize = $"PPG {_recorder.PpgFrames * 18 / 1024.0:0} KB ({_recorder.PpgFrames} 帧)";
        }
        UpdateFreeSpace();
    }

    private static string Measured(Ads1299SerialService s) =>
        s.SamplesEmitted > 0 ? "~" + s.SampleRate : "--";

    private void UpdateElectrode()
    {
        var buf = new double[_rawRing.Capacity];
        int n = _rawRing.Snapshot(buf);
        if (n < 16) { ElectrodeStatus = "等待数据…"; ElectrodeBad = false; return; }
        double mean = 0; for (int i = 0; i < n; i++) mean += buf[i]; mean /= n;
        double var = 0; for (int i = 0; i < n; i++) { double d = buf[i] - mean; var += d * d; } var /= n;
        double std = Math.Sqrt(var);
        if (std > 1500) { ElectrodeStatus = "导联脱落 / 未接电极"; ElectrodeBad = true; }
        else if (std < 0.5) { ElectrodeStatus = "无信号(短路/断开)"; ElectrodeBad = true; }
        else { ElectrodeStatus = $"信号正常 (σ≈{std:0} µV)"; ElectrodeBad = false; }
    }

    private void UpdateFreeSpace()
    {
        try
        {
            var root = Path.GetPathRoot(Path.GetFullPath(OutputDir));
            if (root != null) FreeSpace = $"{new DriveInfo(root).AvailableFreeSpace / 1024.0 / 1024 / 1024:0.0} GB";
        }
        catch { FreeSpace = "--"; }
    }

    public void Shutdown()
    {
        try { if (IsRecording) _recorder.Stop(); } catch { }
        try { _eeg.Dispose(); } catch { }
        try { _ppg.Dispose(); } catch { }
    }

    private static void Post(Action a)
    {
        var d = Application.Current?.Dispatcher;
        if (d == null || d.CheckAccess()) a(); else d.BeginInvoke(a);
    }
}
