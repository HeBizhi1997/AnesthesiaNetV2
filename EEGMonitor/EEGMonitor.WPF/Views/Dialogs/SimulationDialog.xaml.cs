using EEGMonitor.Services.Simulation;
using Microsoft.Win32;
using System.Windows;

namespace EEGMonitor.Views.Dialogs;

public partial class SimulationDialog : Window
{
    private readonly IVitalSimulatorService _simulator;
    private VitalFileInfo? _fileInfo;
    private bool _loaded;

    public SimulationDialog(IVitalSimulatorService simulator)
    {
        InitializeComponent();
        _simulator = simulator;

        // Wire progress events
        _simulator.PositionChanged += OnPositionChanged;
        _simulator.SimulationCompleted += OnSimulationCompleted;

        // Populate fixed options
        SpeedCombo.ItemsSource = new[] { "0.25×", "0.5×", "1×", "2×", "4×", "8×" };
        SpeedCombo.SelectedIndex = 2;

        SampleRateCombo.ItemsSource = new[] { 128, 256, 512 };
        SampleRateCombo.SelectedItem = 256;

        // If simulator already running, restore UI state
        if (_simulator.IsRunning)
            SetRunningState();
    }

    // ── File browsing ─────────────────────────────────────────────────────────

    private void BrowseClick(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "选择 VitalDB Vital 文件",
            Filter = "Vital Files (*.vital)|*.vital|All Files (*.*)|*.*",
            Multiselect = false,
        };
        if (dlg.ShowDialog() == true)
        {
            FilePathBox.Text = dlg.FileName;
            ResetState();
        }
    }

    // ── Query tracks (fast) ───────────────────────────────────────────────────

    private async void QueryTracksClick(object sender, RoutedEventArgs e)
    {
        var path = FilePathBox.Text.Trim();
        if (string.IsNullOrEmpty(path)) { ShowError("请先选择 .vital 文件。"); return; }
        if (!System.IO.File.Exists(path)) { ShowError("文件不存在。"); return; }

        FileInfoText.Text = "正在查询轨道...";
        SetBusy(true);
        try
        {
            _fileInfo = await _simulator.QueryFileAsync(path);
        }
        catch (Exception ex)
        {
            ShowError($"查询失败：{ex.Message}");
            SetBusy(false);
            return;
        }
        SetBusy(false);

        // Display info
        var dur = TimeSpan.FromSeconds(_fileInfo.DurationSeconds);
        FileInfoText.Text = $"时长: {dur:hh\\:mm\\:ss}  |  轨道数: {_fileInfo.Tracks.Count}";

        // Populate track dropdowns
        var waveformTracks = _fileInfo.Tracks.Where(t => t.IsWaveform).Select(t => t.Name).ToList();
        var allTracks = _fileInfo.Tracks.Select(t => t.Name).ToList();

        void Populate(System.Windows.Controls.ComboBox cb, List<string> items, string recommended)
        {
            cb.ItemsSource = items;
            cb.SelectedItem = items.Contains(recommended) ? recommended : items.FirstOrDefault();
        }

        Populate(Eeg1Combo, waveformTracks, _fileInfo.RecommendedEeg1);
        Populate(Eeg2Combo, waveformTracks, _fileInfo.RecommendedEeg2);
        Populate(PulseCombo, waveformTracks, _fileInfo.RecommendedPulse);
        Populate(Spo2Combo, allTracks, _fileInfo.RecommendedSpo2);
        Populate(HrCombo, allTracks, _fileInfo.RecommendedHr);

        LoadBtn.IsEnabled = true;
    }

    // ── Load (resample into Python memory) ───────────────────────────────────

    private async void LoadFileClick(object sender, RoutedEventArgs e)
    {
        LoadBtn.IsEnabled = false;
        StartBtn.IsEnabled = false;
        ProgressLabel.Text = "正在加载文件（数据重采样中，请稍候...）";
        SetBusy(true);

        var cfg = BuildConfig();
        try
        {
            await _simulator.LoadAsync(cfg);
        }
        catch (Exception ex)
        {
            ShowError($"加载失败：{ex.Message}");
            LoadBtn.IsEnabled = true;
            SetBusy(false);
            ProgressLabel.Text = "加载失败";
            return;
        }

        SetBusy(false);
        _loaded = true;
        ProgressLabel.Text = "加载完成，可以开始模拟";
        StartBtn.IsEnabled = true;
        LoadBtn.IsEnabled = true;

        UpdateTimeText(TimeSpan.Zero);
    }

    // ── Playback controls ─────────────────────────────────────────────────────

    private void StartClick(object sender, RoutedEventArgs e)
    {
        if (!_loaded) return;
        _simulator.PlaybackSpeed = ParseSpeed();
        _simulator.Start();
        SetRunningState();
    }

    private void PauseClick(object sender, RoutedEventArgs e)
    {
        if (_simulator.IsRunning)
        {
            _simulator.Pause();
            PauseBtn.Content = "▶ 继续";
        }
        else
        {
            _simulator.Resume();
            PauseBtn.Content = "⏸ 暂停";
        }
    }

    private void StopClick(object sender, RoutedEventArgs e)
    {
        _simulator.Stop();
        SetIdleState();
        ProgressBar.Value = 0;
        ProgressLabel.Text = "已停止";
        UpdateTimeText(TimeSpan.Zero);
    }

    // ── Progress callbacks ────────────────────────────────────────────────────

    private void OnPositionChanged(TimeSpan pos)
    {
        Dispatcher.Invoke(() =>
        {
            UpdateTimeText(pos);
            if (_simulator.TotalDuration.TotalSeconds > 0)
                ProgressBar.Value = pos.TotalSeconds / _simulator.TotalDuration.TotalSeconds * 100;
        });
    }

    private void OnSimulationCompleted()
    {
        Dispatcher.Invoke(() =>
        {
            SetIdleState();
            ProgressBar.Value = 100;
            ProgressLabel.Text = "文件播放完毕";
        });
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private SimulationConfig BuildConfig() => new()
    {
        FilePath = FilePathBox.Text.Trim(),
        Eeg1Track = Eeg1Combo.SelectedItem?.ToString() ?? "BIS/EEG1_WAV",
        Eeg2Track = Eeg2Combo.SelectedItem?.ToString() ?? "BIS/EEG2_WAV",
        PulseTrack = PulseCombo.SelectedItem?.ToString() ?? "SNUADC/PLETH",
        Spo2Track = Spo2Combo.SelectedItem?.ToString() ?? "Solar8000/PLETH_SPO2",
        HrTrack = HrCombo.SelectedItem?.ToString() ?? "Solar8000/HR",
        TargetSampleRate = SampleRateCombo.SelectedItem is int r ? r : 256,
        PlaybackSpeed = ParseSpeed(),
    };

    private double ParseSpeed() => SpeedCombo.SelectedItem?.ToString()?.TrimEnd('×') switch
    {
        "0.25" => 0.25, "0.5" => 0.5, "2" => 2.0, "4" => 4.0, "8" => 8.0, _ => 1.0
    };

    private void UpdateTimeText(TimeSpan pos)
    {
        var total = _simulator.TotalDuration;
        TimeText.Text = $"{pos:hh\\:mm\\:ss} / {total:hh\\:mm\\:ss}";
    }

    private void SetRunningState()
    {
        StartBtn.IsEnabled = false;
        PauseBtn.IsEnabled = true;
        StopBtn.IsEnabled = true;
        PauseBtn.Content = "⏸ 暂停";
        ProgressLabel.Text = "模拟运行中...";
    }

    private void SetIdleState()
    {
        StartBtn.IsEnabled = _loaded;
        PauseBtn.IsEnabled = false;
        StopBtn.IsEnabled = false;
        PauseBtn.Content = "⏸ 暂停";
    }

    private void ResetState()
    {
        _fileInfo = null;
        _loaded = false;
        LoadBtn.IsEnabled = false;
        StartBtn.IsEnabled = false;
        PauseBtn.IsEnabled = false;
        StopBtn.IsEnabled = false;
        FileInfoText.Text = "— 请先选择文件并查询轨道 —";
        ProgressBar.Value = 0;
        ProgressLabel.Text = "准备就绪";
    }

    private void SetBusy(bool busy) => IsEnabled = !busy;

    private static void ShowError(string msg) =>
        MessageBox.Show(msg, "模拟器", MessageBoxButton.OK, MessageBoxImage.Warning);

    protected override void OnClosed(EventArgs e)
    {
        _simulator.PositionChanged -= OnPositionChanged;
        _simulator.SimulationCompleted -= OnSimulationCompleted;
        _simulator.Stop();
        base.OnClosed(e);
    }
}
