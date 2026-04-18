using EEGMonitor.Models;
using EEGMonitor.Services.Playback;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System.Windows;

namespace EEGMonitor.Views.Dialogs;

public partial class PlaybackWindow : Window
{
    private readonly IPlaybackService _playback;
    private bool _seeking = false;

    public PlaybackWindow(IPlaybackService playback, List<string> sessions)
    {
        InitializeComponent();
        _playback = playback;

        _playback.ResultAvailable += OnResult;
        _playback.EventReached += OnEventReached;
        _playback.PlaybackCompleted += () => Dispatcher.Invoke(() =>
        {
            PlayPauseBtn.Content = "▶ Play";
        });

        SessionCombo.ItemsSource = sessions;
        if (sessions.Count > 0) SessionCombo.SelectedIndex = 0;

        SpeedCombo.ItemsSource = new[] { "0.25×", "0.5×", "1×", "2×", "4×", "8×" };
        SpeedCombo.SelectedIndex = 2;

        InitCharts();
    }

    private PlotModel _waveModel = new();
    private PlotModel _trendModel = new();
    private readonly Queue<(double, double)> _bisTrend = new();

    private void InitCharts()
    {
        _waveModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x21, 0x26, 0x2D),
            PlotAreaBackground = OxyColors.Black,
            Title = "EEG Waveform",
        };
        _waveModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, IsAxisVisible = false });
        _waveModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, IsAxisVisible = false });
        _waveModel.Series.Add(new LineSeries
        {
            Color = OxyColor.FromRgb(0x4F, 0xC3, 0xF7), StrokeThickness = 1,
        });
        WaveView.Model = _waveModel;

        _trendModel = new PlotModel
        {
            Background = OxyColor.FromRgb(0x21, 0x26, 0x2D),
            PlotAreaBackground = OxyColors.Black,
            Title = "BIS Trend",
        };
        _trendModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom });
        _trendModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = 0, Maximum = 100 });
        _trendModel.Series.Add(new LineSeries
        {
            Color = OxyColor.FromRgb(0x58, 0xA6, 0xFF), StrokeThickness = 2,
        });
        TrendView.Model = _trendModel;
    }

    private void OnResult(ProcessedEEGResult result)
    {
        Dispatcher.Invoke(() =>
        {
            BisPlayback.Text = double.IsNaN(result.BIS) ? "---" : result.BIS.ToString("F0");
            SpO2Playback.Text = result.SpO2.HasValue ? $"{result.SpO2:F1}%" : "---";

            if (result.RawEEG.Length > 0)
            {
                var series = (LineSeries)_waveModel.Series[0];
                series.Points.Clear();
                for (int i = 0; i < result.RawEEG.Length; i++)
                    series.Points.Add(new DataPoint(i, result.RawEEG[i]));
                _waveModel.InvalidatePlot(true);
            }

            if (!double.IsNaN(result.BIS))
            {
                var elapsed = _playback.CurrentPosition.TotalSeconds;
                _bisTrend.Enqueue((elapsed, result.BIS));
                var trend = (LineSeries)_trendModel.Series[0];
                trend.Points.Clear();
                foreach (var (t, b) in _bisTrend) trend.Points.Add(new DataPoint(t, b));
                _trendModel.InvalidatePlot(true);
            }

            // Update seek slider
            if (!_seeking && _playback.TotalDuration.TotalSeconds > 0)
            {
                SeekSlider.Value = _playback.CurrentPosition.TotalSeconds /
                                   _playback.TotalDuration.TotalSeconds * 100;
                PositionText.Text = _playback.CurrentPosition.ToString(@"mm\:ss");
            }
        });
    }

    private void OnEventReached(ClinicalEvent ev)
    {
        Dispatcher.Invoke(() =>
        {
            // Visual flash on timeline – events already displayed statically
        });
    }

    private async void LoadClick(object sender, RoutedEventArgs e)
    {
        var selected = SessionCombo.SelectedItem?.ToString();
        if (string.IsNullOrEmpty(selected)) return;

        var dir = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "EEGMonitor", "Sessions", selected);

        if (!System.IO.Directory.Exists(dir)) { MessageBox.Show("Session not found."); return; }

        await _playback.LoadSessionAsync(dir);

        if (_playback.Session != null)
        {
            SessionInfoText.Text =
                $"Patient: {_playback.Session.PatientId} | Duration: {_playback.TotalDuration:mm\\:ss}";
            DurationText.Text = _playback.TotalDuration.ToString(@"mm\:ss");

            // Populate timeline with events positioned at their session offsets
            EventsTimeline.ItemsSource = _playback.Session.Events;
        }
        _bisTrend.Clear();
    }

    private async void PlayPauseClick(object sender, RoutedEventArgs e)
    {
        if (_playback.IsPlaying)
        {
            _playback.Pause();
            PlayPauseBtn.Content = "▶ Play";
        }
        else
        {
            PlayPauseBtn.Content = "⏸ Pause";
            await _playback.PlayAsync();
        }
    }

    private void StopClick(object sender, RoutedEventArgs e)
    {
        _playback.Stop();
        PlayPauseBtn.Content = "▶ Play";
        SeekSlider.Value = 0;
    }

    private void SeekStartClick(object sender, RoutedEventArgs e) =>
        _playback.Seek(TimeSpan.Zero);

    private void SeekSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_seeking || _playback.TotalDuration == TimeSpan.Zero) return;
        // Only seek when user drags (mouse capture)
        if (SeekSlider.IsMouseCaptureWithin)
        {
            var pos = TimeSpan.FromSeconds(e.NewValue / 100 * _playback.TotalDuration.TotalSeconds);
            _playback.Seek(pos);
        }
    }

    private void SpeedCombo_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        var text = SpeedCombo.SelectedItem?.ToString() ?? "1×";
        _playback.PlaybackSpeed = text switch
        {
            "0.25×" => 0.25,
            "0.5×" => 0.5,
            "2×" => 2.0,
            "4×" => 4.0,
            "8×" => 8.0,
            _ => 1.0,
        };
    }

    private void SessionCombo_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        // Nothing – load on button click
    }

    protected override void OnClosed(EventArgs e)
    {
        _playback.Stop();
        _playback.ResultAvailable -= OnResult;
        _playback.EventReached -= OnEventReached;
        base.OnClosed(e);
    }
}
