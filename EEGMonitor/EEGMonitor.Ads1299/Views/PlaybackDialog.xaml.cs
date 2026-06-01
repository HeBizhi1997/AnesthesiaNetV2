using Ads1299Monitor.ViewModels;
using System.Windows;
using System.Windows.Controls;

namespace Ads1299Monitor.Views;

public partial class PlaybackDialog : Window
{
    private readonly PlaybackViewModel _vm;

    public PlaybackDialog(string recordingsRoot)
    {
        InitializeComponent();
        _vm = new PlaybackViewModel(recordingsRoot);
        DataContext = _vm;
        Closed += (_, _) => _vm.Dispose();
    }

    private void SpeedBox_Changed(object sender, SelectionChangedEventArgs e)
    {
        if (_vm != null && SpeedBox.SelectedItem is ComboBoxItem it &&
            double.TryParse(it.Tag?.ToString(), out var s))
            _vm.Speed = s;
    }
}
