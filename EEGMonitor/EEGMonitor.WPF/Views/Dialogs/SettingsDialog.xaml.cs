using System.Windows;

namespace EEGMonitor.Views.Dialogs;

public partial class SettingsDialog : Window
{
    public SettingsDialog()
    {
        InitializeComponent();
        SampleRateCombo.ItemsSource = new[] { 128, 250, 256, 500, 512, 1000, 1024 };
        SampleRateCombo.SelectedItem = 256;
        ChannelCombo.ItemsSource = Enumerable.Range(1, 16).ToList();
        ChannelCombo.SelectedItem = 4;
    }

    private void SaveClick(object sender, RoutedEventArgs e)
    {
        // In a real app, persist to appsettings.json or user preferences
        MessageBox.Show("Settings saved.", "EEG Monitor", MessageBoxButton.OK, MessageBoxImage.Information);
        DialogResult = true;
    }

    private void CancelClick(object sender, RoutedEventArgs e) => DialogResult = false;
}
