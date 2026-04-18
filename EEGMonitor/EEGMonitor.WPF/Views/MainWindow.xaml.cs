using EEGMonitor.ViewModels;
using System.Windows;

namespace EEGMonitor.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Closed += OnMainWindowClosed;
    }

    private void OnMainWindowClosed(object? sender, EventArgs e)
    {
        (DataContext as MainViewModel)?.Cleanup();
    }
}
