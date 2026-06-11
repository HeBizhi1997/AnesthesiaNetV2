using System.Windows;
using EEGRecorder.ViewModels;

namespace EEGRecorder.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Closing += (_, _) => (DataContext as MainViewModel)?.Shutdown();
    }
}
