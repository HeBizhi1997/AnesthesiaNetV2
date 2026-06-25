using System.Windows;
using NSMMonitor.ViewModels;
using NSMMonitor.Views;

namespace NSMMonitor;

public partial class MainWindow : Window
{
    private EegComponentsWindow? _componentsWindow;

    public MainWindow()
    {
        InitializeComponent();
    }

    private void OpenEegComponents_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        // 已打开则激活，避免重复弹窗
        if (_componentsWindow is { IsVisible: true })
        {
            _componentsWindow.Activate();
            return;
        }

        _componentsWindow = new EegComponentsWindow(vm) { Owner = this };
        _componentsWindow.Closed += (_, _) => _componentsWindow = null;
        _componentsWindow.Show();
    }
}
