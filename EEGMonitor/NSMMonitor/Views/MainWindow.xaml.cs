using System.Windows;
using NSMMonitor.ViewModels;
using NSMMonitor.Views;

namespace NSMMonitor;

public partial class MainWindow : Window
{
    private EegComponentsWindow? _componentsWindow;
    private SessionReportWindow? _reportWindow;

    public MainWindow()
    {
        InitializeComponent();

        // 调试辅助：NSM_OPENREPORT=1 时延迟自动弹出报告窗，供无人值守截图验证。
        if (System.Environment.GetEnvironmentVariable("NSM_OPENREPORT") == "1")
        {
            var t = new System.Windows.Threading.DispatcherTimer { Interval = System.TimeSpan.FromSeconds(8) };
            t.Tick += (_, _) => { t.Stop(); OpenSessionReport_Click(this, new RoutedEventArgs()); };
            Loaded += (_, _) => t.Start();
        }
        // 调试辅助：NSM_OPENCOMP=1 时延迟自动弹出脑电成分分离窗，供无人值守截图验证。
        if (System.Environment.GetEnvironmentVariable("NSM_OPENCOMP") == "1")
        {
            var t = new System.Windows.Threading.DispatcherTimer { Interval = System.TimeSpan.FromSeconds(8) };
            t.Tick += (_, _) => { t.Stop(); OpenEegComponents_Click(this, new RoutedEventArgs()); };
            Loaded += (_, _) => t.Start();
        }
    }

    /// <summary>打开脑电成分分离弹窗（δ/θ/α/β 频带 FFT 带通分离，实时刷新）。</summary>
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

    /// <summary>打开整场报告弹窗，展示当前会话（实时累积或回放整场）的汇总数据。</summary>
    private void OpenSessionReport_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        if (_reportWindow is { IsVisible: true })
        {
            _reportWindow.Activate();
            _reportWindow.Refresh();
            return;
        }

        _reportWindow = new SessionReportWindow(vm.Session) { Owner = this };
        _reportWindow.Closed += (_, _) => _reportWindow = null;
        _reportWindow.Show();
    }
}
