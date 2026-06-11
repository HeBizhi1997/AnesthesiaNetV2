using System.Windows;
using System.Windows.Threading;

namespace EEGRecorder;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        // Never let an unhandled exception kill a recording silently — show it, keep running.
        DispatcherUnhandledException += (_, args) =>
        {
            MessageBox.Show($"未处理异常:\n{args.Exception.Message}", "EEGRecorder",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            args.Handled = true;
        };
        base.OnStartup(e);
    }
}
