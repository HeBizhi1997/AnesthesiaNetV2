using Ads1299Monitor.Configuration;
using Ads1299Monitor.Services;
using Ads1299Monitor.ViewModels;
using Ads1299Monitor.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Serilog;
using System.IO;
using System.Windows;

namespace Ads1299Monitor;

public partial class App : Application
{
    private ServiceProvider? _services;

    protected override void OnStartup(StartupEventArgs e)
    {
        RegisterGlobalExceptionHandlers();
        base.OnStartup(e);

        try
        {
            var config = AppConfig.Load();

            var logDir = Path.Combine(AppContext.BaseDirectory, "logs");
            Directory.CreateDirectory(logDir);
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .WriteTo.File(Path.Combine(logDir, "ads1299mon-.log"), rollingInterval: RollingInterval.Day)
                .CreateLogger();

            var sc = new ServiceCollection();
            sc.AddLogging(b => b.AddSerilog(dispose: true));
            sc.AddSingleton(config);

            sc.AddHttpClient<EEGProcessingClient>(c =>
            {
                c.BaseAddress = new Uri(config.Python.BaseUrl);
                c.Timeout = TimeSpan.FromSeconds(10);
            });

            sc.AddSingleton<SerialPortService>();
            sc.AddSingleton<PulseSerialService>();
            sc.AddSingleton<PpgSerialService>();
            sc.AddSingleton<RecordingService>();
            sc.AddSingleton<PythonServiceLauncher>();
            sc.AddSingleton<DataPipeline>();
            sc.AddSingleton<MainViewModel>();

            _services = sc.BuildServiceProvider();

            var vm = _services.GetRequiredService<MainViewModel>();
            var window = new MainWindow { DataContext = vm };
            window.Show();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"启动失败:\n{ex}", "Ads1299Monitor", MessageBoxButton.OK, MessageBoxImage.Error);
            Shutdown();
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        try { _services?.GetService<MainViewModel>()?.Shutdown(); } catch { }
        _services?.Dispose();
        Log.CloseAndFlush();
        base.OnExit(e);
    }

    private void RegisterGlobalExceptionHandlers()
    {
        DispatcherUnhandledException += (_, args) =>
        {
            MessageBox.Show($"未处理异常:\n{args.Exception.Message}", "Ads1299Monitor",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            args.Handled = true;
        };
        AppDomain.CurrentDomain.UnhandledException += (_, args) =>
            Log.Error(args.ExceptionObject as Exception, "AppDomain unhandled exception");
        System.Threading.Tasks.TaskScheduler.UnobservedTaskException += (_, args) =>
        {
            Log.Error(args.Exception, "Unobserved task exception");
            args.SetObserved();
        };
    }
}
