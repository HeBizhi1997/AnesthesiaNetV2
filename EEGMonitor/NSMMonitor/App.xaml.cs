using System.Windows;
using System.Windows.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NSMMonitor.Services;
using NSMMonitor.ViewModels;
using Serilog;

namespace NSMMonitor;

public partial class App : Application
{
    private IHost? _host;
    private DispatcherTimer? _clock;

    protected override void OnStartup(StartupEventArgs e)
    {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Information()
            .WriteTo.Console()
            .WriteTo.File("logs/nsm-.log", rollingInterval: RollingInterval.Day)
            .CreateLogger();

        _host = Host.CreateDefaultBuilder()
            .UseSerilog()
            .ConfigureServices(ConfigureServices)
            .Build();

        _host.Start();

        var vm = _host.Services.GetRequiredService<MainViewModel>();
        var window = _host.Services.GetRequiredService<MainWindow>();
        window.DataContext = vm;

        _clock = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _clock.Tick += (_, _) => vm.TickClock();
        _clock.Start();

        window.Show();
        base.OnStartup(e);
    }

    private static void ConfigureServices(IServiceCollection services)
    {
        services.AddSingleton<NsmSerialService>();
        services.AddSingleton<NsmSimulatorService>();
        services.AddSingleton<NsmPlaybackService>();
        services.AddSingleton<NsmRecordingService>();
        services.AddSingleton<MainViewModel>();
        services.AddSingleton<MainWindow>();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _clock?.Stop();
        _host?.Dispose();
        Log.CloseAndFlush();
        base.OnExit(e);
    }
}
