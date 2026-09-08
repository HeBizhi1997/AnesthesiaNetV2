using System.Windows;
using System.Windows.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NSMMonitor.Configuration;
using NSMMonitor.Services;
using NSMMonitor.ViewModels;
using Serilog;

namespace NSMMonitor;

public partial class App : Application
{
    private IHost? _host;
    private DispatcherTimer? _clock;
    private DispatcherTimer? _ppgRender;

    protected override void OnStartup(StartupEventArgs e)
    {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Information()
            .WriteTo.Console()
            .WriteTo.File("logs/nsm-.log", rollingInterval: RollingInterval.Day)
            .CreateLogger();

        var config = NsmConfig.Load();
        if (config.LoadError != null) Log.Warning("配置：{Error}", config.LoadError);

        _host = Host.CreateDefaultBuilder()
            .UseSerilog()
            .ConfigureServices(s => ConfigureServices(s, config))
            .Build();

        _host.Start();

        var vm = _host.Services.GetRequiredService<MainViewModel>();
        var window = _host.Services.GetRequiredService<MainWindow>();
        window.DataContext = vm;

        _clock = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _clock.Tick += (_, _) => vm.TickClock();
        _clock.Start();

        // 血氧渲染节拍：脉搏波由 UI 线程主动取，串口线程不再往 UI 推数据（见 MainViewModel.RenderTick）
        _ppgRender = new DispatcherTimer(DispatcherPriority.Render) { Interval = TimeSpan.FromMilliseconds(40) };
        _ppgRender.Tick += (_, _) => vm.RenderTick();
        _ppgRender.Start();

        window.Show();
        base.OnStartup(e);
    }

    private static void ConfigureServices(IServiceCollection services, NsmConfig config)
    {
        services.AddSingleton(config);
        services.AddSingleton<NsmSerialService>();
        services.AddSingleton<NsmSimulatorService>();
        services.AddSingleton<NsmPlaybackService>();
        services.AddSingleton<NsmRecordingService>();
        services.AddSingleton<SessionBuffer>();
        // 血氧链路：与脑电各自独立
        services.AddSingleton<PpgSerialService>();
        services.AddSingleton<PpgSimulatorService>();
        services.AddSingleton<MainViewModel>();
        services.AddSingleton<MainWindow>();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _clock?.Stop();
        _ppgRender?.Stop();
        _host?.Dispose();
        Log.CloseAndFlush();
        base.OnExit(e);
    }
}
