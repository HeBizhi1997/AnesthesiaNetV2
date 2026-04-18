using EEGMonitor.Infrastructure.Logging;
using EEGMonitor.Services.Events;
using EEGMonitor.Services.Playback;
using EEGMonitor.Services.Processing;
using EEGMonitor.Services.Recording;
using EEGMonitor.Services.SerialPort;
using EEGMonitor.Services.Simulation;
using EEGMonitor.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Serilog;
using System.Windows;

namespace EEGMonitor;

public partial class App : Application
{
    private IHost? _host;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        SerilogSetup.Configure();

        _host = Host.CreateDefaultBuilder()
            .UseSerilog()
            .ConfigureServices(ConfigureServices)
            .Build();

        _host.Start();

        var vm = _host.Services.GetRequiredService<MainViewModel>();
        var mainWindow = _host.Services.GetRequiredService<Views.MainWindow>();
        mainWindow.DataContext = vm;
        mainWindow.Show();

        Log.Information("EEGMonitor application started");
    }

    private static void ConfigureServices(IServiceCollection services)
    {
        // Named HTTP client shared by EEGProcessingClient and VitalSimulatorService
        services.AddHttpClient("simulator", client =>
        {
            client.BaseAddress = new Uri("http://localhost:8765/");
            client.Timeout = TimeSpan.FromMinutes(3); // vital file loading can be slow
        });

        // Typed client for EEG processing (short timeout)
        services.AddHttpClient<IEEGProcessingClient, EEGProcessingClient>(client =>
        {
            client.BaseAddress = new Uri("http://localhost:8765/");
            client.Timeout = TimeSpan.FromSeconds(5);
        });

        // Core services
        services.AddSingleton<ISerialPortService, SerialPortService>();
        services.AddSingleton<IRecordingService, RecordingService>();
        services.AddSingleton<IPlaybackService, PlaybackService>();
        services.AddSingleton<IEventAnnotationService, EventAnnotationService>();
        services.AddSingleton<Infrastructure.Pipeline.DataPipeline>();

        // Simulation service (uses named "simulator" client + DataPipeline)
        services.AddSingleton<IVitalSimulatorService, VitalSimulatorService>();

        // ViewModels
        services.AddSingleton<MainViewModel>();

        // Views
        services.AddSingleton<Views.MainWindow>();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        base.OnExit(e);
        Log.Information("EEGMonitor shutting down");

        // Run async host teardown on a background thread so we don't deadlock the
        // UI thread, then force-exit to guarantee all lingering threads are killed.
        Task.Run(async () =>
        {
            try
            {
                if (_host != null)
                {
                    await _host.StopAsync(TimeSpan.FromSeconds(3)).ConfigureAwait(false);
                    _host.Dispose();
                }
            }
            catch { /* ignore errors during shutdown */ }
            finally
            {
                Log.CloseAndFlush();
            }
        }).Wait(TimeSpan.FromSeconds(5));

        Environment.Exit(0);
    }
}
