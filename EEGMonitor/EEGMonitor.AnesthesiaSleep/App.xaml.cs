using System.Windows;
using EEGMonitor.AnesthesiaSleep.Infrastructure.Messaging;
using EEGMonitor.AnesthesiaSleep.Infrastructure.ReportGeneration;
using EEGMonitor.AnesthesiaSleep.Infrastructure.Simulation;
using EEGMonitor.AnesthesiaSleep.Infrastructure.SleepStaging;
using EEGMonitor.AnesthesiaSleep.ViewModels;
using EEGMonitor.Infrastructure.Logging;
using EEGMonitor.Services.Events;
using EEGMonitor.Services.Playback;
using EEGMonitor.Services.Processing;
using EEGMonitor.Services.PythonService;
using EEGMonitor.Services.Recording;
using EEGMonitor.Services.SerialPort;
using EEGMonitor.Services.Simulation;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Serilog;

namespace EEGMonitor.AnesthesiaSleep;

public partial class App : Application
{
    private IHost? _host;
    private PythonServiceLauncher? _pythonLauncher;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        SerilogSetup.Configure();
        RegisterGlobalExceptionHandlers();

        try
        {
            _host = Host.CreateDefaultBuilder()
                .UseSerilog()
                .ConfigureServices(ConfigureServices)
                .Build();

            _host.Start();

            var vm = _host.Services.GetRequiredService<ShellViewModel>();
            var shellWindow = new Views.ShellWindow(vm);
            shellWindow.Show();
        }
        catch (Exception ex)
        {
            Log.Fatal(ex, "Fatal exception during application startup");
            MessageBox.Show(
                $"程序启动失败：\n\n{ex.GetType().Name}: {ex.Message}\n\n详见日志: %LOCALAPPDATA%\\EEGMonitor\\Logs",
                "EEGMonitor 启动异常", MessageBoxButton.OK, MessageBoxImage.Error);
            Shutdown(1);
            return;
        }

        Log.Information("EEGMonitor.AnesthesiaSleep application started");

        _pythonLauncher = _host.Services.GetRequiredService<PythonServiceLauncher>();
        _ = Task.Run(async () =>
        {
            var ok = await _pythonLauncher.EnsureRunningAsync();
            if (!ok)
                Log.Warning("EEG processing service could not be started automatically");
        });
    }

    /// <summary>
    /// Log every unhandled-exception channel with a full stack trace instead of letting the
    /// process die silently. UI-thread exceptions are kept non-fatal so one bad event/render
    /// doesn't close the whole app.
    /// </summary>
    private void RegisterGlobalExceptionHandlers()
    {
        DispatcherUnhandledException += (_, args) =>
        {
            Log.Error(args.Exception, "Unhandled UI (Dispatcher) exception");
            MessageBox.Show(
                $"发生未处理异常（已记录日志，程序继续运行）：\n\n{args.Exception.GetType().Name}: {args.Exception.Message}",
                "EEGMonitor", MessageBoxButton.OK, MessageBoxImage.Warning);
            args.Handled = true;
        };
        AppDomain.CurrentDomain.UnhandledException += (_, args) =>
            Log.Fatal(args.ExceptionObject as Exception,
                "Unhandled AppDomain exception (terminating={Terminating})", args.IsTerminating);
        System.Threading.Tasks.TaskScheduler.UnobservedTaskException += (_, args) =>
        {
            Log.Error(args.Exception, "Unobserved background task exception");
            args.SetObserved();
        };
    }

    private static void ConfigureServices(IServiceCollection services)
    {
        // ═══ Same as original EEGMonitor — core services ═══════════════════
        services.AddHttpClient("simulator", client =>
        {
            client.BaseAddress = new Uri("http://localhost:8765/");
            client.Timeout = TimeSpan.FromMinutes(3);
        });

        services.AddHttpClient<IEEGProcessingClient, EEGProcessingClient>(client =>
        {
            client.BaseAddress = new Uri("http://localhost:8765/");
            client.Timeout = TimeSpan.FromSeconds(5);
        });

        services.AddSingleton<SerialPortService>();
        services.AddSingleton<NSMSerialService>();
        services.AddSingleton<ISerialPortService>(sp => sp.GetRequiredService<SerialPortService>());
        services.AddSingleton<IPulseSerialService, PulseSerialService>();
        services.AddSingleton<IRecordingService, RecordingService>();
        services.AddSingleton<IPlaybackService, PlaybackService>();
        services.AddSingleton<IEventAnnotationService, EventAnnotationService>();
        services.AddSingleton<EEGMonitor.Infrastructure.Pipeline.DataPipeline>();
        services.AddSingleton<PythonServiceLauncher>();
        services.AddSingleton<IVitalSimulatorService, VitalSimulatorService>();

        // ═══ Sleep-specific infrastructure ═════════════════════════════════
        services.AddSingleton<ResultBroker>();
        services.AddSingleton<SleepStageRuleEngine>();
        services.AddSingleton<ReportGenerator>();
        services.AddSingleton<SleepSimulator>();

        // ═══ ViewModels ════════════════════════════════════════════════════
        services.AddSingleton<SleepStagingViewModel>();
        services.AddSingleton<SleepStatisticsViewModel>();
        services.AddSingleton<SpO2TrendViewModel>();
        services.AddSingleton<DrugAdministrationViewModel>();
        services.AddSingleton<SedationScaleViewModel>();
        services.AddSingleton<PositionTrackingViewModel>();
        services.AddSingleton<ReportViewModel>();
        services.AddSingleton<SimulationViewModel>();
        services.AddSingleton<ShellViewModel>();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        base.OnExit(e);
        Log.Information("EEGMonitor.AnesthesiaSleep shutting down");

        Task.Run(async () =>
        {
            try
            {
                if (_pythonLauncher != null)
                    await _pythonLauncher.DisposeAsync();

                if (_host != null)
                {
                    await _host.StopAsync(TimeSpan.FromSeconds(3)).ConfigureAwait(false);
                    _host.Dispose();
                }
            }
            catch { }
            finally
            {
                Log.CloseAndFlush();
            }
        }).Wait(TimeSpan.FromSeconds(5));

        Environment.Exit(0);
    }
}
