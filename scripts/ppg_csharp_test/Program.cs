// Standalone harness for the finger-clip PPG module — drives the SAME PpgSerialService
// the WPF client uses, so PR / PRV / SpO2 / PI come out of the integrated code, not a demo.
//
//   dotnet run --project scripts/ppg_csharp_test -- [COM7] [seconds]
using Ads1299Monitor.Services;
using Microsoft.Extensions.Logging.Abstractions;

string port = args.Length > 0 ? args[0] : "COM7";
double secs = args.Length > 1 && double.TryParse(args[1], out var s) ? s : 30;

Console.OutputEncoding = System.Text.Encoding.UTF8;
Console.WriteLine($"单独测试指夹 PPG  端口={port}  时长={secs:0}s  —— 请把手指放稳\n");

var svc = new PpgSerialService(NullLogger<PpgSerialService>.Instance);
int count = 0;
svc.ConnectionStatusChanged += msg => Console.WriteLine($"  > {msg}");
svc.ReadingReceived += r =>
{
    count++;
    string pr  = r.Pr  > 0 ? $"{r.Pr:0}"   : "--";
    string prv = r.Prv > 0 ? $"{r.Prv:0.0}" : "--";
    Console.WriteLine(
        $"[{DateTime.Now:HH:mm:ss}]  PR={pr,3} bpm   PRV={prv,5} ms   " +
        $"SpO2={r.Spo2,3}%   设备HR={r.DeviceHr,3}   PI={r.Pi:0.00}%");
};

if (!svc.Connect(port))
{
    Console.WriteLine($"无法打开 {port} —— 检查指夹是否插好 / 是否被占用");
    return;
}

var t0 = DateTime.Now;
while ((DateTime.Now - t0).TotalSeconds < secs)
    Thread.Sleep(200);

svc.Disconnect();
Console.WriteLine($"\n完成:共收到 {count} 次聚合读数。" +
                  (count == 0 ? "  (无数据 —— 检查指夹电源/手指)" : ""));
