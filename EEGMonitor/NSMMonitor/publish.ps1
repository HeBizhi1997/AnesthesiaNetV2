<#
    NSMMonitor 独立发布脚本

    产出：publish\win-x64\ 下两个文件
        NSMMonitor.exe      自包含单文件（约 70 MB，已含 .NET 8 运行时）
        appsettings.json    配置文件，现场可直接改，改完重启生效

    目标机要求：64 位 Windows。无需安装 .NET 运行时。

    用法：
        .\publish.ps1                 发布到默认目录
        .\publish.ps1 -Output D:\NSM  发布到指定目录
        .\publish.ps1 -Clean          先清掉旧产物
#>
param(
    [string]$Output = "$PSScriptRoot\publish\win-x64",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$proj = Join-Path $PSScriptRoot "NSMMonitor.csproj"

if ($Clean -and (Test-Path $Output)) {
    Write-Host "清理 $Output" -ForegroundColor Yellow
    Remove-Item $Output -Recurse -Force
}

Write-Host "正在发布 NSMMonitor (win-x64, 自包含单文件) ..." -ForegroundColor Cyan

dotnet publish $proj `
    -c Release `
    -r win-x64 `
    --self-contained true `
    -p:PublishSingleFile=true `
    -o $Output `
    -v minimal --nologo

if ($LASTEXITCODE -ne 0) { Write-Host "发布失败" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "发布完成: $Output" -ForegroundColor Green
Get-ChildItem $Output | ForEach-Object {
    $size = if ($_.Length -gt 1MB) { "{0,8:N1} MB" -f ($_.Length / 1MB) } else { "{0,8:N0} KB" -f ($_.Length / 1KB) }
    Write-Host ("  {0}  {1}" -f $size, $_.Name)
}

Write-Host ""
Write-Host "部署方式：把整个目录拷到目标机，双击 NSMMonitor.exe 即可。" -ForegroundColor Gray
Write-Host "串口、融合权重、SpO2 标定曲线都在 appsettings.json 里改，不用重新编译。" -ForegroundColor Gray
