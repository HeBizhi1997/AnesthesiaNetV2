# EEGRecorder — 脑电/脉搏 原始信号记录仪

一个**独立**的 WPF 采集程序,只做一件事:把 ADS1299 脑电 + 指夹 PPG 的**原始信号无损落盘**,
便于离线喂给训练管线(VitalDB / Python)。不依赖 `EEGMonitor.Ads1299`(协议代码已内置)。

## 使用
1. 编辑 **`appsettings.json`**(在 exe 同目录)设置串口与存储目录:
   - `Serial.EegPort`(默认 COM6,230400)、`Serial.PpgPort`(默认 COM7,57600,可 `PpgEnabled:false` 关闭)
   - `Recording.OutputDirectory` 存储根目录
2. 运行,点 **“连接设备”** → 看波形确认电极接好(右上角显示「信号正常 / 导联脱落」)。
3. 点 **“● 开始录制”** 开始落盘,**“■ 停止录制”** 结束。每次录制生成一个会话文件夹。

## 输出(每个会话一个文件夹 `{subject}_{yyyyMMdd_HHmmss}/`)
| 文件 | 内容 |
|---|---|
| `eeg.bin` | 连续 `float32`(小端)CH0 µV 流,均匀采样(速率见 meta)。`np.fromfile(.., "<f4")` |
| `ppg.bin` | 定长 18 字节/帧:`[int64 ticks][int32 ir][int32 red][uint8 spo2][uint8 hr]` |
| `meta.json` | 采样率(标称+实测)、增益、端口、起止时间、采样/帧计数 |

录制即原始,**不做任何滤波/重采样**(界面上的波形滤波仅用于显示)。

## 读取(Python)
```bash
python scripts/load.py <会话文件夹> --plot
```
`scripts/load.py` 用 `numpy.fromfile` 秒读,返回 `(meta, eeg, ppg)`。

## 笔记本单独部署(自包含单文件,免装 .NET)
```bash
dotnet publish EEGMonitor/EEGRecorder/EEGRecorder.csproj -c Release -r win-x64 ^
    --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true
```
把 `bin/Release/net8.0-windows/win-x64/publish/` 下的 `EEGRecorder.exe` + `appsettings.json`
拷到笔记本即可运行(无需安装运行时)。CP210x(COM6)/CH340(COM7)驱动需在目标机装好。
