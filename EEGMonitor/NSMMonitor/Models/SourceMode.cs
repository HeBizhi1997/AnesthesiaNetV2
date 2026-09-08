namespace NSMMonitor.Models;

/// <summary>脑电数据来源模式。</summary>
public enum SourceMode
{
    Simulator,
    Serial,
    Playback,
}

/// <summary>血氧（PPG）数据来源模式。无回放：PPG 原始帧当前不落盘。</summary>
public enum PpgSourceMode
{
    Serial,
    Simulator,
}
