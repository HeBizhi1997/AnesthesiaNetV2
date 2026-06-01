namespace Ads1299Monitor.Models;

/// <summary>One clinical event row (用药 / 气道操作 / 电刀 / 缝合 / 体位 …).</summary>
public sealed class EventItem
{
    public string Time { get; set; } = "";
    public string Category { get; set; } = "";   // 事件类型
    public string Name { get; set; } = "";        // 事件名称/药物
    public string Dose { get; set; } = "";         // 剂量/参数
    public string Operator { get; set; } = "";
    public string Note { get; set; } = "";
    public double TimeSeconds { get; set; }        // x position for the trend marker
}
