using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EEGMonitor.AnesthesiaSleep.Models;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class SedationScaleViewModel : ObservableObject
{
    [ObservableProperty] private ObservableCollection<SedationRecord> _records = new();

    [ObservableProperty] private SedationScaleType _selectedScaleType = SedationScaleType.Ramsay;
    [ObservableProperty] private int _selectedScore;
    [ObservableProperty] private string _selectedScoreLabel = "";
    [ObservableProperty] private string _observer = "";
    [ObservableProperty] private string _sedationNotes = "";

    public event Action<SedationRecord>? AssessmentRecorded;

    public static IReadOnlyDictionary<int, string> RamsayLabels { get; } = new Dictionary<int, string>
    {
        [1] = "焦虑、激越、不安",
        [2] = "合作、定向、安静",
        [3] = "仅对指令有反应",
        [4] = "对轻叩眉间或大声听觉刺激有敏捷反应",
        [5] = "对轻叩眉间或大声听觉刺激反应迟钝",
        [6] = "无反应"
    };

    public static IReadOnlyDictionary<int, string> RASSLabels { get; } = new Dictionary<int, string>
    {
        [+4] = "好斗",
        [+3] = "非常激越",
        [+2] = "激越",
        [+1] = "烦躁不安",
        [0] = "警觉且安静",
        [-1] = "嗜睡",
        [-2] = "轻度镇静",
        [-3] = "中度镇静",
        [-4] = "深度镇静",
        [-5] = "无法唤醒"
    };

    public static IReadOnlyDictionary<int, string> MOAASLabels { get; } = new Dictionary<int, string>
    {
        [5] = "正常语调呼唤姓名即能轻易反应",
        [4] = "正常语调呼唤姓名反应迟钝",
        [3] = "仅在大声和/或反复呼唤姓名后有反应",
        [2] = "仅在轻度推搡或摇动后有反应",
        [1] = "仅在疼痛刺激（挤压斜方肌）后有反应",
        [0] = "疼痛刺激后无反应"
    };

    private void UpdateSelectedLabel()
    {
        SelectedScoreLabel = SelectedScaleType switch
        {
            SedationScaleType.Ramsay => RamsayLabels.GetValueOrDefault(SelectedScore, ""),
            SedationScaleType.RASS => RASSLabels.GetValueOrDefault(SelectedScore, ""),
            SedationScaleType.MOAAS => MOAASLabels.GetValueOrDefault(SelectedScore, ""),
            _ => ""
        };
    }

    partial void OnSelectedScaleTypeChanged(SedationScaleType value) => UpdateSelectedLabel();
    partial void OnSelectedScoreChanged(int value) => UpdateSelectedLabel();

    [RelayCommand]
    private void SetScaleType(string typeStr)
    {
        if (Enum.TryParse<SedationScaleType>(typeStr, out var t))
        {
            SelectedScaleType = t;
        }
    }

    [RelayCommand]
    private void SetScore(string scoreStr)
    {
        if (int.TryParse(scoreStr, out int score))
        {
            SelectedScore = score;
            UpdateSelectedLabel();
        }
    }

    [RelayCommand]
    private void RecordAssessment()
    {
        var record = new SedationRecord
        {
            ScaleType = SelectedScaleType,
            Score = SelectedScore,
            ScoreLabel = SelectedScoreLabel,
            Observer = string.IsNullOrWhiteSpace(Observer) ? null : Observer.Trim(),
            Notes = string.IsNullOrWhiteSpace(SedationNotes) ? null : SedationNotes.Trim()
        };
        Records.Add(record);
        AssessmentRecorded?.Invoke(record);
        SedationNotes = "";
    }

    public void Reset()
    {
        Records.Clear();
    }

    public List<SedationRecord> GetRecords() => Records.ToList();
}
