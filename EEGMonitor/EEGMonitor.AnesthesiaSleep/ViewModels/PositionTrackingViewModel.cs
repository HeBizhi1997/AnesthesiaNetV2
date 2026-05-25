using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EEGMonitor.AnesthesiaSleep.Models;

namespace EEGMonitor.AnesthesiaSleep.ViewModels;

public partial class PositionTrackingViewModel : ObservableObject
{
    [ObservableProperty] private ObservableCollection<PositionRecord> _positionRecords = new();
    [ObservableProperty] private BodyPosition _currentPosition = BodyPosition.Supine;
    [ObservableProperty] private string _currentPositionLabel = "Supine";
    [ObservableProperty] private string _positionNotes = "";

    public event Action<PositionRecord>? PositionChanged;

    partial void OnCurrentPositionChanged(BodyPosition value)
    {
        CurrentPositionLabel = value switch
        {
            BodyPosition.Supine => "仰卧",
            BodyPosition.Prone => "俯卧",
            BodyPosition.LeftLateral => "左侧卧",
            BodyPosition.RightLateral => "右侧卧",
            BodyPosition.Sitting => "坐位",
            BodyPosition.Standing => "站立",
            _ => "未知"
        };
    }

    [RelayCommand]
    private void SetPosition(string positionName)
    {
        if (Enum.TryParse<BodyPosition>(positionName, out var pos))
        {
            var record = new PositionRecord
            {
                Position = pos,
                Notes = string.IsNullOrWhiteSpace(PositionNotes) ? null : PositionNotes.Trim()
            };
            PositionRecords.Add(record);

            CurrentPosition = pos;
            PositionChanged?.Invoke(record);
            PositionNotes = "";
        }
    }

    public void Reset()
    {
        PositionRecords.Clear();
        CurrentPosition = BodyPosition.Supine;
    }

    public List<PositionRecord> GetRecords() => PositionRecords.ToList();
}
