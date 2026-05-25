using System.Windows;
using EEGMonitor.AnesthesiaSleep.ViewModels;

namespace EEGMonitor.AnesthesiaSleep.Views.Dialogs;

public partial class SimulationDialog : Window
{
    public SimulationDialog(SimulationViewModel vm)
    {
        InitializeComponent();
        DataContext = vm;
    }
}
