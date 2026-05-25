using System.Windows;

namespace EEGMonitor.AnesthesiaSleep.Views;

public partial class ShellWindow : Window
{
    private readonly ViewModels.ShellViewModel _vm;

    public ShellWindow(ViewModels.ShellViewModel vm)
    {
        InitializeComponent();
        _vm = vm;
        DataContext = vm;
        Closed += (_, _) => _vm.Cleanup();
    }
}
