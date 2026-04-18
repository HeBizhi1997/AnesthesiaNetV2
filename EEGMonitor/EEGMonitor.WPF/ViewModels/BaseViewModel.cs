using CommunityToolkit.Mvvm.ComponentModel;
using System.Windows;

namespace EEGMonitor.ViewModels;

public abstract partial class BaseViewModel : ObservableObject
{
    protected static void RunOnUI(Action action)
    {
        if (Application.Current?.Dispatcher.CheckAccess() == true)
            action();
        else
            Application.Current?.Dispatcher.Invoke(action);
    }
}
