using System.Windows;

namespace EEGMonitor.AnesthesiaSleep.Views.Dialogs;

public partial class ReportPreviewDialog : Window
{
    public ReportPreviewDialog(string previewText)
    {
        InitializeComponent();
        PreviewTextBox.Text = previewText;
    }

    private void Close_Click(object sender, RoutedEventArgs e)
    {
        Close();
    }
}
