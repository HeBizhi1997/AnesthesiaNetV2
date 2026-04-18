using EEGMonitor.Models;

namespace EEGMonitor.Views.Dialogs;

public partial class EventAnnotationDialog : System.Windows.Window
{
    public ClinicalEventType SelectedEventType { get; private set; }
    public string EventLabel { get; private set; } = "";
    public string Notes { get; private set; } = "";

    public EventAnnotationDialog()
    {
        InitializeComponent();
        EventTypeCombo.ItemsSource = Enum.GetNames<ClinicalEventType>()
                                         .Where(n => !n.StartsWith("Auto"))
                                         .ToList();
        EventTypeCombo.SelectedIndex = 0;
    }

    private void AddClick(object sender, System.Windows.RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(LabelBox.Text))
        {
            System.Windows.MessageBox.Show("Please enter a label.", "Validation",
                System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Warning);
            return;
        }
        Enum.TryParse<ClinicalEventType>(EventTypeCombo.SelectedItem?.ToString(), out var t);
        SelectedEventType = t;
        EventLabel = LabelBox.Text.Trim();
        Notes = NotesBox.Text.Trim();
        DialogResult = true;
    }

    private void CancelClick(object sender, System.Windows.RoutedEventArgs e)
    {
        DialogResult = false;
    }
}
