using Ads1299Monitor.Models;
using System.Windows;
using System.Windows.Controls;

namespace Ads1299Monitor.Views;

public partial class EventDialog : Window
{
    public EventItem? Result { get; private set; }

    public EventDialog()
    {
        InitializeComponent();
    }

    private void Ok_Click(object sender, RoutedEventArgs e)
    {
        var category = (CategoryBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "其他";
        var name = string.IsNullOrWhiteSpace(NameBox.Text) ? category : NameBox.Text.Trim();
        Result = new EventItem
        {
            Category = category,
            Name = name,
            Dose = string.IsNullOrWhiteSpace(DoseBox.Text) ? "—" : DoseBox.Text.Trim(),
            Operator = string.IsNullOrWhiteSpace(OperatorBox.Text) ? "—" : OperatorBox.Text.Trim(),
            Note = string.IsNullOrWhiteSpace(NoteBox.Text) ? "—" : NoteBox.Text.Trim(),
        };
        DialogResult = true;
    }

    private void Cancel_Click(object sender, RoutedEventArgs e) => DialogResult = false;
}
