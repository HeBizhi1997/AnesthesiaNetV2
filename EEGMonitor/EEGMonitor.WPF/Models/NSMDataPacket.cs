namespace EEGMonitor.Models;

/// <summary>
/// Parsed NSM anesthesia monitor data packet (355-byte UART protocol).
/// </summary>
public record NSMDataPacket
{
    public DateTime LocalTimestamp { get; init; }

    // ── Device ──
    public uint DeviceTimeSec { get; init; }

    // ── Electrode status ──
    public bool ElectrodeAlarm { get; init; }
    public bool ImpedanceHigh { get; init; }
    public bool ElectrodeInvalid { get; init; }
    public int BlackImpedance { get; init; }  // 1-10, 15=too high (Electrode 3)
    public int WhiteImpedance { get; init; }  // 1-10, 15=too high (Electrode 1)

    // ── Anesthesia indices ──
    /// <summary>Anesthesia depth index (0-99), equivalent to BIS. 0xEE/0xFF = invalid.</summary>
    public int CSI { get; init; }
    /// <summary>Burst Suppression ratio (0-100). 0xFF = invalid.</summary>
    public int BS { get; init; }
    /// <summary>Signal Quality Index (0-100). 0xFF = invalid.</summary>
    public int SQI { get; init; }
    /// <summary>EMG signal index (0-100). 0xFF = invalid.</summary>
    public int EMG { get; init; }
    /// <summary>Nociception index (0-99). 0xFF = invalid.</summary>
    public int NOX { get; init; }

    // ── Clinical event ──
    public int EventNumber { get; init; }
    public NSMEventType EventType { get; init; }

    // ── Alarms ──
    public int AlarmHigh { get; init; }
    public int AlarmLow { get; init; }

    // ── EEG waveform (100 samples, signed byte → µV) ──
    public double[] EEGSamplesUv { get; init; } = Array.Empty<double>();

    // ── Band powers (dB, signed byte) ──
    public int DeltaPowerDb { get; init; }
    public int ThetaPowerDb { get; init; }
    public int AlphaPowerDb { get; init; }
    public int BetaPowerDb { get; init; }
    public int GammaPowerDb { get; init; }

    // ── Derived ──
    public int SEF95 { get; init; }    // 1-44 Hz
    public int EOG { get; init; }      // 0-100

    // ── Validity ──
    public bool CSIValid => CSI != 0xEE && CSI != 0xFF;
    public bool BSValid => BS != 0xFF;
    public bool SQIValid => SQI != 0xFF;
    public bool EMGValid => EMG != 0xFF;
    public bool NOXValid => NOX != 0xFF;
}

public enum NSMEventType : byte
{
    General = 0,
    Induction = 1,
    Intubation = 2,
    Maintenance = 3,
    Surgery = 4,
    Injection = 5,
    Note = 6,
    EndMaintenance = 7,
    Movement = 8,
}
