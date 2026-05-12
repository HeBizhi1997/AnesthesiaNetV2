"""
Parse NSM record files (.nsm) and extract all packet data for analysis.
Usage: python scripts/nsm_file_parser.py doc/pl_男_35_002_202604102339.nsm
"""
import sys
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

# FILE format: 128 bytes/pkt (subset of live 353-byte protocol)
# [0x80] [len:1] [payload:124 bytes] [CRC:2 bytes]
PACKET_SIZE = 128
PAYLOAD_LENGTH = 124   # data bytes after length field (excludes CRC)
EEG_OFFSET = 22        # same position as live format
EEG_SAMPLES = 100       # same count as live format


def parse_packet(buf: bytes):
    """Parse a single 128-byte NSM file-format packet."""
    if len(buf) < PACKET_SIZE or buf[0] != 0x80:
        return None
    pkt_len = buf[1]
    if pkt_len != PAYLOAD_LENGTH:
        return None

    def sbyte(b): return b if b < 128 else b - 256

    eeg = [sbyte(buf[i]) for i in range(EEG_OFFSET, EEG_OFFSET + EEG_SAMPLES)]

    # File format: only first 124 bytes of data fields stored
    # NOX, band powers, SEF, EOG not in file format (positions 125+)
    return {
        "device_time": buf[6] | (buf[7] << 8) | (buf[8] << 16) | (buf[9] << 24),
        "block_status": buf[10],
        "event_number": buf[11],
        "event_type": buf[12],
        "csi": buf[13] if buf[13] not in (0xEE, 0xFF) else None,
        "bs": buf[14] if buf[14] != 0xFF else None,
        "sqi": buf[15] if buf[15] != 0xFF else None,
        "emg": buf[18] if buf[18] != 0xFF else None,
        "nox": None,       # not in file format
        "black_imp": buf[16],
        "white_imp": buf[17],
        "alarm_high": buf[20],
        "alarm_low": buf[21],
        "delta_db": 0,     # not in file format
        "theta_db": 0,
        "alpha_db": 0,
        "beta_db": 0,
        "gamma_db": 0,
        "eog": 0,
        "sef95": 0,
        "eeg": eeg,
    }


def to_linear(db):
    """dB to linear power: P = 10^(dB/10)"""
    return math.pow(10, max(-40, min(40, db)) / 10.0)


def rel_powers(pkt):
    """Convert dB band powers to relative percentages (0-100)."""
    dl = to_linear(pkt["delta_db"])
    tl = to_linear(pkt["theta_db"])
    al = to_linear(pkt["alpha_db"])
    bl = to_linear(pkt["beta_db"])
    gl = to_linear(pkt["gamma_db"])
    total = dl + tl + al + bl + gl
    return {
        "delta": dl / total * 100,
        "theta": tl / total * 100,
        "alpha": al / total * 100,
        "beta": bl / total * 100,
        "gamma": gl / total * 100,
    }


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("doc/pl_男_35_002_202604102339.nsm")
    raw = path.read_bytes()
    file_size = len(raw)
    expected_packets = file_size // PACKET_SIZE
    print(f"File: {path.name}")
    print(f"Size: {file_size:,} bytes, expected {expected_packets:,} packets")
    print(f"Patient info from filename: {path.stem}")

    # Parse all packets
    packets = []
    error_at = []
    for i in range(0, file_size - PACKET_SIZE + 1, PACKET_SIZE):
        pkt = parse_packet(raw[i:i + PACKET_SIZE])
        if pkt:
            packets.append(pkt)
        else:
            error_at.append(i)

    print(f"Valid packets: {len(packets):,} / {expected_packets:,} ({len(packets)/expected_packets*100:.1f}%)")
    if error_at:
        print(f"Parse failures at offsets: {error_at[:5]}{'...' if len(error_at) > 5 else ''}")

    if len(packets) == 0:
        print("No valid packets found!")
        return

    # ── Time analysis ──
    print(f"\n{'='*60}")
    print("TIME ANALYSIS")
    t0 = packets[0]["device_time"]
    t1 = packets[-1]["device_time"]
    duration = t1 - t0
    hours = duration / 3600
    actual_rate = len(packets) / duration if duration > 0 else 0
    print(f"  Start time (device): {t0}")
    print(f"  End time (device):   {t1}")
    print(f"  Duration: {duration:,} sec = {hours:.1f} hours")
    print(f"  Packet rate: {actual_rate:.2f} Hz")
    print(f"  Sample rate (100 samples/pkt): {actual_rate * 100:.0f} Hz")

    # ── CSI / BIS analysis ──
    csi_vals = [p["csi"] for p in packets if p["csi"] is not None]
    bs_vals = [p["bs"] for p in packets if p["bs"] is not None]
    sqi_vals = [p["sqi"] for p in packets if p["sqi"] is not None]
    emg_vals = [p["emg"] for p in packets if p["emg"] is not None]
    nox_vals = [p["nox"] for p in packets if p["nox"] is not None]

    print(f"\n{'='*60}")
    print("ANESTHESIA INDICES")
    for name, vals in [("CSI", csi_vals), ("BS", bs_vals), ("SQI", sqi_vals),
                        ("EMG", emg_vals), ("NOX", nox_vals)]:
        if vals:
            arr = np.array(vals)
            print(f"  {name:4s}: mean={arr.mean():5.1f}  std={arr.std():4.1f}  "
                  f"min={arr.min():5.1f}  max={arr.max():5.1f}  "
                  f"median={np.median(arr):5.1f}")

    # ── Band powers: compute from EEG via FFT (file format omits them) ──
    print(f"\n{'='*60}")
    print("BAND POWERS (computed from EEG FFT, relative %)")
    # Use mid-section of recording for stable estimate
    mid_start = len(packets) // 3
    mid_end = mid_start + min(600, len(packets) // 3)
    eeg_seg = np.array([v for p in packets[mid_start:mid_end] for v in p["eeg"]], dtype=np.float64)
    eeg_seg = eeg_seg - eeg_seg.mean()
    fft = np.abs(np.fft.rfft(eeg_seg))
    freqs = np.fft.rfftfreq(len(eeg_seg), d=1/100)
    total = float(fft.sum()) + 1e-12
    bands_def = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13),
                 ("beta", 13, 30), ("gamma", 30, 47)]
    for name, lo, hi in bands_def:
        mask = (freqs >= lo) & (freqs < hi)
        pct = float(fft[mask].sum()) / total * 100
        print(f"  {name:5s} ({lo:4.1f}-{hi:4.1f}Hz): {pct:5.1f}%")

    # ── EEG analysis ──
    print(f"\n{'='*60}")
    print("EEG ANALYSIS")
    all_eeg = np.array([v for p in packets for v in p["eeg"]], dtype=np.float64)
    eeg_std = float(np.std(all_eeg))
    eeg_mean = float(np.mean(all_eeg))
    print(f"  Total samples: {len(all_eeg):,}")
    print(f"  Mean: {eeg_mean:.2f} uV  (DC offset)")
    print(f"  Std:  {eeg_std:.2f} uV")
    print(f"  Range: {all_eeg.min():.0f} .. {all_eeg.max():.0f} uV")
    print(f"  Saturation (-128/127): min_sat={(all_eeg <= -127).sum()}, max_sat={(all_eeg >= 127).sum()}")

    # Frequency analysis on first 60 seconds
    n_fft_samples = min(6000, len(all_eeg))  # ~60 sec at 100Hz
    if n_fft_samples >= 256:
        eeg_seg = all_eeg[:n_fft_samples]
        eeg_seg = eeg_seg - eeg_seg.mean()  # remove DC
        fft = np.abs(np.fft.rfft(eeg_seg))
        freqs = np.fft.rfftfreq(n_fft_samples, d=1/100)  # assume 100 Hz
        print(f"\n  FFT (first {n_fft_samples/100:.0f}s, DC removed):")
        for lo, hi, name in [(0.5, 4, "delta"), (4, 8, "theta"), (8, 13, "alpha"),
                              (13, 30, "beta"), (30, 47, "gamma")]:
            mask = (freqs >= lo) & (freqs < hi)
            power = float(fft[mask].sum())
            print(f"    {name:5s} ({lo:4.1f}-{hi:4.1f}Hz): power={power:10.1f}")

    # ── CSI timeline (simplified) ──
    print(f"\n{'='*60}")
    print("CSI TIMELINE (every 200 packets)")
    for i in range(0, len(packets), max(1, len(packets) // 15)):
        p = packets[i]
        dt = p["device_time"] - t0
        mins = dt / 60
        print(f"  {mins:6.1f} min  CSI={p['csi']:3d}  BS={p['bs']:3d}  "
              f"SQI={p['sqi']:3d}  EMG={p['emg']:3d}  "
              f"Imp(B/W)={p['black_imp']}/{p['white_imp']}")

    # ── Events ──
    events = [p for p in packets if p["event_number"] > 0]
    if events:
        print(f"\n{'='*60}")
        print(f"CLINICAL EVENTS ({len(events)} found):")
        for e in events[:20]:
            dt = e["device_time"] - t0
            print(f"  @{dt/60:6.1f} min  #{e['event_number']}  type={e['event_type']}")
    else:
        print(f"\nNo clinical events found.")

    # ── Impedance ──
    print(f"\n{'='*60}")
    print("IMPEDANCE SUMMARY")
    black = [p["black_imp"] for p in packets]
    white = [p["white_imp"] for p in packets]
    print(f"  Black(3): mean={np.mean(black):.1f}  max={max(black)}  high(>=15)={(np.array(black)>=15).sum()} packets")
    print(f"  White(1): mean={np.mean(white):.1f}  max={max(white)}  high(>=15)={(np.array(white)>=15).sum()} packets")


if __name__ == "__main__":
    main()
