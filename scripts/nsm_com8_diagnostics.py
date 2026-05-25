"""
NSM COM8 real-time signal diagnostic tool.
Captures raw NSM packets, runs EEG preprocessing, and compares pipeline
band powers against the device's own band power estimates.

Usage: python scripts/nsm_com8_diagnostics.py [--duration 60] [--rate 200]
"""

import sys
import struct
import time
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import serial

# Ensure the tianjin root is on sys.path for EEGPreprocessor import
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from EEGMonitor.EEGProcessingService.preprocessing.eeg_preprocessor import EEGPreprocessor

# ── NSM Protocol Constants ──────────────────────────────────────────────
FRAME_HEADER = 0x80
PAYLOAD_LENGTH = 351
CRC_LENGTH = 2
PACKET_SIZE = PAYLOAD_LENGTH + CRC_LENGTH  # 353
EEG_SAMPLES_PER_PACKET = 100
EEG_OFFSET = 22
BAND_OFFSET = 126  # delta=126, theta=127, alpha=128, beta=129, gamma=130


def parse_nsm_packet(buf: bytes) -> dict | None:
    """Parse a 353-byte NSM packet. Returns dict with EEG samples and device metrics, or None."""
    if len(buf) < PACKET_SIZE:
        return None
    if buf[0] != FRAME_HEADER:
        return None
    length = buf[1] | (buf[2] << 8)
    if length != PAYLOAD_LENGTH:
        return None

    # Helper: C# (sbyte) semantics — unsigned byte → signed value
    def sb(b: int) -> int:
        return b if b < 128 else b - 256

    # EEG samples: bytes 22-121, signed byte → uV
    eeg = np.array([sb(buf[EEG_OFFSET + i]) for i in range(EEG_SAMPLES_PER_PACKET)],
                   dtype=np.float64)

    # Device band powers: bytes 126-130, signed byte in dB
    band_db = {
        "delta": sb(buf[BAND_OFFSET]),
        "theta": sb(buf[BAND_OFFSET + 1]),
        "alpha": sb(buf[BAND_OFFSET + 2]),
        "beta":  sb(buf[BAND_OFFSET + 3]),
        "gamma": sb(buf[BAND_OFFSET + 4]),
    }

    # Device indices
    csi = buf[13]
    sqi = buf[15]
    emg = buf[18]
    bs  = buf[14]

    # Electrode status
    block_status = buf[10]
    electrode_alarm   = (block_status & (1 << 1)) != 0
    impedance_high    = (block_status & (1 << 3)) != 0
    electrode_invalid = (block_status & (1 << 7)) != 0
    black_imp = buf[16]
    white_imp = buf[17]

    return {
        "eeg": eeg,
        "band_db": band_db,
        "csi": csi if csi not in (0xEE, 0xFF) else None,
        "sqi": sqi if sqi != 0xFF else None,
        "emg": emg if emg != 0xFF else None,
        "bs": bs if bs != 0xFF else None,
        "electrode_alarm": electrode_alarm,
        "impedance_high": impedance_high,
        "electrode_invalid": electrode_invalid,
        "black_imp": black_imp,
        "white_imp": white_imp,
    }


def band_db_to_relative(band_db: dict) -> dict:
    """Convert NSM dB band powers to relative ratios."""
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    linear = {}
    for b in bands:
        linear[b] = 10.0 ** (band_db[b] / 10.0)
    total = sum(linear.values()) or 1.0
    return {b: linear[b] / total for b in bands}


def print_header():
    print()
    print("=" * 130)
    print(f"{'Time':>8} | {'#':>5} | {'Clip%':>5} | {'DCOff':>6} | "
          f"{'δ_pipe':>6} {'θ_pipe':>6} {'α_pipe':>6} {'β_pipe':>6} {'γ_pipe':>6} | "
          f"{'δ_dev':>6} {'θ_dev':>6} {'α_dev':>6} {'β_dev':>6} {'γ_dev':>6} | "
          f"{'δΔ':>6} | {'RMS':>6} | {'SQI':>4} {'CSI':>4} {'Imp':>5} | Warnings")
    print("-" * 130)


def print_row(elapsed_s, chunk_idx, result, pipe_pow, dev_pow, discrepancy,
              rms_uv, sqi, csi, black_imp, white_imp, warnings):
    ts = f"{elapsed_s:>5.0f}s"
    pipe_str = f"{pipe_pow['delta']*100:5.1f}% {pipe_pow['theta']*100:5.1f}% {pipe_pow['alpha']*100:5.1f}% {pipe_pow['beta']*100:5.1f}% {pipe_pow['gamma']*100:5.1f}%"
    dev_str  = f"{dev_pow['delta']*100:5.1f}% {dev_pow['theta']*100:5.1f}% {dev_pow['alpha']*100:5.1f}% {dev_pow['beta']*100:5.1f}% {dev_pow['gamma']*100:5.1f}%" if dev_pow and dev_pow.get('delta') is not None else f"{'--':>36}"
    disc_str = f"{discrepancy['delta']*100:5.1f}%" if discrepancy else "  --- "
    warn_str = " | ".join(warnings) if warnings else ""
    clip = result.get("hw_clipping_pct", 0)
    dcoff = result.get("hw_dc_offset_uv", 0)

    print(f"{ts:>8} | {chunk_idx:>5} | {clip:4.0f}% | {dcoff:+5.0f}uV | "
          f"{pipe_str} | {dev_str} | {disc_str} | "
          f"{rms_uv:5.1f}uV | {sqi:3.0f} {csi:>4} {black_imp:>2}/{white_imp:<2} | {warn_str}")


def main():
    parser = argparse.ArgumentParser(description="NSM COM8 EEG signal diagnostics")
    parser.add_argument("--port", default="COM8", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--rate", type=int, default=200, help="NSM sample rate (Hz)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Duration in seconds (0 = run until Ctrl+C)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Samples per processing chunk")
    args = parser.parse_args()

    print(f"Opening {args.port} at {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    ser.reset_input_buffer()
    print(f"Connected. Sample rate: {args.rate} Hz. Chunk size: {args.chunk_size} samples.")

    preprocessor = EEGPreprocessor(sample_rate=args.rate)
    eeg_buffer = deque(maxlen=args.chunk_size + 200)
    device_band_accum = {b: [] for b in ["delta", "theta", "alpha", "beta", "gamma"]}

    packet_count = 0
    chunk_idx = 0
    start_time = time.time()
    last_status = start_time
    byte_buf = bytearray()
    last_device_db = {}

    print_header()

    try:
        while True:
            elapsed = time.time() - start_time
            if args.duration > 0 and elapsed > args.duration:
                break

            # Read available bytes
            waiting = ser.in_waiting
            if waiting > 0:
                chunk = ser.read(waiting)
                byte_buf.extend(chunk)

            # Find and parse packets
            while len(byte_buf) >= PACKET_SIZE:
                # Find 0x80 header
                hdr_idx = -1
                for i in range(min(len(byte_buf) - PACKET_SIZE + 1, 2000)):
                    if byte_buf[i] == FRAME_HEADER:
                        # Validate length field
                        if i + 2 < len(byte_buf):
                            length = byte_buf[i + 1] | (byte_buf[i + 2] << 8)
                            if length == PAYLOAD_LENGTH and i + PACKET_SIZE <= len(byte_buf):
                                hdr_idx = i
                                break
                if hdr_idx < 0:
                    # No valid header — discard all but last PACKET_SIZE bytes
                    if len(byte_buf) > PACKET_SIZE:
                        byte_buf = byte_buf[-(PACKET_SIZE):]
                    break

                if hdr_idx > 0:
                    del byte_buf[:hdr_idx]

                if len(byte_buf) < PACKET_SIZE:
                    break

                pkt_bytes = bytes(byte_buf[:PACKET_SIZE])
                del byte_buf[:PACKET_SIZE]

                packet = parse_nsm_packet(pkt_bytes)
                if packet is None:
                    continue

                packet_count += 1

                # Accumulate EEG samples
                for v in packet["eeg"]:
                    eeg_buffer.append(v)

                # Accumulate device band powers
                for b in ["delta", "theta", "alpha", "beta", "gamma"]:
                    device_band_accum[b].append(packet["band_db"][b])

                # Process when we have enough samples
                if len(eeg_buffer) >= args.chunk_size:
                    # Drain exactly chunk_size samples
                    chunk_data = np.array([eeg_buffer.popleft() for _ in range(args.chunk_size)],
                                          dtype=np.float64).reshape(-1, 1)

                    # Average device band powers over this chunk period.
                    # Carry forward last-known values when no new packets arrived.
                    avg_device_db = {}
                    for b in ["delta", "theta", "alpha", "beta", "gamma"]:
                        vals = device_band_accum[b]
                        if vals:
                            avg_device_db[b] = int(np.mean(vals))
                            device_band_accum[b].clear()
                    # If no fresh device data this chunk, reuse last known
                    if not avg_device_db:
                        avg_device_db = last_device_db
                    else:
                        last_device_db = avg_device_db

                    # Run preprocessor
                    result = preprocessor.preprocess(chunk_data, device_band_db=avg_device_db if avg_device_db else None)

                    # Compute device relative ratios
                    device_ratios = band_db_to_relative(avg_device_db) if avg_device_db else None

                    # Discrepancy
                    discrepancy = {}
                    if device_ratios:
                        for b in ["delta", "theta", "alpha", "beta", "gamma"]:
                            pipe_val = result[f"{b}_power"]
                            dev_val = device_ratios[b]
                            discrepancy[b] = abs(pipe_val - dev_val)

                    # Build warnings
                    warnings = []
                    if result["hw_is_saturated"]:
                        warnings.append(f"ADC饱和({result['hw_clipping_pct']:.0f}%)")
                    elif result["hw_clipping_pct"] > 5:
                        warnings.append(f"近饱和({result['hw_clipping_pct']:.0f}%)")
                    if abs(result["hw_dc_offset_uv"]) > 50:
                        warnings.append(f"DC偏置({result['hw_dc_offset_uv']:.0f}uV)")
                    if discrepancy.get("delta", 0) > 0.30:
                        warnings.append(f"δ偏差{discrepancy['delta']*100:.0f}%")
                    if result["sqi"] < 30:
                        warnings.append(f"低SQI({result['sqi']:.0f})")
                    if packet["electrode_alarm"] or packet["electrode_invalid"]:
                        warnings.append("电极异常!")
                    if packet["impedance_high"] or packet["black_imp"] >= 15 or packet["white_imp"] >= 15:
                        warnings.append(f"阻抗高({packet['black_imp']}/{packet['white_imp']})")
                    if result["eeg_tonal_ratio"] > 0.40:
                        warnings.append(f"工频干扰({result['eeg_dominant_hz']:.0f}Hz)")

                    rms_uv = float(np.std(chunk_data))
                    csi = packet["csi"]
                    sqi_val = result["sqi"]

                    pipe_powers = {b: result[f"{b}_power"] for b in
                                   ["delta", "theta", "alpha", "beta", "gamma"]}

                    print_row(elapsed, chunk_idx, result, pipe_powers,
                              device_ratios or {}, discrepancy, rms_uv,
                              sqi_val, csi if csi is not None else -1,
                              packet["black_imp"], packet["white_imp"],
                              warnings)

                    chunk_idx += 1

            # Status update every 3 seconds
            if time.time() - last_status > 3:
                print(f"  [{elapsed:.0f}s] packets={packet_count} chunks={chunk_idx} "
                      f"buf={len(eeg_buffer)}/{args.chunk_size}", end="\r")
                last_status = time.time()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        ser.close()
        print(f"\nTotal: {packet_count} packets, {chunk_idx} chunks processed.")
        print(f"Session duration: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
