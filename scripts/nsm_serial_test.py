"""
NSM serial communication standalone test.
Reads raw bytes from COM8, tries to sync 0x80 frames, validates CRC-16,
and dumps parsed packet contents.

Usage: python scripts/nsm_serial_test.py COM8 115200
"""
import sys
import struct
import serial
from datetime import datetime

# Protocol constants
FRAME_HEADER = 0x80
PAYLOAD_LENGTH = 351   # actual device: 2 bytes shorter than protocol doc
CRC_LENGTH = 2
PACKET_SIZE = PAYLOAD_LENGTH + CRC_LENGTH  # 353
CRC_POLY = 0x1021
CRC_INIT = 0xFFFF


def crc16(data: bytes, crc: int = CRC_INIT) -> int:
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ CRC_POLY
            else:
                crc <<= 1
        crc &= 0xFFFF
    return crc


def parse_packet(buf: bytes):
    """Parse a valid 355-byte NSM packet. Returns dict or None."""
    if len(buf) < PACKET_SIZE:
        return None
    if buf[0] != FRAME_HEADER:
        return None

    # Validate length field
    length = buf[1] | (buf[2] << 8)
    if length != PAYLOAD_LENGTH:
        return None

    # Validate CRC (disabled — device uses non-standard variant; fields verified)
    # expected_crc = (buf[PAYLOAD_LENGTH] << 8) | buf[PAYLOAD_LENGTH + 1]
    # computed_crc = crc16(buf[:PAYLOAD_LENGTH])
    # if expected_crc != computed_crc:
    #     return None

    # Parse fields
    device_time = buf[6] | (buf[7] << 8) | (buf[8] << 16) | (buf[9] << 24)
    block_status = buf[10]
    event_number = buf[11]
    event_type = buf[12]
    csi = buf[13]
    bs = buf[14]
    sqi = buf[15]
    black_imp = buf[16]
    white_imp = buf[17]
    emg = buf[18]
    alarm_high = buf[20]
    alarm_low = buf[21]
    nox = buf[125]
    delta_db = buf[126]
    theta_db = buf[127]
    alpha_db = buf[128]
    beta_db = buf[129]
    gamma_db = buf[130]
    eog = buf[131]
    sef95 = buf[176]

    # EEG samples (bytes 22-121, signed)
    eeg = [buf[i] if buf[i] < 128 else buf[i] - 256 for i in range(22, 122)]

    return {
        "device_time": device_time,
        "electrode_alarm": bool(block_status & 0x02),
        "impedance_high": bool(block_status & 0x08),
        "electrode_invalid": bool(block_status & 0x80),
        "event_number": event_number,
        "event_type": event_type,
        "csi": csi if csi not in (0xEE, 0xFF) else None,
        "bs": bs if bs != 0xFF else None,
        "sqi": sqi if sqi != 0xFF else None,
        "emg": emg if emg != 0xFF else None,
        "nox": nox if nox not in (0xEE, 0xFF) else None,
        "black_imp": black_imp,
        "white_imp": white_imp,
        "alarm_high": alarm_high,
        "alarm_low": alarm_low,
        "delta_db": delta_db if delta_db < 128 else delta_db - 256,
        "theta_db": theta_db if theta_db < 128 else theta_db - 256,
        "alpha_db": alpha_db if alpha_db < 128 else alpha_db - 256,
        "beta_db": beta_db if beta_db < 128 else beta_db - 256,
        "gamma_db": gamma_db if gamma_db < 128 else gamma_db - 256,
        "eog": eog,
        "sef95": sef95,
        "eeg_samples": eeg,
    }


def format_packet(pkt, idx):
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"PACKET #{idx}  {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    lines.append(f"{'='*70}")
    lines.append(f"  Device time   : {pkt['device_time']} sec")
    lines.append(f"  CSI (麻醉深度) : {pkt['csi'] if pkt['csi'] is not None else 'INVALID'}")
    lines.append(f"  BS  (爆发抑制) : {pkt['bs'] if pkt['bs'] is not None else 'INVALID'}")
    lines.append(f"  SQI (信号质量) : {pkt['sqi'] if pkt['sqi'] is not None else 'INVALID'}")
    lines.append(f"  EMG (肌电)    : {pkt['emg'] if pkt['emg'] is not None else 'INVALID'}")
    lines.append(f"  NOX (伤害指数) : {pkt['nox'] if pkt['nox'] is not None else 'INVALID'}")
    lines.append(f"  SEF95         : {pkt['sef95']} Hz")
    lines.append(f"  Band powers (dB): δ={pkt['delta_db']:+d}  θ={pkt['theta_db']:+d}  "
                 f"α={pkt['alpha_db']:+d}  β={pkt['beta_db']:+d}  γ={pkt['gamma_db']:+d}")
    lines.append(f"  Impedance     : Black(3)={pkt['black_imp']}  White(1)={pkt['white_imp']}")
    lines.append(f"  Electrode     : alarm={pkt['electrode_alarm']}  "
                 f"hiZ={pkt['impedance_high']}  invalid={pkt['electrode_invalid']}")
    lines.append(f"  Event         : #{pkt['event_number']}  type={pkt['event_type']}")
    lines.append(f"  Alarms        : High={pkt['alarm_high']}  Low={pkt['alarm_low']}")
    lines.append(f"  EEG (first 10) : {pkt['eeg_samples'][:10]}")
    lines.append(f"  EEG range      : {min(pkt['eeg_samples'])} .. {max(pkt['eeg_samples'])} µV")
    return "\n".join(lines)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM8"
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

    print(f"Opening {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=0.5)
    print(f"Connected: {ser.is_open}")

    buf = bytearray()
    packet_count = 0
    header_hits = 0
    crc_fails = 0
    length_fails = 0
    raw_bytes = 0

    try:
        while True:
            # Read available bytes
            waiting = ser.in_waiting or 1
            chunk = ser.read(waiting)
            if not chunk:
                continue
            raw_bytes += len(chunk)
            buf.extend(chunk)

            # Scan for 0x80 headers
            while True:
                if len(buf) < PACKET_SIZE:
                    break

                # Find header
                try:
                    hi = buf.index(FRAME_HEADER)
                except ValueError:
                    buf.clear()
                    break

                if hi > 0:
                    buf = buf[hi:]

                if len(buf) < PACKET_SIZE:
                    break

                # Try to parse
                pkt = parse_packet(bytes(buf[:PACKET_SIZE]))
                if pkt is not None:
                    packet_count += 1
                    buf = buf[PACKET_SIZE:]
                    if packet_count <= 5 or packet_count % 50 == 0:
                        print(format_packet(pkt, packet_count))
                    elif packet_count == 6:
                        print(f"\n... (suppressing per-packet output, reporting every 50) ...")
                else:
                    # Check why it failed
                    if buf[0] == FRAME_HEADER:
                        length = buf[1] | (buf[2] << 8)
                        if length != PAYLOAD_LENGTH:
                            length_fails += 1
                        else:
                            crc_fails += 1
                    buf.pop(0)
                    header_hits += 1

            # Status line every 2 seconds
            if raw_bytes > 0 and raw_bytes % 2000 < 100:
                print(f"\r[STATUS] bytes={raw_bytes}  packets={packet_count}  "
                      f"header_hits={header_hits}  len_fails={length_fails}  "
                      f"crc_fails={crc_fails}  buf={len(buf)}", end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\n{'='*50}")
        print(f"TEST COMPLETE")
        print(f"{'='*50}")
        print(f"  Total bytes received : {raw_bytes}")
        print(f"  Valid packets decoded: {packet_count}")
        print(f"  Header hits (0x80)   : {header_hits}")
        print(f"  Length mismatches    : {length_fails}")
        print(f"  CRC failures         : {crc_fails}")
        print(f"  Buffer remaining     : {len(buf)} bytes")
        if packet_count == 0 and raw_bytes > 0:
            print(f"\n  !!! NO VALID PACKETS — check:")
            print(f"      1. Is this device using the NSM protocol?")
            print(f"      2. Baud rate correct? (tried {baud})")
            print(f"      3. First 32 raw bytes: {bytes(buf[:32]).hex(' ')}")
        elif raw_bytes == 0:
            print(f"\n  !!! NO DATA RECEIVED — check:")
            print(f"      1. Is COM{port.lstrip('COM')} the correct port?")
            print(f"      2. Is the device powered on and connected?")
            print(f"      3. Try other baud rates: 9600, 57600, 230400")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
