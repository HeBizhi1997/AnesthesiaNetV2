"""
Analyze actual NSM device output to determine real protocol structure.
Dumps raw bytes around each 0x80 header.
"""
import sys
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM8"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

ser = serial.Serial(PORT, BAUD, timeout=1.0)
print(f"Connected to {PORT} at {BAUD}")

buf = bytearray()
analyzed = 0
MAX_ANALYZE = 5

try:
    while analyzed < MAX_ANALYZE:
        waiting = ser.in_waiting or 1
        chunk = ser.read(waiting)
        if not chunk:
            continue
        buf.extend(chunk)

        while analyzed < MAX_ANALYZE:
            try:
                hi = buf.index(0x80)
            except ValueError:
                break
            if hi > 0:
                buf = buf[hi:]

            if len(buf) < 32:
                break

            # Dump first 32 bytes around header
            print(f"\n{'='*60}")
            print(f"Header at offset +0: 0x80 found, total buf={len(buf)} bytes")
            print(f"Raw bytes (hex): {bytes(buf[:355]).hex(' ') if len(buf) >= 355 else bytes(buf).hex(' ')}")

            # Try length at different positions
            len_lsb = buf[1]
            len_msb = buf[2]
            reported_len = len_lsb | (len_msb << 8)
            print(f"\n  Byte[1-2] as length LSB|MSB<<8 = {reported_len} (0x{reported_len:04X})")
            print(f"  Byte[1]={buf[1]} (0x{buf[1]:02X})  Byte[2]={buf[2]} (0x{buf[2]:02X})")

            # Show key potential fields
            print(f"\n  Byte[0]      = 0x{buf[0]:02X}  (header, should be 0x80)")
            print(f"  Byte[1-2]    = length or other")
            print(f"  Byte[3-5]    = 0x{buf[3]:02X} 0x{buf[4]:02X} 0x{buf[5]:02X}")
            print(f"  Byte[6-9]    = 0x{buf[6]:02X} 0x{buf[7]:02X} 0x{buf[8]:02X} 0x{buf[9]:02X}  (device time?)")
            print(f"  Byte[10]     = 0x{buf[10]:02X}  (block status?)")
            print(f"  Byte[11]     = 0x{buf[11]:02X}  (event number?)")
            print(f"  Byte[12]     = 0x{buf[12]:02X}  (event type?)")
            print(f"  Byte[13]     = 0x{buf[13]:02X}  (0x{buf[13]:3d} = CSI?)")
            print(f"  Byte[14]     = 0x{buf[14]:02X}  (0x{buf[14]:3d} = BS?)")
            print(f"  Byte[15]     = 0x{buf[15]:02X}  (0x{buf[15]:3d} = SQI?)")
            print(f"  Byte[16-17]  = 0x{buf[16]:02X} 0x{buf[17]:02X}  (impedance?)")

            # Check if reported length makes sense
            if 50 < reported_len < 1000:
                print(f"\n  >>> Length {reported_len} looks plausible!")
                print(f"  >>> Full packet would be {reported_len + 3} bytes (header + length[2] + payload)")
                print(f"  >>> Or {reported_len + 2} bytes if length includes CRC")
                # Try to show end-of-packet markers
                end_offset = reported_len
                if len(buf) > end_offset:
                    print(f"  >>> Byte at offset {end_offset}: 0x{buf[end_offset]:02X}")
                    print(f"  >>> Next header search would start here")

            # Also try: maybe there's no length field, just a fixed-size packet
            # Check if another 0x80 appears at a regular interval
            rest = buf[1:]
            try:
                next_header = rest.index(0x80)
                print(f"\n  Next 0x80 at offset +{next_header + 1} → packet size = {next_header + 1}")
            except ValueError:
                print(f"\n  No next 0x80 found in remaining {len(rest)} bytes")

            analyzed += 1
            # Skip past this candidate for next iteration
            buf = buf[1:]

except KeyboardInterrupt:
    pass
finally:
    ser.close()
    print(f"\nDone. Analyzed {analyzed} header occurrences.")
