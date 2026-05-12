"""
Brute-force CRC variant test. Captures one NSM packet and tries different CRC-16 variants.
"""
import sys
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM8"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

ser = serial.Serial(PORT, BAUD, timeout=2.0)
print(f"Connected to {PORT} at {BAUD}")

# Collect enough data to get a valid packet
buf = bytearray()
PAYLOAD_LENGTH = 351
PACKET_SIZE = PAYLOAD_LENGTH + 2

# Read until we have enough
while len(buf) < PACKET_SIZE * 2:
    buf.extend(ser.read(ser.in_waiting or 1))

# Find the first 0x80 header
hi = buf.index(0x80)
buf = buf[hi:]

if len(buf) < PACKET_SIZE:
    print(f"Not enough data: {len(buf)} < {PACKET_SIZE}")
    ser.close()
    exit()

packet = bytes(buf[:PACKET_SIZE])
data = packet[:PAYLOAD_LENGTH]
expected_crc = (packet[PAYLOAD_LENGTH] << 8) | packet[PAYLOAD_LENGTH + 1]
print(f"Packet length: {len(packet)} bytes")
print(f"Data length: {PAYLOAD_LENGTH} bytes")
print(f"Expected CRC: 0x{expected_crc:04X}")
print(f"Data first 32 bytes: {data[:32].hex(' ')}")
print()

# CRC-16 variants to test
results = []

# Standard CRC-16
def crc16(data, poly, init, refin=False, refout=False, xorout=0):
    crc = init
    for byte in data:
        b = byte
        if refin:
            b = int(f'{b:08b}'[::-1], 2)
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
        crc &= 0xFFFF
    if refout:
        crc = int(f'{crc:016b}'[::-1], 2)
    return crc ^ xorout

# Test different variants
variants = [
    # (name, poly, init, refin, refout, xorout)
    ("CRC-16/XMODEM     ", 0x1021, 0x0000, False, False, 0x0000),
    ("CRC-16/CCITT-FALSE", 0x1021, 0xFFFF, False, False, 0x0000),
    ("CRC-16/CCITT      ", 0x1021, 0x0000, True,  True,  0x0000),
    ("CRC-16/MODBUS     ", 0x8005, 0xFFFF, True,  True,  0x0000),
    ("CRC-16/IBM        ", 0x8005, 0x0000, True,  True,  0x0000),
    ("CRC-16/DNP        ", 0x3D65, 0x0000, True,  True,  0xFFFF),
    ("CRC-16/GENIBUS    ", 0x1021, 0xFFFF, False, False, 0xFFFF),
    ("CRC-16/KERMIT     ", 0x1021, 0x0000, True,  True,  0x0000),
]

for name, poly, init, refin, refout, xorout in variants:
    crc = crc16(data, poly, init, refin, refout, xorout)
    match = "✓ MATCH" if crc == expected_crc else ""
    results.append((name, crc, match))

# Also try with different data lengths (CRC might start at a different offset)
for offset in range(1, 5):
    crc = crc16(data[offset:], 0x1021, 0xFFFF, False, False, 0x0000)
    if crc == expected_crc:
        results.append((f"DOC CRC, skip {offset} bytes", crc, "✓ MATCH"))

for name, crc, match in results:
    print(f"  {name}: 0x{crc:04X}  {match}")

# Also show the raw CRC bytes position
print(f"\nRaw CRC bytes at end: 0x{packet[PAYLOAD_LENGTH]:02X} 0x{packet[PAYLOAD_LENGTH+1]:02X}")
print(f"CRC as LSB-first:    0x{packet[PAYLOAD_LENGTH] | (packet[PAYLOAD_LENGTH+1] << 8):04X}")

ser.close()
