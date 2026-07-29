"""
CSI Logger - reads CSI_DATA lines from ESP32 RX board and saves to CSV.
Usage:  python csi_logger.py --port COM8 --label 0 --duration 60 --tag room3 --note "empty a"
"""
import serial, argparse, csv, time, os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="COM8", help="Serial port of RX board")
parser.add_argument("--baud", type=int, default=921600, help="Baud rate")
parser.add_argument("--label", type=int, required=True, help="Number of people in room (ground truth)")
parser.add_argument("--duration", type=int, default=60, help="Recording length in seconds")
parser.add_argument("--tag", default="room1", help="Environment tag, goes into the filename")
parser.add_argument("--note", default="", help="Free-text note about the session")
args = parser.parse_args()

filename = f"data/csi_{args.tag}_{datetime.now():%Y%m%d_%H%M%S}_label{args.label}.csv"
os.makedirs("data", exist_ok=True)

ser = serial.Serial(args.port, args.baud, timeout=1)
try:
    ser.set_buffer_size(rx_size=262144, tx_size=16384)
except Exception:
    pass

print(f"Recording {args.duration}s with label={args.label} tag={args.tag} -> {filename}")

count, bad = 0, 0
t_end = time.time() + args.duration

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pc_timestamp", "label", "tag", "note", "raw_line"])
    while time.time() < t_end:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line.startswith("CSI_DATA"):
            writer.writerow([time.time(), args.label, args.tag, args.note, line])
            count += 1
            if count % 100 == 0:
                print(f"  {count} packets...", end="\r")
        elif line:
            bad += 1

ser.close()
print(f"\nDone. Saved {count} CSI packets ({bad} non-CSI lines skipped) to {filename}")