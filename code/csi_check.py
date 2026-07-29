"""Check all logged CSV files: counts valid vs truncated lines."""
import glob

for path in sorted(glob.glob("data/*.csv")):
    total, valid = 0, 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "CSI_DATA" not in line:
                continue
            total += 1
            try:
                start = line.rindex('[') + 1
                end   = line.rindex(']')
                if len(line[start:end].split(",")) == 384:
                    valid += 1
            except ValueError:
                pass
    pct = 100 * valid / total if total else 0
    flag = "OK " if pct >= 90 else ("~  " if pct >= 80 else "BAD")
    print(f"{flag} {path}: {valid}/{total} valid ({pct:.1f}%)")

