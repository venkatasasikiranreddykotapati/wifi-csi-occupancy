"""
CSI Visualizer v2 - robust line-based parser (immune to truncated lines).
Usage: python csi_plot.py data/file1.csv data/file2.csv
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_csi(path):
    amps, times = [], []
    total, bad = 0, 0
    with open(path, "r", errors="ignore") as f:
        next(f)                                   # skip header row
        for line in f:
            if "CSI_DATA" not in line:
                continue
            total += 1
            try:
                # PC timestamp = first field of the CSV row
                ts = float(line.split(",", 1)[0])
                # CSI array = between the LAST '[' and the LAST ']'
                start = line.rindex('[') + 1
                end   = line.rindex(']')
                vals = [int(x) for x in line[start:end].split(",")]
                if len(vals) != 384:
                    bad += 1
                    continue
                iq = np.array(vals, dtype=float).reshape(-1, 2)
                amps.append(np.sqrt(iq[:, 0]**2 + iq[:, 1]**2))
                times.append(ts)
            except (ValueError, IndexError):
                bad += 1
                continue
    A = np.array(amps)
    print(f"{path}: {total} CSI lines found, {A.shape[0]} valid, {bad} truncated/dropped")
    return np.array(times, dtype=float), A

def main():
    files = sys.argv[1:]
    if not files:
        print("Give me at least one CSV file!"); return
    fig, axes = plt.subplots(2, len(files), figsize=(7*len(files), 8), squeeze=False)
    for i, path in enumerate(files):
        t, A = load_csi(path)
        if A.size == 0:
            print(f"  !! no valid packets in {path}"); continue
        t = t - t[0]
        im = axes[0][i].imshow(A.T, aspect="auto", origin="lower",
                               extent=[0, t[-1], 0, A.shape[1]], cmap="viridis")
        axes[0][i].set_title(path.split("/")[-1].split("\\")[-1])
        axes[0][i].set_xlabel("Time [s]")
        axes[0][i].set_ylabel("Subcarrier index")
        fig.colorbar(im, ax=axes[0][i], label="Amplitude")
        for sc in (70, 100, 150):
            axes[1][i].plot(t, A[:, sc], label=f"subcarrier {sc}", linewidth=0.7)
        axes[1][i].set_xlabel("Time [s]")
        axes[1][i].set_ylabel("Amplitude")
        axes[1][i].legend()
        axes[1][i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("csi_comparison.png", dpi=200)
    print("Saved csi_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()