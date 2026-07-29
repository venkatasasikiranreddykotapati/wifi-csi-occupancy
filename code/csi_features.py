"""
CSI Feature Extractor v3: 10 s windows, 20 features.
Writes TWO files:
  features.csv          - main campaign (rooms 1/2/3), used by all analysis scripts
  features_persons.csv  - person-independence study (csi_room1p*), used by csi_loso.py
"""
import glob
import numpy as np
import pandas as pd

FS = 30
WIN = 300          # 10 s windows
STEP = 150         # 50% overlap

def parse_file(path):
    amps = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "CSI_DATA" not in line:
                continue
            try:
                s = line.rindex('[') + 1
                e = line.rindex(']')
                vals = [int(x) for x in line[s:e].split(",")]
                if len(vals) != 384:
                    continue
                iq = np.array(vals, dtype=float).reshape(-1, 2)
                amps.append(np.sqrt(iq[:, 0]**2 + iq[:, 1]**2))
            except (ValueError, IndexError):
                continue
    return np.array(amps)

def window_features(W):
    f = {}
    std_t = W.std(axis=0)
    f["std_mean"] = std_t.mean()
    f["std_max"] = std_t.max()
    f["std_p90"] = np.percentile(std_t, 90)
    f["std_spread"] = std_t.std()
    d = np.abs(np.diff(W, axis=0))
    f["motion_mean"] = d.mean()
    f["motion_max"] = d.mean(axis=1).max()
    f["motion_std"] = d.mean(axis=1).std()
    ms = W.mean(axis=0)
    f["spec_std"] = ms.std()
    f["spec_range"] = ms.max() - ms.min()
    nb = max(1, W.shape[1] // 4)
    for b in range(4):
        f[f"band{b}_std"] = W[:, b*nb:(b+1)*nb].std(axis=0).mean()
    sub = W[:, ::8]
    C = np.corrcoef(sub.T)
    iu = np.triu_indices_from(C, k=1)
    f["corr_mean"] = np.nanmean(C[iu])
    f["corr_std"] = np.nanstd(C[iu])
    ts = W.mean(axis=1); ts = ts - ts.mean()
    fft = np.abs(np.fft.rfft(ts))**2
    fr = np.fft.rfftfreq(len(ts), d=1/FS)
    tot = fft.sum() + 1e-12
    f["band_0_1Hz"] = fft[(fr > 0.1) & (fr <= 1)].sum()/tot
    f["band_1_3Hz"] = fft[(fr > 1) & (fr <= 3)].sum()/tot
    f["band_3_8Hz"] = fft[(fr > 3) & (fr <= 8)].sum()/tot
    p = fft/tot
    f["spec_entropy"] = -(p*np.log2(p+1e-12)).sum()
    f["peak_freq"] = fr[np.argmax(fft[1:]) + 1] if len(fft) > 1 else 0.0
    return f

print(">>> FEATURE EXTRACTOR v3 (10 s windows) <<<")
main_rows, person_rows = [], []

for path in sorted(glob.glob("data/*.csv")):
    p = path.replace("\\", "/")
    is_person = "csi_room1p" in p
    label = int(p.split("label")[1].split(".")[0])
    A = parse_file(path)
    if A.shape[0] < WIN:
        print(f"skip {path}: only {A.shape[0]} packets")
        continue
    active = A.mean(axis=0) > 1.0
    A = A[:, active]
    A = A / (A.mean(axis=1, keepdims=True) + 1e-12)
    n = 0
    for s in range(0, A.shape[0]-WIN+1, STEP):
        f = window_features(A[s:s+WIN])
        f["label"] = label
        f["source"] = path
        (person_rows if is_person else main_rows).append(f)
        n += 1
    tag = "PERSON" if is_person else "main  "
    print(f"[{tag}] {path}: {A.shape[0]} pkts, {A.shape[1]} sc, {n} windows")

main = pd.DataFrame(main_rows)
main.to_csv("features.csv", index=False)
print("\n=== features.csv (main campaign) ===")
print(main["label"].value_counts().sort_index().rename("windows per class"))
print(f"Total: {len(main)} windows, {len(main.columns)-2} features")

if person_rows:
    per = pd.DataFrame(person_rows)
    per.to_csv("features_persons.csv", index=False)
    print("\n=== features_persons.csv (person study) ===")
    print(f"Total: {len(per)} windows from {per['source'].nunique()} sessions")