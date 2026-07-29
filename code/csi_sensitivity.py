"""
Parameter sensitivity study: window length, packet rate, subcarrier count.
Produces three thesis figures from the existing recordings.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

FS_BASE = 30

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

print("Loading all recordings once...")
DATA = {}
for path in sorted(glob.glob("data/*.csv")):
    if "room1p" in path.replace("\\", "/"):
        continue
    A = parse_file(path)
    if A.shape[0] > 200:
        DATA[path] = A
print(f"{len(DATA)} sessions loaded\n")

def features(W, fs):
    f = {}
    std_t = W.std(axis=0)
    f["std_mean"] = std_t.mean(); f["std_max"] = std_t.max()
    f["std_p90"] = np.percentile(std_t, 90); f["std_spread"] = std_t.std()
    d = np.abs(np.diff(W, axis=0))
    f["motion_mean"] = d.mean(); f["motion_max"] = d.mean(axis=1).max()
    f["motion_std"] = d.mean(axis=1).std()
    ms = W.mean(axis=0)
    f["spec_std"] = ms.std(); f["spec_range"] = ms.max() - ms.min()
    sub = W[:, ::max(1, W.shape[1]//20)]
    C = np.corrcoef(sub.T); iu = np.triu_indices_from(C, k=1)
    f["corr_mean"] = np.nanmean(C[iu]); f["corr_std"] = np.nanstd(C[iu])
    ts = W.mean(axis=1); ts = ts - ts.mean()
    fft = np.abs(np.fft.rfft(ts))**2
    fr = np.fft.rfftfreq(len(ts), d=1/fs); tot = fft.sum() + 1e-12
    f["b01"] = fft[(fr > 0.1) & (fr <= 1)].sum()/tot
    f["b13"] = fft[(fr > 1) & (fr <= 3)].sum()/tot
    f["b38"] = fft[(fr > 3) & (fr <= 8)].sum()/tot
    p = fft/tot; f["entropy"] = -(p*np.log2(p+1e-12)).sum()
    return f

def build(win_sec=5.0, decim=1, n_sub=None):
    fs = FS_BASE / decim
    win = int(win_sec * fs); step = win // 2
    rows = []
    for path, A0 in DATA.items():
        label = int(path.split("label")[1].split(".")[0])
        A = A0[::decim]
        active = A.mean(axis=0) > 1.0
        A = A[:, active]
        if n_sub is not None and n_sub < A.shape[1]:
            idx = np.linspace(0, A.shape[1]-1, n_sub).astype(int)
            A = A[:, idx]
        A = A / (A.mean(axis=1, keepdims=True) + 1e-12)
        if A.shape[0] < win:
            continue
        for s in range(0, A.shape[0]-win+1, step):
            f = features(A[s:s+win], fs)
            f["label"] = label; f["source"] = path
            rows.append(f)
    return pd.DataFrame(rows)

def score(df):
    X = df.drop(columns=["label","source"]).values
    y = df["label"].values; g = df["source"].values
    m = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    gkf = GroupKFold(n_splits=5)
    pres = cross_val_score(m, X, (y>0).astype(int), cv=gkf, groups=g)
    lvl  = cross_val_score(m, X, np.minimum(y,2),   cv=gkf, groups=g)
    return pres.mean()*100, pres.std()*100, lvl.mean()*100

# ---- 1) window length ----
print("### Window length ###")
wins = [1, 2, 3, 5, 8, 10]
res_w = [score(build(win_sec=w)) for w in wins]
for w, r in zip(wins, res_w):
    print(f"  {w:2d}s: presence {r[0]:5.1f}% ± {r[1]:4.1f}   level {r[2]:5.1f}%")

# ---- 2) packet rate ----
print("\n### Packet rate ###")
decims = [1, 2, 3, 6]
rates = [FS_BASE//d for d in decims]
res_r = [score(build(decim=d)) for d in decims]
for r_, r in zip(rates, res_r):
    print(f"  {r_:2d} Hz: presence {r[0]:5.1f}% ± {r[1]:4.1f}   level {r[2]:5.1f}%")

# ---- 3) subcarrier count ----
print("\n### Subcarrier count ###")
subs = [8, 16, 32, 64, 166]
res_s = [score(build(n_sub=n)) for n in subs]
for n, r in zip(subs, res_s):
    print(f"  {n:3d} sc: presence {r[0]:5.1f}% ± {r[1]:4.1f}   level {r[2]:5.1f}%")

# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, xs, res, xlabel, title in [
    (ax[0], wins,  res_w, "Window length [s]",  "Effect of window length"),
    (ax[1], rates, res_r, "Packet rate [Hz]",   "Effect of packet rate"),
    (ax[2], subs,  res_s, "Subcarriers used",   "Effect of subcarrier count")]:
    a.errorbar(xs, [r[0] for r in res], yerr=[r[1] for r in res],
               marker="o", capsize=4, label="Presence (0 vs 1+)")
    a.plot(xs, [r[2] for r in res], marker="s", label="Level (0/1/2+)")
    a.set_xlabel(xlabel); a.set_ylabel("Accuracy [%]"); a.set_title(title)
    a.grid(alpha=0.3); a.legend()
ax[2].set_xscale("log", base=2)
plt.tight_layout(); plt.savefig("parameter_sensitivity.png", dpi=200)
print("\nSaved parameter_sensitivity.png")
plt.show()