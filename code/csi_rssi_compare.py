"""
RSSI vs CSI comparison.
Builds RSSI-only features from the same recordings, evaluates with the same
session-grouped protocol, and compares against the CSI features in features.csv.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
FS, WIN, STEP = 30, 300, 150         # 10 s windows, matches the CSI pipeline
# ---------- build RSSI features ----------
def rssi_series(path):
    """Extract the RSSI value (field index 3) of every valid CSI_DATA line."""
    vals = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            i = line.find("CSI_DATA")
            if i < 0:
                continue
            parts = line[i:].split(",")
            try:
                vals.append(float(parts[3]))       # RSSI in dBm
            except (IndexError, ValueError):
                continue
    return np.array(vals)

def rssi_features(w):
    f = {}
    f["rssi_mean"]  = w.mean()
    f["rssi_std"]   = w.std()
    f["rssi_min"]   = w.min()
    f["rssi_max"]   = w.max()
    f["rssi_range"] = w.max() - w.min()
    d = np.abs(np.diff(w))
    f["rssi_motion_mean"] = d.mean()
    f["rssi_motion_max"]  = d.max()
    f["rssi_motion_std"]  = d.std()
    ts = w - w.mean()
    fft = np.abs(np.fft.rfft(ts))**2
    freqs = np.fft.rfftfreq(len(ts), d=1/FS)
    tot = fft.sum() + 1e-12
    f["rssi_band_0_1Hz"] = fft[(freqs > 0.1) & (freqs <= 1)].sum() / tot
    f["rssi_band_1_3Hz"] = fft[(freqs > 1) & (freqs <= 3)].sum() / tot
    f["rssi_band_3_8Hz"] = fft[(freqs > 3) & (freqs <= 8)].sum() / tot
    p = fft / tot
    f["rssi_entropy"] = -(p * np.log2(p + 1e-12)).sum()
    return f

rows = []
for path in sorted(glob.glob("data/*.csv")):
    if "room1p" in path.replace("\\", "/"):
        continue
    label = int(path.split("label")[1].split(".")[0])
    r = rssi_series(path)
    if len(r) < WIN:
        continue
    for s in range(0, len(r) - WIN + 1, STEP):
        f = rssi_features(r[s:s+WIN])
        f["label"] = label
        f["source"] = path
        rows.append(f)

rssi_df = pd.DataFrame(rows)
rssi_df.to_csv("features_rssi.csv", index=False)
print(f"RSSI features: {len(rssi_df)} windows, {len(rssi_df.columns)-2} features\n")

# ---------- evaluate both feature sets ----------
def evaluate(df, name):
    X = df.drop(columns=["label", "source"]).values
    y = df["label"].values
    g = df["source"].values
    gkf = GroupKFold(n_splits=5)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "SVM (RBF)":    make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=42)),
    }
    out = {}
    print(f"########## {name} ({X.shape[1]} features, {X.shape[0]} windows) ##########")
    for mname, m in models.items():
        acc4 = cross_val_score(m, X, y, cv=gkf, groups=g)
        accP = cross_val_score(m, X, (y > 0).astype(int), cv=gkf, groups=g)
        acc3 = cross_val_score(m, X, np.minimum(y, 2), cv=gkf, groups=g)
        print(f"{mname:14s} presence {accP.mean()*100:5.1f}% ± {accP.std()*100:4.1f}   "
              f"level {acc3.mean()*100:5.1f}%   exact {acc4.mean()*100:5.1f}%")
        out[mname] = (accP.mean()*100, acc3.mean()*100, acc4.mean()*100)
    print()
    return out

csi_df = pd.read_csv("features.csv")
csi_res  = evaluate(csi_df,  "CSI features")
rssi_res = evaluate(rssi_df, "RSSI features")

# ---------- comparison figure ----------
tasks = ["Presence\n(0 vs 1+)", "Level\n(0/1/2+)", "Exact count\n(0-3)"]
best_csi  = [max(csi_res[m][i]  for m in csi_res)  for i in range(3)]
best_rssi = [max(rssi_res[m][i] for m in rssi_res) for i in range(3)]
x = np.arange(3); w = 0.35
plt.figure(figsize=(7, 4.5))
plt.bar(x - w/2, best_csi,  w, label="CSI (192 subcarriers)")
plt.bar(x + w/2, best_rssi, w, label="RSSI (1 value/packet)")
plt.xticks(x, tasks); plt.ylabel("Accuracy [%]")
plt.title("CSI vs RSSI - session-grouped cross-validation")
plt.axhline(25, ls="--", c="grey", lw=1)
plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig("csi_vs_rssi.png", dpi=200)
print("Saved csi_vs_rssi.png")
plt.show()