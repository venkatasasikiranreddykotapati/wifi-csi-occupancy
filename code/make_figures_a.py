"""
Thesis figures, part A: generated from the data.
Outputs 300 dpi PNG + PDF into figures/
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})
os.makedirs("figures", exist_ok=True)

def save(fig, name):
    fig.savefig(f"figures/{name}.png")
    fig.savefig(f"figures/{name}.pdf")
    plt.close(fig)
    print("saved figures/" + name)

def parse(path, limit=1800):
    amps = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "CSI_DATA" not in line:
                continue
            try:
                s = line.rindex('[') + 1; e = line.rindex(']')
                v = [int(x) for x in line[s:e].split(",")]
                if len(v) != 384:
                    continue
                iq = np.array(v, dtype=float).reshape(-1, 2)
                amps.append(np.sqrt(iq[:, 0]**2 + iq[:, 1]**2))
                if len(amps) >= limit:
                    break
            except (ValueError, IndexError):
                continue
    return np.array(amps)

# ---------- FIG 1: raw CSI, empty vs three people ----------
f0 = sorted(glob.glob("data/csi_20260719_13*label0.csv"))[0]
f3 = sorted(glob.glob("data/csi_20260719_18*label3.csv"))[0]
fig, ax = plt.subplots(2, 2, figsize=(9, 5.5))
for col, (path, title) in enumerate([(f0, "Vacant room"), (f3, "Three occupants")]):
    A = parse(path)
    t = np.arange(A.shape[0]) / 30.0
    im = ax[0][col].imshow(A.T, aspect="auto", origin="lower",
                           extent=[0, t[-1], 0, A.shape[1]], cmap="viridis")
    ax[0][col].set_title(title)
    ax[0][col].set_xlabel("Time [s]"); ax[0][col].set_ylabel("Subcarrier index")
    ax[0][col].grid(False)
    fig.colorbar(im, ax=ax[0][col], label="Amplitude")
    for sc in (70, 100, 150):
        ax[1][col].plot(t, A[:, sc], lw=0.6, label=f"sc {sc}")
    ax[1][col].set_xlabel("Time [s]"); ax[1][col].set_ylabel("Amplitude")
    ax[1][col].legend(ncol=3, fontsize=7)
fig.tight_layout()
save(fig, "fig_raw_csi")

# ---------- FIG 2 + 3: confusion matrix and feature importance ----------
df = pd.read_csv("features.csv")
X = df.drop(columns=["label", "source"]).values
y = df["label"].values
g = df["source"].values
names = df.drop(columns=["label", "source"]).columns

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
pred = cross_val_predict(rf, X, y, cv=GroupKFold(n_splits=5), groups=g)
cm = confusion_matrix(y, pred)
cmn = cm / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]:.0f}%", ha="center", va="center",
                fontsize=8, color="white" if cmn[i, j] > 55 else "black")
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xlabel("Predicted occupancy"); ax.set_ylabel("True occupancy")
ax.set_title("Random Forest, session-grouped CV")
ax.grid(False)
fig.colorbar(im, ax=ax, label="Row-normalised [%]")
save(fig, "fig_confusion_matrix")

rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=names).sort_values()
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.barh(imp.index, imp.values, color="#1f77b4")
ax.set_xlabel("Mean decrease in impurity")
ax.set_title("Random Forest feature importance")
save(fig, "fig_feature_importance")

print("\nPart A complete.")