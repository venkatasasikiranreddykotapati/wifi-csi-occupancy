"""
Thesis figures, part B: results figures from the locked final numbers.
Outputs 300 dpi PNG + PDF into figures/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})
os.makedirs("figures", exist_ok=True)
C = {"rf": "#1f77b4", "svm": "#ff7f0e", "knn": "#2ca02c", "base": "#999999"}

def save(fig, name):
    fig.savefig(f"figures/{name}.png"); fig.savefig(f"figures/{name}.pdf")
    plt.close(fig); print("saved figures/" + name)

# ---------- FIG 4: main results ----------
tasks = ["Presence\n(0 vs 1+)", "Level\n(0/1/2+)", "Exact count\n(0-3)"]
base = [50.0, 33.3, 25.0]
knn  = [85.0, 65.2, 51.6]; knn_s = [7.4, 10.3, 11.7]
svm  = [86.2, 61.0, 46.9]; svm_s = [4.4, 9.8, 6.4]
rf   = [85.2, 67.1, 52.4]; rf_s  = [8.5, 9.3, 6.8]
x = np.arange(3); w = 0.2
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(x - 1.5*w, base, w, color=C["base"], label="Majority baseline")
ax.bar(x - 0.5*w, knn, w, yerr=knn_s, capsize=3, color=C["knn"], label="kNN (k=5)")
ax.bar(x + 0.5*w, svm, w, yerr=svm_s, capsize=3, color=C["svm"], label="SVM (RBF)")
ax.bar(x + 1.5*w, rf,  w, yerr=rf_s,  capsize=3, color=C["rf"],  label="Random Forest")
ax.set_xticks(x); ax.set_xticklabels(tasks)
ax.set_ylabel("Balanced accuracy [%]"); ax.set_ylim(0, 100)
ax.set_title("Classification performance, session-grouped cross-validation")
ax.legend(ncol=2)
save(fig, "fig_main_results")

# ---------- FIG 5: cross-environment transfer matrix ----------
rooms = ["Bedroom", "Kitchen", "Classroom"]
M = np.array([[80.5, 61.2, 93.0],
              [67.3, 83.1, 96.2],
              [80.3, 61.4, 94.0]])
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(M, cmap="RdYlGn", vmin=50, vmax=100)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                fontweight="bold" if i == j else "normal", fontsize=10)
ax.set_xticks(range(3), rooms); ax.set_yticks(range(3), rooms)
ax.set_xlabel("Test environment"); ax.set_ylabel("Training environment")
ax.set_title("Cross-environment transfer, presence detection\n(diagonal = within-room)")
ax.grid(False); fig.colorbar(im, ax=ax, label="Balanced accuracy [%]")
save(fig, "fig_crossroom_matrix")

# ---------- FIG 6: generalisation strategies ----------
labels = ["Within\nroom", "Leave-one-\nroom-out", "Combined\ntraining"]
bed = [80.5, 74.7, None]; kit = [83.1, 69.0, None]; cls = [94.0, 96.0, None]
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(3); w = 0.25
ax.bar(x[:2] - w, bed[:2], w, color=C["rf"], label="Bedroom")
ax.bar(x[:2],     kit[:2], w, color=C["svm"], label="Kitchen")
ax.bar(x[:2] + w, cls[:2], w, color=C["knn"], label="Classroom")
ax.bar(x[2], 84.1, w*2, color="#7f4fbf", label="All rooms pooled")
ax.errorbar(x[2], 84.1, yerr=7.6, fmt="none", ecolor="black", capsize=4)
ax.axhline(50, ls="--", c="grey", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Balanced accuracy [%]"); ax.set_ylim(0, 105)
ax.set_title("Effect of training strategy on generalisation")
ax.legend(ncol=2, fontsize=8)
save(fig, "fig_generalisation")

# ---------- FIG 7: person independence ----------
subj = ["A", "B", "C", "S"]
ba = [81.0, 82.3, 79.2, 81.2]; auc = [87.0, 98.1, 96.8, 93.0]
fig, ax = plt.subplots(figsize=(5.5, 4))
x = np.arange(4); w = 0.35
ax.bar(x - w/2, ba, w, color=C["rf"], label="Balanced accuracy")
ax.bar(x + w/2, auc, w, color=C["svm"], label="ROC-AUC")
ax.axhline(82.3, ls="--", c="black", lw=1.2, label="Subject seen in training (82.3%)")
ax.axhline(50, ls=":", c="grey", lw=1, label="Chance")
ax.set_xticks(x); ax.set_xticklabels([f"Subject {s}" for s in subj])
ax.set_ylabel("Performance [%]"); ax.set_ylim(0, 105)
ax.set_title("Leave-one-subject-out evaluation (gap: 1.4 pp)")
ax.legend(fontsize=8, loc="lower right")
save(fig, "fig_person_independence")

# ---------- FIG 8: few-shot adaptation ----------
n = [0, 2, 4, 6, 8, 10, 12, 14, 16]
sc_ba = [62.2, 69.0, 66.5, 68.1, 72.6, 71.9, 71.9, 74.1, 79.5]
sc_sd = [16.9, 17.2, 16.3, 17.0, 15.7, 18.0, 15.8, 15.3, 15.5]
sc_auc = [69.8, 76.8, 77.1, 77.9, 84.9, 82.2, 83.9, 89.7, 93.3]
to_ba = [np.nan, 55.4, 48.5, 61.1, 68.4, 68.2, 72.7, 78.5, 79.0]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.errorbar(n, sc_ba, yerr=sc_sd, marker="o", capsize=3, color=C["rf"],
            label="Source + calibration (bal. acc.)")
ax.plot(n, sc_auc, marker="s", ls="--", color=C["svm"], label="Source + calibration (AUC)")
ax.plot(n, to_ba, marker="^", color=C["knn"], label="Calibration only (bal. acc.)")
ax.axhline(50, ls=":", c="grey", lw=1, label="Chance")
ax.set_xlabel("Calibration sessions from the target environment (1 min each)")
ax.set_ylabel("Performance [%]")
ax.set_title("Few-shot adaptation to an unseen environment")
ax.legend(fontsize=8, loc="lower right")
save(fig, "fig_fewshot")

# ---------- FIG 9: CSI vs RSSI ----------
tasks2 = ["Presence", "Level (0/1/2+)", "Exact count"]
csi = [88.0, 67.8, 52.3]; rssi = [89.2, 66.1, 48.6]
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - w/2, csi, w, color=C["rf"], label="CSI (166 subcarriers)")
ax.bar(x + w/2, rssi, w, color=C["svm"], label="RSSI (1 value/packet)")
for i, (a, b) in enumerate(zip(csi, rssi)):
    ax.text(i - w/2, a + 1, f"{a:.1f}", ha="center", fontsize=8)
    ax.text(i + w/2, b + 1, f"{b:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(tasks2)
ax.set_ylabel("Accuracy [%]"); ax.set_ylim(0, 100)
ax.set_title("CSI versus RSSI features (Random Forest)")
ax.legend()
save(fig, "fig_csi_vs_rssi")

# ---------- FIG 10: embedded latency ----------
cfg = ["10 s\n166 sc", "10 s\n32 sc", "10 s\n16 sc", "5 s\n166 sc", "5 s\n16 sc"]
feat = [1.95, 1.00, 0.90, 1.43, 0.85]
infer = [8.61, 8.25, 8.18, 8.54, 8.12]
x = np.arange(5)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, feat, 0.6, color=C["knn"], label="Feature extraction")
ax.bar(x, infer, 0.6, bottom=feat, color=C["rf"], label="Model inference")
for i, (a, b) in enumerate(zip(feat, infer)):
    ax.text(i, a + b + 0.2, f"{a+b:.1f} ms", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cfg)
ax.set_ylabel("Processing time [ms]")
ax.set_title("Inference latency on Raspberry Pi 5\n(prediction interval 2000 ms)")
ax.legend()
save(fig, "fig_embedded_latency")

print("\nPart B complete - 7 figures written to figures/")