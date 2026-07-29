"""
Few-shot environment adaptation (balanced calibration).
Calibration sessions are added in balanced pairs (1 empty + 1 occupied),
which is how calibration is performed in practice.
Also compares against training on target data only.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

RNG = np.random.default_rng(42)
REPEATS = 40

df = pd.read_csv("features.csv")

def get_room(s):
    m = re.search(r"csi_(room\d+)_", str(s).replace("\\", "/"))
    return m.group(1) if m else "room1"

df["room"] = df["source"].apply(get_room)
df = df[df["label"].isin([0, 1])].copy()
FEATS = [c for c in df.columns if c not in ("label", "source", "room")]

src = df[df["room"] == "room1"]
tgt = df[df["room"] == "room2"]
tgt0 = sorted(tgt[tgt["label"] == 0]["source"].unique())
tgt1 = sorted(tgt[tgt["label"] == 1]["source"].unique())
print("Source (Room 1):", len(src), "windows /", src["source"].nunique(), "sessions")
print("Target (Room 2):", len(tgt), "windows / empty:", len(tgt0), " occupied:", len(tgt1))

N_TEST = 2                      # held-out target sessions per class
max_pairs = min(len(tgt0), len(tgt1)) - N_TEST
pairs = list(range(0, max_pairs + 1))
print("Calibration pairs evaluated:", pairs, "\n")

def one_run(n_pairs, use_source):
    t0 = list(RNG.permutation(tgt0))
    t1 = list(RNG.permutation(tgt1))
    test_s = t0[:N_TEST] + t1[:N_TEST]
    cal_s = t0[N_TEST:N_TEST + n_pairs] + t1[N_TEST:N_TEST + n_pairs]
    parts = []
    if use_source:
        parts.append(src)
    if cal_s:
        parts.append(tgt[tgt["source"].isin(cal_s)])
    if not parts:
        return None
    train = pd.concat(parts)
    test = tgt[tgt["source"].isin(test_s)]
    if len(np.unique(train["label"])) < 2 or len(np.unique(test["label"])) < 2:
        return None
    m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    m.fit(train[FEATS].values, train["label"].values)
    p = m.predict(test[FEATS].values)
    prob = m.predict_proba(test[FEATS].values)[:, 1]
    return (balanced_accuracy_score(test["label"], p) * 100,
            roc_auc_score(test["label"], prob) * 100)

def sweep(use_source, label):
    ba_m, ba_s, auc_m = [], [], []
    print(label)
    print("%14s %16s %14s" % ("cal. sessions", "Bal.Acc [%]", "AUC [%]"))
    for n in pairs:
        rows = [r for _ in range(REPEATS) if (r := one_run(n, use_source))]
        if not rows:
            ba_m.append(np.nan); ba_s.append(np.nan); auc_m.append(np.nan)
            print("%14d %16s %14s" % (2 * n, "-", "-"))
            continue
        a = np.array(rows)
        ba_m.append(a[:, 0].mean()); ba_s.append(a[:, 0].std()); auc_m.append(a[:, 1].mean())
        print("%14d %10.1f +- %4.1f %13.1f" % (2 * n, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()))
    print()
    return ba_m, ba_s, auc_m

ba1, sd1, auc1 = sweep(True,  "### Source (Room 1) + N calibration sessions from Room 2")
ba2, sd2, auc2 = sweep(False, "### Target-only: trained on the calibration sessions alone")

x = [2 * n for n in pairs]
plt.figure(figsize=(7, 4.8))
plt.errorbar(x, ba1, yerr=sd1, marker="o", capsize=4, label="Source + calibration (bal. acc.)")
plt.plot(x, auc1, marker="s", ls="--", label="Source + calibration (AUC)")
plt.errorbar(x, ba2, yerr=sd2, marker="^", capsize=4, label="Calibration only (bal. acc.)")
plt.axhline(50, ls=":", c="grey", lw=1, label="Chance")
plt.xlabel("Calibration sessions from the target room (1 min each, balanced)")
plt.ylabel("Performance [%]")
plt.title("Few-shot adaptation to an unseen environment")
plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("fewshot_adaptation.png", dpi=200)
print("Saved fewshot_adaptation.png")
plt.show()