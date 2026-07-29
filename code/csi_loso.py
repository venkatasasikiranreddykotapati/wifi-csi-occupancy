"""
Leave-one-subject-out: does presence detection work for people never seen in training?
Reads features.csv (main campaign, for empty-room data) and
features_persons.csv (person-independence study, all label 1).
"""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score

RNG = np.random.default_rng(42)

main = pd.read_csv("features.csv")
per = pd.read_csv("features_persons.csv")
main["src"] = main["source"].str.replace("\\", "/", regex=False)
per["src"] = per["source"].str.replace("\\", "/", regex=False)

def person_of(s):
    m = re.search(r"csi_room1p([A-Za-z])_", s)
    return m.group(1).upper() if m else None

per["person"] = per["src"].apply(person_of)
FEATS = [c for c in main.columns if c not in ("label", "source", "src")]

# empty-room windows from the bedroom only (exclude room2/room3)
empty = main[(main["label"] == 0) & (~main["src"].str.contains("room2|room3"))].copy()
persons = sorted(per["person"].dropna().unique())

print("Subjects:", persons)
print("Bedroom empty windows:", len(empty))
print("Person windows:", {p: int((per["person"] == p).sum()) for p in persons}, "\n")

# fixed 50/50 split of empty sessions
sess = sorted(empty["source"].unique())
RNG.shuffle(sess)
half = len(sess) // 2
empty_tr = empty[empty["source"].isin(sess[:half])]
empty_te = empty[empty["source"].isin(sess[half:])]
print(f"Empty sessions: {len(sess)} -> {half} train / {len(sess)-half} test\n")

def model():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)

print("%-18s %12s %9s %11s" % ("Held-out subject", "Bal.Acc [%]", "AUC [%]", "Recall [%]"))
rows = []
for p in persons:
    tr = pd.concat([empty_tr[FEATS + ["label"]], per[per["person"] != p][FEATS + ["label"]]])
    te = pd.concat([empty_te[FEATS + ["label"]], per[per["person"] == p][FEATS + ["label"]]])
    m = model().fit(tr[FEATS].values, (tr["label"] > 0).astype(int).values)
    y = (te["label"] > 0).astype(int).values
    pred = m.predict(te[FEATS].values)
    prob = m.predict_proba(te[FEATS].values)[:, 1]
    ba = balanced_accuracy_score(y, pred) * 100
    auc = roc_auc_score(y, prob) * 100
    rec = recall_score(y, pred) * 100
    rows.append((ba, auc, rec))
    print("%-18s %11.1f %8.1f %10.1f" % (p, ba, auc, rec))

a = np.array(rows)
print("\n%-18s %11.1f %8.1f %10.1f" % ("MEAN", a[:,0].mean(), a[:,1].mean(), a[:,2].mean()))
print("%-18s %11.1f %8.1f %10.1f" % ("STD", a[:,0].std(), a[:,1].std(), a[:,2].std()))

print("\n--- Reference: subject INCLUDED in training ---")
tr_all = pd.concat([empty_tr[FEATS + ["label"]], per[FEATS + ["label"]]])
m = model().fit(tr_all[FEATS].values, (tr_all["label"] > 0).astype(int).values)
ref = []
for p in persons:
    te = pd.concat([empty_te[FEATS + ["label"]], per[per["person"] == p][FEATS + ["label"]]])
    y = (te["label"] > 0).astype(int).values
    ba = balanced_accuracy_score(y, m.predict(te[FEATS].values)) * 100
    ref.append(ba)
    print("%-18s balanced accuracy %5.1f%%" % (p, ba))
print("\nPerson-independence gap: %.1f pp" % (np.mean(ref) - a[:,0].mean()))