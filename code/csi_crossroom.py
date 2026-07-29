"""
Cross-environment evaluation with deterministic balanced folds.
Each CV fold contains both classes, so ROC-AUC is always defined.
Usage:
    python csi_crossroom.py
    python csi_crossroom.py --fair
"""
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--fair", action="store_true")
args = ap.parse_args()

df = pd.read_csv("features.csv")

def get_room(source):
    m = re.search(r"csi_(room\d+)_", str(source).replace("\\", "/"))
    return m.group(1) if m else "room1"

df["room"] = df["source"].apply(get_room)
df = df[df["label"].isin([0, 1])].copy()

if args.fair:
    keep = (df["room"] != "room1") | df["source"].str.contains("_20260719_13")
    df = df[keep].copy()
    print(">>> FAIR MODE: Room 1 restricted to the 19.07 afternoon block <<<")

FEATS = [c for c in df.columns if c not in ("label", "source", "room")]
rooms = sorted(df["room"].unique())

print("Windows per room / class:")
print(df.groupby(["room", "label"]).size())
print("Sessions per room:", {r: df[df["room"] == r]["source"].nunique() for r in rooms})
print()

def make_model():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)

def paired_folds(sub, max_folds=5):
    """Folds that always contain at least one session of each class."""
    s0 = sorted(sub[sub["label"] == 0]["source"].unique())
    s1 = sorted(sub[sub["label"] == 1]["source"].unique())
    k = min(len(s0), len(s1), max_folds)
    if k < 2:
        return None
    src = sub["source"].values
    idx = np.arange(len(sub))
    folds = []
    for i in range(k):
        test_sessions = set(s0[i::k]) | set(s1[i::k])
        mask = np.isin(src, list(test_sessions))
        te, tr = idx[mask], idx[~mask]
        if len(te) > 0 and len(tr) > 0:
            folds.append((tr, te))
    return folds

def eval_within(sub, title):
    folds = paired_folds(sub)
    if folds is None:
        print("###", title, "- not enough sessions")
        print()
        return np.nan, np.nan
    X = sub[FEATS].values
    y = sub["label"].values
    ba = cross_val_score(make_model(), X, y, cv=folds, scoring="balanced_accuracy")
    auc = cross_val_score(make_model(), X, y, cv=folds, scoring="roc_auc")
    print("###", title)
    print("    sessions:", sub["source"].nunique(), " windows:", len(X), " folds:", len(folds))
    print("    balanced accuracy %5.1f%% +- %4.1f    ROC-AUC %5.1f%% +- %4.1f"
          % (ba.mean()*100, ba.std()*100, auc.mean()*100, auc.std()*100))
    print()
    return ba.mean()*100, auc.mean()*100

def eval_transfer(train_df, test_df, title):
    Xtr, ytr = train_df[FEATS].values, train_df["label"].values
    Xte, yte = test_df[FEATS].values, test_df["label"].values
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        print("###", title, "- not enough data")
        print()
        return np.nan, np.nan
    m = make_model().fit(Xtr, ytr)
    p = m.predict(Xte)
    prob = m.predict_proba(Xte)[:, 1]
    ba = balanced_accuracy_score(yte, p) * 100
    auc = roc_auc_score(yte, prob) * 100
    print("###", title)
    print("    train windows:", len(Xtr), " test windows:", len(Xte))
    print("    balanced accuracy %5.1f%%    ROC-AUC %5.1f%%" % (ba, auc))
    print("    confusion matrix (rows = true 0,1):")
    print(confusion_matrix(yte, p))
    print()
    return ba, auc

summary = []

print("=" * 72)
print("WITHIN-ROOM (reference performance)")
print("=" * 72)
for r in rooms:
    summary.append(("Within " + r, eval_within(df[df["room"] == r], r + " only")))

print("=" * 72)
print("CROSS-ROOM TRANSFER (unseen environment)")
print("=" * 72)
for rtr in rooms:
    for rte in rooms:
        if rtr != rte:
            t = "Train " + rtr + "  ->  Test " + rte
            summary.append((t, eval_transfer(df[df["room"] == rtr], df[df["room"] == rte], t)))

if len(rooms) > 1:
    print("=" * 72)
    print("MULTI-ROOM TRAINING")
    print("=" * 72)
    summary.append(("Combined training (all rooms)", eval_within(df, "All rooms combined")))

if len(rooms) >= 3:
    print("=" * 72)
    print("LEAVE-ONE-ROOM-OUT")
    print("=" * 72)
    for r in rooms:
        t = "Train all except " + r + "  ->  Test " + r
        summary.append(("LORO: test " + r, eval_transfer(df[df["room"] != r], df[df["room"] == r], t)))

print("=" * 72)
print("SUMMARY TABLE")
print("=" * 72)
print("%-38s %12s %10s" % ("Setting", "Bal.Acc [%]", "AUC [%]"))
for name, res in summary:
    if not np.isnan(res[0]):
        print("%-38s %11.1f %10.1f" % (name, res[0], res[1]))