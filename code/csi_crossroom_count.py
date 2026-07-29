"""
Cross-room evaluation for multi-class occupancy (0/1/2/3 and 0/1/2+).
Uses balanced folds containing every class.
"""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

df = pd.read_csv("features.csv")

def get_room(s):
    m = re.search(r"csi_(room\d+)_", str(s).replace("\\", "/"))
    return m.group(1) if m else "room1"

df["room"] = df["source"].apply(get_room)
FEATS = [c for c in df.columns if c not in ("label", "source", "room")]
rooms = sorted(df["room"].unique())

print("Windows per room / class:")
print(df.groupby(["room", "label"]).size().unstack(fill_value=0))
print()

def make_model():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)

def balanced_folds(sub, y, k=5):
    """Folds built so that every class appears in each test fold."""
    idx = np.arange(len(sub))
    src = sub["source"].values
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        sessions = sorted(sub[y == cls]["source"].unique())
        for i, s in enumerate(sessions):
            folds[i % k].append(s)
    out = []
    for f in folds:
        mask = np.isin(src, f)
        if mask.sum() > 0 and (~mask).sum() > 0:
            out.append((idx[~mask], idx[mask]))
    return out if len(out) >= 2 else None

def within(sub, ycol, title):
    y = sub[ycol].values
    folds = balanced_folds(sub, y)
    if folds is None:
        print("###", title, "- not enough data\n"); return
    X = sub[FEATS].values
    ba = cross_val_score(make_model(), X, y, cv=folds, scoring="balanced_accuracy")
    f1 = cross_val_score(make_model(), X, y, cv=folds, scoring="f1_macro")
    print("###", title)
    print("    balanced accuracy %5.1f%% +- %4.1f    macro-F1 %5.1f%%"
          % (ba.mean()*100, ba.std()*100, f1.mean()*100))
    print()

def transfer(tr, te, ycol, title, tol=False):
    ytr, yte = tr[ycol].values, te[ycol].values
    m = make_model().fit(tr[FEATS].values, ytr)
    p = m.predict(te[FEATS].values)
    ba = balanced_accuracy_score(yte, p) * 100
    f1 = f1_score(yte, p, average="macro") * 100
    print("###", title)
    line = "    balanced accuracy %5.1f%%    macro-F1 %5.1f%%" % (ba, f1)
    if tol:
        line += "    within +-1: %5.1f%%" % ((np.abs(p - yte) <= 1).mean() * 100)
    print(line)
    print("    confusion matrix:")
    print(confusion_matrix(yte, p))
    print()

df["level"] = np.minimum(df["label"], 2)

for ycol, name, tol in [("level", "3-CLASS OCCUPANCY LEVEL (0/1/2+)", False),
                        ("label", "4-CLASS EXACT COUNT (0/1/2/3)", True)]:
    print("=" * 72); print(name); print("=" * 72)
    for r in rooms:
        within(df[df["room"] == r], ycol, "Within " + r)
    for rtr in rooms:
        for rte in rooms:
            if rtr != rte:
                transfer(df[df["room"] == rtr], df[df["room"] == rte], ycol,
                         "Train " + rtr + " -> Test " + rte, tol)
    within(df, ycol, "All rooms combined")