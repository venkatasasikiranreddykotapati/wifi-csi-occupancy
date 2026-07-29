"""
Hold-out evaluation v2: train on day-1 blocks, test on night block.
Runs twice: ALL features vs DYNAMIC-ONLY (static channel-profile features removed).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

DAY2_TAG = "20260720"
STATIC_FEATURES = ["spec_std", "spec_range"]   # static channel-profile features

df = pd.read_csv("features.csv")
is_day2 = df["source"].str.contains(DAY2_TAG)
train, test = df[~is_day2], df[is_day2]
print(f"Train: {len(train)} windows / Test: {len(test)} windows (night block)")

for mode in ["ALL features", "DYNAMIC-only"]:
    drop = ["label", "source"] + (STATIC_FEATURES if mode == "DYNAMIC-only" else [])
    Xtr = train.drop(columns=drop).values; ytr = train["label"].values
    Xte = test.drop(columns=drop).values;  yte = test["label"].values
    print(f"\n################ {mode} ({Xtr.shape[1]} features) ################")
    for name, m in {
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "SVM (RBF)":    make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=42)),
    }.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        print(f"=== {name} ===  presence: {accuracy_score(yte > 0, p > 0)*100:.1f}%   "
              f"exact: {accuracy_score(yte, p)*100:.1f}%   +-1: {(np.abs(p-yte)<=1).mean()*100:.1f}%")
        print(confusion_matrix(yte, p))