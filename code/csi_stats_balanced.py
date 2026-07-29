"""
Imbalance-robust evaluation: balanced accuracy, macro-F1, ROC-AUC.
Corrects the misleading raw-accuracy presence result.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score

df = pd.read_csv("features.csv")
X = df.drop(columns=["label", "source"]).values
y = df["label"].values
g = df["source"].values
gkf = GroupKFold(n_splits=5)

print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Empty windows: {(y==0).sum()},  occupied windows: {(y>0).sum()}\n")

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "kNN (k=5)":    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM (RBF)":    make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced",
                                                        probability=True, random_state=42)),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
}

def report(yt, title, metrics):
    print("="*78); print(title); print("="*78)
    store = {}
    header = f"{'Model':22s}" + "".join(f"{m:>22s}" for m in metrics)
    print(header)
    for mname, m in models.items():
        row, store[mname] = f"{mname:22s}", {}
        for met in metrics:
            s = cross_val_score(m, X, yt, cv=gkf, groups=g, scoring=met)
            store[mname][met] = s
            row += f"{s.mean()*100:12.1f}% ± {s.std()*100:4.1f} "
        print(row)
    base = store["Majority baseline"][metrics[0]]
    print(f"\n  Significance vs baseline on {metrics[0]} (paired t-test):")
    for mname in list(models)[1:]:
        t, p = stats.ttest_rel(store[mname][metrics[0]], base)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {mname:22s} p={p:.4f}  {star}")
    print()

# Presence: binary -> balanced accuracy, F1, AUC
report((y > 0).astype(int), "PRESENCE (0 vs 1+)  -  imbalance-robust metrics",
       ["balanced_accuracy", "f1", "roc_auc"])

# Level and exact: multiclass -> balanced accuracy + macro F1
report(np.minimum(y, 2), "OCCUPANCY LEVEL (0 / 1 / 2+)",
       ["balanced_accuracy", "f1_macro"])

report(y, "EXACT COUNT (0-3)",
       ["balanced_accuracy", "f1_macro"])