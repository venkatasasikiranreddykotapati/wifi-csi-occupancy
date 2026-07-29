"""
CSI ML Evaluation: Random Forest / SVM / kNN on features.csv
- Honest evaluation: GroupKFold by session (source file)
- Also shows naive random-split accuracy to demonstrate leakage inflation
- Outputs: accuracies, confusion matrix plot, feature importance plot
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("features.csv")
X = df.drop(columns=["label", "source"]).values
y = df["label"].values
groups = df["source"].values
feature_names = df.drop(columns=["label", "source"]).columns

print(f"Dataset: {X.shape[0]} windows, {X.shape[1]} features, "
      f"{len(np.unique(groups))} sessions, classes {np.unique(y)}")

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "SVM (RBF)":    make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=42)),
    "kNN (k=5)":    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

# ---------- 1) NAIVE random split (for comparison ONLY - inflated by leakage) ----------
print("\n--- Naive random 70/30 split (LEAKY - reported only as a caution) ---")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
for name, m in models.items():
    m.fit(Xtr, ytr)
    print(f"{name:14s}: {accuracy_score(yte, m.predict(Xte))*100:.1f}%")

# ---------- 2) HONEST session-grouped 5-fold CV ----------
print("\n--- GroupKFold (5 folds, split by session) - THE honest numbers ---")
gkf = GroupKFold(n_splits=5)
best_name, best_acc, best_pred = None, 0, None
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=gkf, groups=groups)
    pred = cross_val_predict(m, X, y, cv=gkf, groups=groups)
    acc = scores.mean()
    print(f"{name:14s}: {acc*100:.1f}%  (folds: {[f'{s*100:.0f}' for s in scores]})")
    if acc > best_acc:
        best_name, best_acc, best_pred = name, acc, pred

# ---------- 3) Confusion matrix of the best model ----------
print(f"\nBest model: {best_name} ({best_acc*100:.1f}%)")
print(classification_report(y, best_pred, digits=3))
cm = confusion_matrix(y, best_pred)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.xlabel("Predicted people count")
plt.ylabel("True people count")
plt.title(f"{best_name} - session-grouped CV")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)

# ---------- 4) Feature importance (Random Forest) ----------
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
plt.figure(figsize=(6, 4.5))
imp.plot.barh()
plt.xlabel("Feature importance")
plt.title("Random Forest feature importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)

# ---------- 5) Binary presence detection (0 vs any people) ----------
print("\n--- Presence detection (0 vs 1+) - session-grouped ---")
yb = (y > 0).astype(int)
scores = cross_val_score(models["RandomForest"], X, yb, cv=gkf, groups=groups)
print(f"RandomForest presence accuracy: {scores.mean()*100:.1f}%")

print("\nSaved: confusion_matrix.png, feature_importance.png")
# ---------- 6) 3-class occupancy level (0 / 1 / 2+) ----------
print("\n--- Occupancy level (0 / 1 / 2+) - session-grouped ---")
y3 = np.minimum(y, 2)
for name, m in models.items():
    scores = cross_val_score(m, X, y3, cv=gkf, groups=groups)
    print(f"{name:14s}: {scores.mean()*100:.1f}%")

# ---------- 7) Count accuracy with +-1 tolerance (best 4-class model) ----------
tol = (np.abs(best_pred - y) <= 1).mean()
print(f"\nCount within +-1 person ({best_name}): {tol*100:.1f}%")
plt.show()