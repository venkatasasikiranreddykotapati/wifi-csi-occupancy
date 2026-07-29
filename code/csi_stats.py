"""
Statistical rigor: mean +- std, 95% confidence intervals, significance tests,
baseline comparison, and a LaTeX-ready results table.
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

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "kNN (k=5)":    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM (RBF)":    make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=42)),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
}

tasks = {
    "Presence (0 vs 1+)": (y > 0).astype(int),
    "Level (0/1/2+)":     np.minimum(y, 2),
    "Exact count (0-3)":  y,
}

def ci95(scores):
    """95% confidence interval of the mean using the t-distribution."""
    n = len(scores)
    m, se = scores.mean(), stats.sem(scores)
    h = se * stats.t.ppf(0.975, n - 1)
    return m - h, m + h

all_scores = {}
for tname, yt in tasks.items():
    print(f"\n{'='*72}\n{tname}\n{'='*72}")
    print(f"{'Model':22s} {'Mean':>7s} {'Std':>7s} {'95% CI':>18s}   folds")
    all_scores[tname] = {}
    for mname, m in models.items():
        s = cross_val_score(m, X, yt, cv=gkf, groups=g)
        all_scores[tname][mname] = s
        lo, hi = ci95(s)
        folds = " ".join(f"{v*100:.0f}" for v in s)
        print(f"{mname:22s} {s.mean()*100:6.1f}% {s.std()*100:6.1f}% "
              f"[{lo*100:5.1f}, {hi*100:5.1f}]   {folds}")

    # significance vs baseline (paired t-test across folds)
    base = all_scores[tname]["Majority baseline"]
    print("\n  Significance vs majority baseline (paired t-test over folds):")
    for mname in ["kNN (k=5)", "SVM (RBF)", "RandomForest"]:
        t, p = stats.ttest_rel(all_scores[tname][mname], base)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {mname:22s} t={t:6.2f}  p={p:.4f}  {star}")

    # best two models against each other
    best = sorted(["kNN (k=5)", "SVM (RBF)", "RandomForest"],
                  key=lambda k: all_scores[tname][k].mean(), reverse=True)[:2]
    t, p = stats.ttest_rel(all_scores[tname][best[0]], all_scores[tname][best[1]])
    verdict = "significant" if p < 0.05 else "NOT significant"
    print(f"\n  {best[0]} vs {best[1]}: p={p:.4f} -> difference is {verdict}")

# ---------- LaTeX-ready table ----------
print(f"\n{'='*72}\nLaTeX table (copy into your thesis)\n{'='*72}")
print(r"\begin{tabular}{lccc}")
print(r"\hline")
print(r"Model & " + " & ".join(tasks.keys()) + r" \\")
print(r"\hline")
for mname in models:
    cells = []
    for tname in tasks:
        s = all_scores[tname][mname]
        cells.append(f"{s.mean()*100:.1f} $\\pm$ {s.std()*100:.1f}")
    print(f"{mname} & " + " & ".join(cells) + r" \\")
print(r"\hline")
print(r"\end{tabular}")
print("\nAll values: accuracy [%] mean +- std over 5 session-grouped CV folds.")