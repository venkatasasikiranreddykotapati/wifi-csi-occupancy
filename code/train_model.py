"""Train the final model on ALL features and save it for the Pi."""
import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("features.csv")
X = df.drop(columns=["label", "source"]).values
y = df["label"].values

model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
model.fit(X, y)
joblib.dump(
    {"model": model, "features": list(df.drop(columns=["label","source"]).columns)},
    "model.pkl"
)
print(f"Saved model.pkl trained on {len(X)} windows, classes {sorted(set(y))}")