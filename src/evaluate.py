import pandas as pd
import pickle
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def main():
    print("Loading test data...")
    test = pd.read_csv("data/processed/test.csv")

    X_test = test.drop("target", axis=1)
    y_test = test["target"]

    print("Loading trained model...")
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Calculating metrics...")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("Saving metrics to metrics.json...")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()