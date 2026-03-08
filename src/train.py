import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading training data...")

    train_data = pd.read_csv("data/processed/train.csv")

    X = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]

    print("Loading hyperparameters...")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    print("Training Random Forest model...")

    model = RandomForestClassifier(**params)
    model.fit(X, y)

    print("Saving model...")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved successfully!")

if __name__ == "__main__":
    main()