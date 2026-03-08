import pandas as pd
import pickle

def main():
    print("Loading trained model...")
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Loading training data for feature names...")
    train = pd.read_csv("data/processed/train.csv")

    X = train.drop("Survived", axis=1)

    print("Extracting feature importances...")

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="importance",
        ascending=False
    )

    top_10 = feature_importance_df.head(10)

    print("Saving top 10 features...")
    top_10.to_csv("feature_importance.csv", index=False)

    print("Feature importance saved successfully!")

if __name__ == "__main__":
    main()