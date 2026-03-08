import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("Loading processed dataset...")

    df = pd.read_csv("data/processed/data.csv")

    print("Splitting into train and test...")

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Survived"]
    )

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Train and test datasets saved successfully!")

if __name__ == "__main__":
    main()