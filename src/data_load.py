import pandas as pd
import os

def main():
    print("Loading raw dataset...")

    # Load raw data
    df = pd.read_csv("data/raw/train.csv")

    # Drop text columns that ML cannot use
    df = df.drop(columns=["Name", "Ticket", "Cabin"])

    # Convert categorical columns to numbers
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].fillna("S")
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    print("Dataset shape before cleaning:", df.shape)

    # Basic validation: check for missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Dropping...")
        df = df.dropna()
    else:
        print("No missing values found.")

    print("Dataset shape after cleaning:", df.shape)

    # Create processed folder if not exists
    os.makedirs("data/processed", exist_ok=True)

    # Save cleaned data
    df.to_csv("data/processed/data.csv", index=False)

    print("Cleaned dataset saved successfully!")

if __name__ == "__main__":
    main()