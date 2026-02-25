from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

# Create directory if not exists
os.makedirs("data/raw", exist_ok=True)

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

df.to_csv("data/raw/data.csv", index=False)

print("Dataset saved successfully!")