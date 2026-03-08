import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')

def main():
    print("Loading test data...")
    test = pd.read_csv("data/processed/test.csv")

    X_test = test.drop("Survived", axis=1)
    y_test = test["Survived"]

    print("Loading trained model...")
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Making predictions...")
    y_pred = model.predict(X_test)

    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()