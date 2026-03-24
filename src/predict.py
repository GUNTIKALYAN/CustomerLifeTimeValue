import joblib
import pandas as pd

from src.data_loader import load_data

def main():
    
    train, test = load_data()

    pipeline = joblib.load("model.pkl")

    test_ids = test['id']

    preds = pipeline.predict(test)

    submission = pd.DataFrame({
        "id": test_ids,
        "cltv": preds
    })

    submission.to_csv("submission.csv", index=False, float_format="%.2f")

    print("✅ Submission file created!")


if __name__ == "__main__":
    main()