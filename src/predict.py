import joblib
import pandas as pd
import os

from src.data_loader import load_data


def main():
    _, test = load_data()

    model_path = os.path.join(os.path.dirname(__file__), "..", "lightgbm_model.pkl")
    pipeline = joblib.load(model_path)

    test_ids = test['id']

    preds = pipeline.predict(test)

    submission = pd.DataFrame({
        "id": test_ids,
        "cltv": preds
    })

    submission.to_csv("submission.csv", index=False)

    print("Submission created!")


if __name__ == "__main__":
    main()