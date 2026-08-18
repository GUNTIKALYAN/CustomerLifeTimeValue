import joblib
import numpy as np
import pandas as pd
import os

from src.data_loader import load_data


def main():
    _, test = load_data()

    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    pipeline = joblib.load(model_path)

    preds = pipeline.predict(test)
    preds = np.expm1(preds)

    df = test.copy()
    df["predicted_cltv"] = preds.round(2)

    df["cltv_segment"] = pd.qcut(
        df["predicted_cltv"],
        q=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["Low", "Medium", "High", "Premium"],
    )

    df["income_band"] = df["income"]
    df["policy_band"] = df["num_policies"]
    df["has_claims"] = (df["claim_amount"] > 0).astype(int)

    out_path = os.path.join(os.path.dirname(__file__), "..", "powerbi_cltv_data.csv")
    df.to_csv(out_path, index=False)

    print(f"Power BI dataset saved: {out_path}")
    print(df["cltv_segment"].value_counts())


if __name__ == "__main__":
    main()
