import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.data_loader import load_data
from src.pipeline import Preprocessor, FeatureEngineer, OneHotEncoderCustom
from src.model import get_model


def main():

    # Load data
    train, _ = load_data()

    
    # Handle outliers (clipping)
    upper_limit = train['cltv'].quantile(0.99)
    train['cltv_clipped'] = train['cltv'].clip(upper=upper_limit)

    # Target (log transform)
    y = np.log1p(train['cltv_clipped'])

    # Features
    X = train.drop(columns=['cltv', 'cltv_clipped'])


    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    # FULL PIPELINE (IMPORTANT)
    pipeline = Pipeline([
        ("preprocess", Preprocessor()),
        ("feature_engineering", FeatureEngineer()),
        ("encoding", OneHotEncoderCustom()),
        ("model", get_model())
    ])

    
    # Train pipeline (CRITICAL FIX)
    pipeline.fit(
        X_train,
        y_train,
        model__eval_set=[(
            pipeline[:-1].transform(X_val), y_val
        )],
        model__callbacks=[lgb.early_stopping(100)]
    )

    # Validation
    val_preds = pipeline.predict(X_val)

    val_preds = np.expm1(val_preds)
    y_val_actual = np.expm1(y_val)

    mae = mean_absolute_error(y_val_actual, val_preds)

    print(f"Validation MAE: {mae:.2f}")


    # Save pipeline
    joblib.dump(pipeline, "model.pkl")

    print("✅ Model trained & saved!")


if __name__ == "__main__":
    main()