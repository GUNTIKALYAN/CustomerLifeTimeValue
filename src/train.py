import joblib
import numpy as np
import lightgbm as lgb
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.data_loader import load_data
from src.pipeline import Preprocessor, FeatureEngineer, OneHotEncoderCustom, TargetEncoder
from src.model import get_model


def main():
    train, _ = load_data()

    upper_limit = train['cltv'].quantile(0.99)

    train['cltv_clipped'] = train['cltv'].clip(upper=upper_limit)

    y = np.log1p(train['cltv_clipped'])

    X = train.drop(columns=['cltv', 'cltv_clipped'])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocess_pipeline = Pipeline([
        ("preprocess", Preprocessor()),
        ("feature_engineering", FeatureEngineer()),
        ("encoding", OneHotEncoderCustom())
    ])

    X_train_transformed = preprocess_pipeline.fit_transform(X_train)
    X_val_transformed = preprocess_pipeline.transform(X_val)

    model = get_model()

    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_val_transformed, y_val)],
        callbacks=[lgb.early_stopping(100)]
    )

    val_preds = model.predict(X_val_transformed)

    val_preds = np.expm1(val_preds)
    y_val_actual = np.expm1(y_val)

    mae = mean_absolute_error(y_val_actual, val_preds)

    print(f" Validation MAE: {mae:.2f}")

    for col in ['policy', 'type_of_policy', 'qualification']:
        mean_map = train.groupby(col)['cltv'].mean()
        X[col + "_te"] = X[col].map(mean_map)

    pipeline = Pipeline([
        ("preprocess", Preprocessor()),
        ("feature_engineering", FeatureEngineer()),
        ("target_encoding", TargetEncoder()),
        ("encoding", OneHotEncoderCustom()),
        ("model", model)
    ])

    joblib.dump(pipeline, "model.pkl")

    print(" Model trained & saved!")


if __name__ == "__main__":
    main()