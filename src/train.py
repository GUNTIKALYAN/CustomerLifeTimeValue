import joblib
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.pipeline import Preprocessor, FeatureEngineer, Encoder
from src.model import get_model


def main():
    train, test = load_data()

    y = train['cltv']
    X = train.drop(columns=['cltv'])

    pipeline = Pipeline([
        ("preprocess", Preprocessor()),
        ("feature_engineering", FeatureEngineer()),
        ("encoding", Encoder()),
        ("model", get_model())
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, "model.pkl")

    print("✅ Full pipeline saved!")


if __name__ == "__main__":
    main()