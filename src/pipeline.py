import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# ---------------------------
# Custom Preprocessing
# ---------------------------
class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Drop id if exists
        if 'id' in X.columns:
            X = X.drop(columns=['id'])

        # Convert numeric columns
        num_cols = ['gender', 'area', 'num_policies']
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        return X


# ---------------------------
# Feature Engineering
# ---------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        income_map = {1: 1.5, 2: 3.5, 3: 7.5, 4: 12}
        X['income'] = X['income'].map(income_map).fillna(0)

        X['zero_claim'] = (X['claim_amount'] == 0).astype(int)
        X['customer_intensity'] = X['num_policies'] * X['vintage'].astype(float)
        X['income_claim_gap'] = X['income'] - X['claim_amount']

        return X


# ---------------------------
# Encoder (PERSISTED)
# ---------------------------
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.cat_cols = ['qualification', 'policy', 'type_of_policy', 'marital_status']

    def fit(self, X, y=None):
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.cat_cols:
            if col not in X.columns:
                X[col] = "Unknown"
            le = self.encoders[col]
            X[col] = X[col].astype(str)

            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            X[col] = X[col].map(mapping).fillna(-1)

        return X