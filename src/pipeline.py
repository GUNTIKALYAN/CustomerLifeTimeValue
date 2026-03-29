import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



# Preprocessing
class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Drop ID if exists
        if 'id' in X.columns:
            X = X.drop(columns=['id'])

        return X



# Feature Engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        
        # Numeric conversions
        X['claim_amount'] = pd.to_numeric(X['claim_amount'], errors='coerce').fillna(0)
        X['vintage'] = pd.to_numeric(X['vintage'], errors='coerce').fillna(0)

    
        # Convert income to numeric
        def income_to_num(x):
            if isinstance(x, str):
                if "Below" in x:
                    return 2
                elif "5L-10L" in x:
                    return 7.5
                elif "More than 10L" in x:
                    return 12
            return 0

        X['income_num'] = X['income'].apply(income_to_num)

    
        # Policy count mapping
        def policy_count(x):
            if x == "More than 1":
                return 2
            elif x == "1":
                return 1
            return 1

        X['policy_count'] = X['num_policies'].apply(policy_count)

        # Core strong features
        X['claim_to_income'] = X['claim_amount'] / (X['income_num'] + 1)

        X['avg_policy_age'] = X['vintage'] / (X['policy_count'] + 1)

        # Interaction features

        # Claim over time
        X['claim_per_vintage'] = X['claim_amount'] / (X['vintage'] + 1)

        # Policy impact
        X['policy_claim_interaction'] = X['claim_amount'] * X['policy_count']

        # Income strength
        X['income_vintage_interaction'] = X['income_num'] * X['vintage']

        # High value customer flag
        X['high_value_customer'] = (
            (X['income_num'] > 7) & (X['vintage'] > 5)
        ).astype(int)

        # Claim intensity
        X['claim_intensity'] = X['claim_amount'] / (X['policy_count'] + 1)

        return X



# OneHot Encoding
class OneHotEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None

    def fit(self, X, y=None):
        X = pd.get_dummies(X)
        self.columns = X.columns
        return self

    def transform(self, X):
        X = pd.get_dummies(X)

        # Align columns with training
        X = X.reindex(columns=self.columns, fill_value=0)

        return X
