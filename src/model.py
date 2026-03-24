import lightgbm as lgb

def get_model():
    return lgb.LGBMRegressor(
        objective='regression',
        learning_rate=0.02,
        num_leaves=64,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.2,
        n_estimators=3000,
        random_state=18
    )