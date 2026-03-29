import lightgbm as lgb

def get_model():
    return lgb.LGBMRegressor(
        objective='huber',
        learning_rate=0.02,
        num_leaves=128,          
        min_child_samples=20, 
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_estimators=3000,
        random_state=18,
        n_jobs=-1
    )