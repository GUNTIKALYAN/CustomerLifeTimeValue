import pandas as pd
import os

def load_data():
    base_path = os.path.dirname(os.path.dirname(__file__))
    
    train_path = os.path.join(base_path, "RawData", "train.csv")
    test_path = os.path.join(base_path, "RawData", "test.csv")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    return train, test