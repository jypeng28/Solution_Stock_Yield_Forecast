import os
import pandas as pd
import numpy as np
import gc
import math

def reduce_memory_usage(df, features):
    for feature in features:
        item = df[feature].astype(np.float16)
        df[feature] = item
        del item
        gc.collect()


def reduce_memory_usage(df, features):
    for feature in features:
        item = df[feature].astype(np.float16)
        df[feature] = item
        del item
        gc.collect()

n_features = 300
features = [str(i) for i in range(n_features)]
feature_columns = ['stock_id', 'time_id'] + features
train = pd.read_csv('./train.csv')
reduce_memory_usage(train, features + ["label"])
train.to_pickle("train.pkl")
