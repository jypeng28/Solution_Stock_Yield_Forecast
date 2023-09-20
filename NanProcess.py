import pandas as pd
from sklearn import linear_model
import pickle
from split import PurgedGroupTimeSeriesSplit
import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr as p
from tqdm import tqdm


train_df = pd.read_pickle('./train.pkl')
train_df_drop = train_df.dropna(axis=0, how='any')
train_df_drop.to_pickle("train_dropna.pkl")