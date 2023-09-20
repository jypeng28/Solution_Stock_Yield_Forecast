import os
import pandas as pd
import numpy as np
import gc
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import logging
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二，三块GPU（从0开始）
import warnings
from split import CombinatorialPurgedGroupKFold, PurgedGroupTimeSeriesSplit
import pickle as pkl
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract
from keras import backend as K
from keras import regularizers 
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from sklearn.base import clone
from typing import Dict
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold, GroupKFold
from tqdm import tqdm
from tensorflow.python.ops import math_ops
from sklearn.preprocessing import scale
import pdb
def get_model():
    week_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    
    week_x = layers.Embedding(7, 32, input_length=1)(week_inputs)
    week_x = layers.Reshape((-1, ))(week_x)
    x = layers.Concatenate(axis=1)([week_x, features_inputs])
    x = layers.Dense(512, activation='swish')(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.Dense(32, activation='swish')(x)
    output = layers.Dense(1)(x)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[week_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
    return model


warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)



class Logger:
    logger = None
    @staticmethod
    def get_logger(filename: str = None):
        if not Logger.logger:
            Logger.init_logger(filename=filename)
        return Logger.logger
    @staticmethod
    def init_logger(level=logging.INFO,
                    fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filename: str = None):
        logger = logging.getLogger(filename)
        logger.setLevel(level)

        fmt = logging.Formatter(fmt)

        # stream handler
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if os.path.exists(filename):
            os.remove(filename)
        if filename:
            # file handler
            fh = logging.FileHandler(filename)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.setLevel(level)
        Logger.logger = logger
        return logger



def scale_by_time_id(df, features):
    
    def lambda_scale(d):
        d = scale(d)
        return d
    
    data = df.groupby("time_id").apply(lambda x: lambda_scale(x[features]))
    data = np.concatenate(data.values)
    
    df[features] = data
    
    return df

def preprocess(X, y):
    return X, y

def make_dataset(week, feature, y, mode="train"):
    ds = tf.data.Dataset.from_tensor_slices(((week, feature), y))
    if mode == "train":
        ds = ds.shuffle(4096)
    ds = ds.batch(1024).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds
    
def main():
    nfolds = 5
    parser = argparse.ArgumentParser(description='Validate Performance with Lightgbm baseline')
    parser.add_argument('--data_path', type=str, default='train_dropna')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    log_path = './logs/' + args.data_path + str(args.seed) +  '.log'

    data_path = 'train_dropna.pkl'
    
    logger = Logger.init_logger(filename=log_path)
    
    train_df = pd.read_pickle(data_path)
#     device = torch.device("cuda:2")
#     set_seed(args.seed)
#     with open('feature_corr_list.pkl', 'rb') as f:
#         best_feats = pickle.load(f)

#     # feat_top_100_stock_before
#     feat_top_100_stock_before_pd = pd.read_pickle('top100_feat_stock_before.pkl')
#     train_df = pd.concat([train_df,feat_top_100_stock_before_pd],axis=1)
    
#     # 

# # Week

    train_df['week'] = (train_df['time_id']%7).astype('category')
# # # # 
    week = True
#     float_columns = list(train_df.columns)

#     float_columns.remove('label')
#     float_columns.remove('time_id')
#     float_columns.remove('stock_id')
#     float_columns.remove('week')
 
#     train_df = scale_by_time_id(train_df, float_columns)
# # # Lagging 1
#     for fe in tqdm(best_feats[:100]):
#         train_df[f"lagging_1_{fe}_rate"] = train_df.groupby("stock_id")[fe].diff(1)
#         train_df[f"lagging_1_{fe}_rate"] = train_df[f"lagging_1_{fe}_rate"].fillna(0).astype(np.float16)


#     mapper = train_df.groupby(['time_id'])['0'].count().to_dict()
#     train_df['new'] =train_df['time_id'].map(mapper)
    train_df = train_df.reset_index(drop=True)

    time_train= train_df['time_id'].copy()
    df_val = train_df[['stock_id','time_id','label']]
    train_df.drop(['stock_id','time_id'],axis=1,inplace=True)
    
    train_columns = list(train_df.columns)
    train_columns.remove('label')

    
    X = train_df[train_columns]
    Y = train_df['label']
    cv = PurgedGroupTimeSeriesSplit(n_splits=5,group_gap=10)
    n_continue_feature = len(train_columns)
    if week:
        n_continue_feature -= 1
    scores = np.zeros((5, 1))
    models = []
    for (ii, (id0, id1)) in enumerate(cv.split(X, groups = pd.DataFrame(time_train)['time_id'])):

        x0, x1 = X.loc[id0], X.loc[id1]
        y0, y1 = Y.loc[id0], Y.loc[id1]

        float_columns = list(x0.columns)
        float_columns.remove('week')
        
        df_val_tmp = df_val.iloc[id1]

        train_ds = make_dataset(x0['week'], x0[float_columns], y0)
        valid_ds = make_dataset(x1['week'], x1[float_columns], y1, mode="valid")
        model = get_model()
        checkpoint = keras.callbacks.ModelCheckpoint(f"model_{ii}", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(patience=10)
        history = model.fit(train_ds, epochs=10, validation_data=valid_ds, callbacks=[checkpoint, early_stop])
        model = keras.models.load_model(f"model_{ii}",custom_objects={"RootMeanSquaredError": keras.metrics.RootMeanSquaredError(name="rmse") })
        models.append(model)    
        pred_val = model.predict(valid_ds).ravel()
        pdb.set_trace()
        df_val_tmp['pred'] = pred_val
        score = df_val_tmp.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()    
        print(score)

        

    logger.info(f"average_score:{np.mean(scores)}")
if __name__ == '__main__':
    main()