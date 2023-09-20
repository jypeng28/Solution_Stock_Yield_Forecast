import pandas as pd
from sklearn import linear_model, preprocessing
import pickle
from split import CombinatorialPurgedGroupKFold, PurgedGroupTimeSeriesSplit
import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr as p
import argparse
import logging
import os
import pdb
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.preprocessing import scale
import warnings 
import optuna
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings('ignore')
from pytorch_tabnet.tab_model import TabNetRegressor
import random
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


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def RMSELoss(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred)) ** 2))


def objective(trial, X, y, time_train, df_val):
    nfolds = 5
    param_grid = {
    "n_d": trial.suggest_categorical("n_d",[32,64]) ,
    "n_a": trial.suggest_categorical("n_a",[32,64]) ,
    "n_steps": trial.suggest_int("n_steps", 5,8),
    "gamma": 1.0,
    "n_independent": trial.suggest_int("n_independent", 5, 10),
    "n_shared": trial.suggest_int("n_shared", 5, 10),
    "lambda_sparse": 0,
    "optimizer_fn": Adam,
    "optimizer_params": {
        "lr":trial.suggest_categorical("lr",[0.01, 0.02]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-5),
    },
    "mask_type": "entmax",
    "scheduler_params": dict(T_0=200, T_mult=1, eta_min=1e-6, last_epoch=-1, verbose=False),
    "scheduler_fn": CosineAnnealingWarmRestarts,
    "seed": 43,
    "verbose": 2,
    
    }

    param_grid['n_a'] = param_grid['n_d']

    res_vec = np.zeros((nfolds, 1))
    cv = CombinatorialPurgedGroupKFold(n_splits = 5, n_test_splits = 1)
    for (ii, (id0, id1)) in enumerate(cv.split(X, groups = pd.DataFrame(time_train)['time_id'])):

        x0, x1 = X.loc[id0].values, X.loc[id1].values
        y0, y1 = y.loc[id0].values.reshape(-1,1), y.loc[id1].values.reshape(-1,1)
        
        df_val_tmp = df_val.iloc[id1]
        
        model = TabNetRegressor(**param_grid)
        print(param_grid)
        model.fit(
            x0, y0, eval_metric=['rmse'], eval_set=[ (x1, y1)],
            max_epochs = 355,
            patience = 10,
            batch_size = 1024*10, 
            virtual_batch_size = 128*10,
            num_workers = 4,
            drop_last = False,
            loss_fn = RMSELoss,
        )

        val_preds = model.predict(x1)
        # validation score    
        df_val_tmp['pred'] = val_preds
        # validation score    
        score = df_val_tmp.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
        print("validation score: " + str(score))
        print("test score: " + str(score))
        if score is np.nan:
            return np.nan
        res_vec[ii] = score
    return np.mean(res_vec)


def main():

    train_df = pd.read_pickle('train_df_1.pkl')
    train_df.reset_index(drop=True, inplace=True)

    train_df = train_df.fillna(0)
    train_columns = list(train_df.columns)
    train_columns.remove('label')
    train_columns.remove('time_id')
    train_columns.remove('stock_id')
    
    time_train= train_df['time_id'].copy()
    df_val = train_df[['stock_id','time_id','label']]

    X = train_df[train_columns]
    y = train_df['label']
    
    study = optuna.create_study(direction="maximize", study_name="Tabnet Tuning")
    func = lambda trial: objective(trial, X, y, time_train, df_val)
    study.optimize(func, n_trials=20)
    print(study.best_params)
    print(study.best_value)
    print('Best trial:', study.best_trial.params)

if __name__ == '__main__':
    main()