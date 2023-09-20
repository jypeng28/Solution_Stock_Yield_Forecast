import optuna
import pandas as pd
from sklearn import linear_model
import pickle
from split import PurgedGroupTimeSeriesSplit
from catboost import CatBoostRegressor
import numpy as np
from scipy.stats import pearsonr as p
import argparse
import logging
import os
from optuna.integration import LightGBMPruningCallback
import warnings
warnings.filterwarnings("ignore")
import pdb
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


def objective(trial, X, y, time_train, df_val):
    nfolds = 5
    param_grid = {

    "iterations": 1500,
    "learning_rate": 0.028,
    # "task_type" : "GPU",
    "depth": trial.suggest_int("depth", 3, 16),
    "verbose" : 1000,
    "eval_metric": "RMSE",
    "objective": "RMSE",
    'l2_leaf_reg': 5.0,
    'subsample': 0.85,
    'rsm': trial.suggest_float("rsm", 0.1, 0.5),
    'early_stopping_rounds': 20,
    'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 100, 1000, step=100),

    }
    res_vec = np.zeros((nfolds, 1))
    cv = PurgedGroupTimeSeriesSplit(n_splits=5,group_gap=10)
    for (ii, (id0, id1)) in enumerate(cv.split(X, groups = pd.DataFrame(time_train)['time_id'])):

        x0, x1 = X.loc[id0], X.loc[id1]
        y0, y1 = y.loc[id0], y.loc[id1]
        
        df_val_tmp = df_val.iloc[id1]
        
        model = CatBoostRegressor(**param_grid)
        model.fit(x0, y0, eval_set=[ (x1, y1)], use_best_model = True)
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

    train_df['time_id'] = train_df['time_id'].astype(np.int32)
    train_df['stock_id'] = train_df['stock_id'].astype(np.int32)
    
    train_df = train_df.reset_index(drop=True)
    time_train= train_df['time_id'].copy()
    df_val = train_df[['stock_id','time_id','label']]
    train_df.drop(['stock_id','time_id'],axis=1,inplace=True)
    
    train_columns = list(train_df.columns)
    train_columns.remove('label')
    X = train_df[train_columns]
    y = train_df['label']
    study = optuna.create_study(direction="maximize", study_name="Catboost Classifier")
    func = lambda trial: objective(trial, X, y, time_train, df_val)
    study.optimize(func, n_trials=30)
    print(study.best_params)
    print(study.best_value)
    print('Best trial:', study.best_trial.params)

if __name__ == '__main__':
    main()