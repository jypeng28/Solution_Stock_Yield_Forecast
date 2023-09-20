import optuna
import pandas as pd
from sklearn import linear_model
import pickle
from split import PurgedGroupTimeSeriesSplit
import lightgbm as lgb
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
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    'subsample': trial.suggest_float("subsample", 0.4, 0.95, step=0.1),
    'learning_rate': 0.05,
    "max_depth": trial.suggest_int("max_depth",3,15),
    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.95, step=0.1),
    "min_child_samples": trial.suggest_int("min_child_samples", 100, 1000, step=100),
    "max_bin": trial.suggest_int("max_bin", 100, 500, step=50),
    "bagging_freq": trial.suggest_int("bagging_freq", 1,9),
    "verbosity": -1,
    'random_state':42,
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    "n_estimators": trial.suggest_int('n_estimators', 1000, 5000,step=1000),
    'num_leaves': trial.suggest_int('num_leaves', 15, 599),
    }
    res_vec = np.zeros((nfolds, 1))
    cv = PurgedGroupTimeSeriesSplit(n_splits=5,group_gap=10)
    for (ii, (id0, id1)) in enumerate(cv.split(X, groups = pd.DataFrame(time_train)['time_id'])):

        x0, x1 = X.loc[id0], X.loc[id1]
        y0, y1 = y.loc[id0], y.loc[id1]
        
        df_val_tmp = df_val.iloc[id1]
        
        model = lgb.LGBMRegressor(**param_grid)
        model.fit(x0, y0, eval_metric='rmse', eval_set=[(x1, y1)], early_stopping_rounds=100, verbose=-1)
        val_preds = model.predict(x1)
        # validation score    
        df_val_tmp['pred'] = val_preds
        # validation score    
        score = df_val_tmp.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
        if score is np.nan:
            return np.nan
        res_vec[ii] = score
    return np.mean(res_vec)


def main():
    
    train_df = pd.read_pickle('train_df_v2.pkl')

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
    study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X, y, time_train, df_val)
    study.optimize(func, n_trials=30)
    print(study.best_params)
    print(study.best_value)
    print('Best trial:', study.best_trial.params)

if __name__ == '__main__':
    main()

# tmux 0