import pandas as pd
from sklearn import linear_model, preprocessing
import pickle
from split import CombinatorialPurgedGroupKFold, PurgedGroupTimeSeriesSplit
import numpy as np
from scipy.stats import pearsonr as p
import argparse
import logging
import os
import pdb
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from catboost import CatBoostRegressor
import warnings 
import random
warnings.filterwarnings('ignore')
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



def main():
    nfolds = 5
    parser = argparse.ArgumentParser(description='Validate Performance with Lightgbm baseline')
    parser.add_argument('--seed', type=int, default='42')
    args = parser.parse_args()

    log_path = './logs/cat_train_new_param' + str(args.seed) + '.log'

    data_path = 'train_df_1.pkl'
    seed_everything(args.seed)
    logger = Logger.init_logger(filename=log_path)
    

    train_df = pd.read_pickle(data_path)
    train_df.reset_index(drop=True, inplace=True)

    ###
    train_df_test = pd.read_pickle('./test_df.pkl')
    label_test = pd.read_csv('test_label.csv')
    train_df_test['label'] = label_test['label']
    train_df = pd.concat([train_df, train_df_test], axis=0)
    train_df.reset_index(drop=True, inplace=True)
    ###



    train_df = train_df.fillna(0)

    train_columns = list(train_df.columns)
    train_columns.remove('label')
    train_columns.remove('time_id')
    train_columns.remove('stock_id')

    X = train_df[train_columns]
    y = train_df['label']
    cv = CombinatorialPurgedGroupKFold(n_splits = 5, n_test_splits = 1)

    # cat_parameters = {
    # "iterations": 1500,
    # "learning_rate": 0.028,
    # # "task_type" : "GPU",
    # "depth": 5,
    # "verbose" : 1000,
    # "eval_metric": "RMSE",
    # "objective": "RMSE",
    # 'l2_leaf_reg': 8.83,
    # 'subsample': 0.85,
    # 'rsm': 0.4,
    # 'early_stopping_rounds': 20,
    # 'random_seed': args.seed,
    # }
    cat_parameters = {
    "iterations": 1500,
    "learning_rate": 0.028,
    # "task_type" : "GPU",
    "depth": 10,
    "verbose" : 1000,
    "eval_metric": "RMSE",
    "objective": "RMSE",
    'l2_leaf_reg': 5.0,
    'subsample': 0.85,
    'min_data_in_leaf':500,
    'rsm': 0.4,
    'early_stopping_rounds': 20,
    'random_seed': args.seed,
    }
    time_train= train_df['time_id'].copy()
    df_val = train_df[['stock_id','time_id','label']]
    res_vec = np.zeros((nfolds, 1))
    for (ii, (id0, id1)) in enumerate(cv.split(X, groups = pd.DataFrame(time_train)['time_id'])):
        x0, x1 = X.loc[id0], X.loc[id1]
        y0, y1 = y.loc[id0], y.loc[id1]
        
        df_val_tmp = df_val.iloc[id1]
        
        model = CatBoostRegressor(**cat_parameters)
        model.fit(x0, y0, eval_set=[ (x1, y1)], use_best_model = True)
        val_preds = model.predict(x1)
        # validation score    
        df_val_tmp['pred'] = val_preds
        # validation score    
        score = df_val_tmp.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()

        res_vec[ii] = score
        logger.info("validation score: " + str(score))
        logger.info("test score: " + str(score))
        model_path = './models/catboost_final' + str(args.seed) + str(ii) + '.pkl'

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
  
    logger.info("overall score " + str(np.mean(res_vec)))

    
if __name__ == '__main__':
    main()
