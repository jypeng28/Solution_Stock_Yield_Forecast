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
from sklearn.preprocessing import scale
import warnings 
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



def scale_by_time_id(df, features):
    
    def lambda_scale(d):
        d = scale(d)
        return d
    
    data = df.groupby("time_id").apply(lambda x: lambda_scale(x[features]))
    data = np.concatenate(data.values)
    
    df[features] = data
    
    return df

def main():
    nfolds = 5
    parser = argparse.ArgumentParser(description='Validate Performance with Lightgbm baseline')
    parser.add_argument('--data_path', type=str, default='train_dropna')
    args = parser.parse_args()

    log_path = './logs/' + args.data_path + '.log'

    data_path = 'train_dropna.pkl'
    
    logger = Logger.init_logger(filename=log_path)
    

    train_df = pd.read_pickle(data_path)

    # pca = PCA(n_components=2).fit(train_df)
    
    # train_df = compute_rank_percentile(train_df, col_features_org, pca)
    # train_df = scale_by_time_id(train_df, col_features_org)
    # # del last important features

    # train_df = train_df.drop(['259', '239', '250', '262', '45', '233', '279', '84', '216', '243', '225', '282'], axis=1)
# feat_top_100_time_corr

    # feat_top_100_time_pd = pd.read_pickle('feat_top_100_time_corr.pkl')
    # train_df = pd.concat([train_df,feat_top_100_time_pd],axis=1)
    # for col in tqdm(best_corr):
    #     mapper = train_df.groupby(['time_id'])[col].mean().to_dict()
    #     train_df[f'time_id_{col}_mean'] = train_df['time_id'].map(mapper)
    #     train_df[f'time_id_{col}_mean'] = train_df[f'time_id_{col}_mean'].astype(np.float16)

    #     mapper = train_df.groupby(['time_id'])[col].var().to_dict()
    #     train_df[f'time_id_{col}_var'] = train_df['time_id'].map(mapper)
    #     train_df[f'time_id_{col}_var'] = train_df[f'time_id_{col}_var'].astype(np.float16)

# best_col_selected
    # best_feats = ['212','118','221','95','86','191','156','155','275','185']
    # for col in tqdm(best_feats):
    #     mapper = train_df.groupby(['time_id'])[col].mean().to_dict()
    #     train_df[f'time_id_{col}_mean'] = train_df['time_id'].map(mapper)
    #     train_df[f'time_id_{col}_mean'] = train_df[f'time_id_{col}_mean'].astype(np.float16)

    #     mapper = train_df.groupby(['time_id'])[col].var().to_dict()
    #     train_df[f'time_id_{col}_var'] = train_df['time_id'].map(mapper)
    #     train_df[f'time_id_{col}_var'] = train_df[f'time_id_{col}_var'].astype(np.float16)

    # feat_top_10_time_pd = pd.read_pickle('feat_top_10_time_corr.pkl')
    # train_df = pd.concat([train_df,feat_top_10_time_pd],axis=1)
#

# # feat_top_10_cross
#     feat_top_10_cross_pd = pd.read_pickle('feat_top_10_cross_filtered.pkl')
#     train_df = pd.concat([train_df,feat_top_10_cross_pd],axis=1)
# #

    with open('feature_corr_list.pkl', 'rb') as f:
        best_feats = pickle.load(f)

    # feat_top_100_stock_before
    feat_top_100_stock_before_pd = pd.read_pickle('top50_feat_stock_before.pkl')
    train_df = pd.concat([train_df,feat_top_100_stock_before_pd],axis=1)
    # 

# Week
    # week_df = pd.read_pickle('week.pkl')
    # train_df = pd.concat([train_df,week_df],axis=1)
    # train_df['week'] = (train_df['time_id']%7).astype('category')

# # #



# #
# # Lagging 2
#     lagging_2_df = pd.read_pickle('lagging_df2.pkl')
#     train_df = pd.concat([train_df,lagging_2_df],axis=1)
# #

# # Lagging 3
#     lagging_3_df = pd.read_pickle('lagging_df3.pkl')
#     train_df = pd.concat([train_df,lagging_3_df],axis=1)
# #

    # train_df = train_df.sample(frac=0.2,replace=False,axis=0)
# New idea:

    # train_df['global_var'] = train_df.var(axis=1)
    float_columns = list(train_df.columns)

    float_columns.remove('label')
    float_columns.remove('time_id')
    float_columns.remove('stock_id')
    # float_columns.remove('week')
 
    train_df = scale_by_time_id(train_df, float_columns)
# # Lagging 1
    for fe in tqdm(best_feats[:100]):
        train_df[f"lagging_1_{fe}_rate"] = train_df.groupby("stock_id")[fe].diff(1)
        train_df[f"lagging_1_{fe}_rate"] = train_df[f"lagging_1_{fe}_rate"].fillna(0).astype(np.float16)


    mapper = train_df.groupby(['time_id'])['0'].count().to_dict()
    train_df['new'] =train_df['time_id'].map(mapper)
    train_df = train_df.reset_index(drop=True)
    time_train= train_df['time_id'].copy()
    df_val = train_df[['stock_id','time_id','label']]
    train_df.drop(['stock_id','time_id'],axis=1,inplace=True)
    
    train_columns = list(train_df.columns)
    train_columns.remove('label')

    # for col in tqdm(train_columns):
    #     data_transformed = preprocessing.quantile_transform(train_df[col].values.reshape(-1,1))
    #     train_df[col] = data_transformed.reshape(-1)
    
    
    
    X = train_df[train_columns]
    y = train_df['label']
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=10)
    # Lightgbm baseline
    # lgb_parameters = {

    #     'learning_rate':0.05,
    #     "objective": "regression",
    #     "metric": "rmse",
    #     'boosting_type': "gbdt",
    #     'verbosity': -1,
    #     'n_jobs': -1, 
    #     'seed': 43,
    #     'lambda_l1': 0.03627602394442367, 
    #     'lambda_l2': 0.43523855951142926, 
    #     'num_leaves': 114, 
    #     'feature_fraction': 0.9505625064462319, 
    #     'bagging_fraction': 0.9785558707339647, 
    #     'bagging_freq': 7, 
    #     'max_depth': -1, 
    #     'max_bin': 501, 
    #     'min_data_in_leaf': 374,
    #     'n_estimators': 1000, 
    # }
    lgb_parameters = {
        'learning_rate':0.05,
        "objective":"regression",
        "metric": "rmse",
        'boosting_type': "gbdt",
        'verbosity': -1,
        'n_jobs': -1, 
        'seed': 43,
        'lambda_l1': 2.8, 
        'lambda_l2': 0.0006, 
        'num_leaves': 20, 
        'feature_fraction': 0.4, 
        'subsample':0.6,
        'bagging_freq': 9, 
        'max_depth': 8, 
        'max_bin': 500, 
        'min_child_samples': 500,
        'n_estimators': 3000, 
    }
    res_vec = np.zeros((nfolds, 1))
    for (ii, (id0, id1)) in enumerate(cv.split(X, groups = pd.DataFrame(time_train)['time_id'])):

        x0, x1 = X.loc[id0], X.loc[id1]
        y0, y1 = y.loc[id0], y.loc[id1]
        
        df_val_tmp = df_val.iloc[id1]
        
        model = lgb.LGBMRegressor(**lgb_parameters)
        model.fit(x0, y0, eval_metric='rmse', eval_set=[(x0, y0), (x1, y1)],
                  verbose= 500, early_stopping_rounds=100)
        val_preds = model.predict(x1)
        # validation score    
        df_val_tmp['pred'] = val_preds
        # validation score    
        score = df_val_tmp.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()

        res_vec[ii] = score
        logger.info("validation score: " + str(score))
        logger.info("test score: " + str(score))
        model_path = './models/' + args.data_path + str(ii) + '.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
  
    logger.info("overall score " + str(np.mean(res_vec)))

    
if __name__ == '__main__':
    main()