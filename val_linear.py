import pandas as pd
from sklearn import linear_model
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



def main():
    nfolds = 5
    parser = argparse.ArgumentParser(description='Validate Performance with Lightgbm baseline')
    parser.add_argument('--data_path', type=str, default='train_dropna')
    args = parser.parse_args()

    log_path = './logs/' + args.data_path + '.log'

    data_path = 'train_dropna.pkl'
 
    
    train_df = pd.read_pickle(data_path)
    # # del last important features

    # train_df = train_df.drop(['259', '239', '250', '262', '45', '233', '279', '84', '216', '243', '225', '282'], axis=1)
# feat_top_10_time_mean
    with open('feature_corr_list.pkl', 'rb') as f:
        best_feats =pickle.load(f)[:100]
    for col in tqdm(best_feats):
        mapper = train_df.groupby(['time_id'])[col].mean().to_dict()
        train_df[f'time_id_{col}_mean'] = train_df['time_id'].map(mapper)
        train_df[f'time_id_{col}_mean'] = train_df[f'time_id_{col}_mean'].astype(np.float16)

        mapper = train_df.groupby(['time_id'])[col].var().to_dict()
        train_df[f'time_id_{col}_var'] = train_df['time_id'].map(mapper)
        train_df[f'time_id_{col}_var'] = train_df[f'time_id_{col}_var'].astype(np.float16)


    # feat_top_10_time_pd = pd.read_pickle('feat_top_10_time_corr.pkl')
    # train_df = pd.concat([train_df,feat_top_10_time_pd],axis=1)
#

# # feat_top_10_cross
#     feat_top_10_cross_pd = pd.read_pickle('feat_top_10_cross_filtered.pkl')
#     train_df = pd.concat([train_df,feat_top_10_cross_pd],axis=1)
# #


# # feat_top_100_stock_before
#     feat_top_100_stock_before_pd = pd.read_pickle('feat_top_100_stock_before_filtered.pkl')
#     train_df = pd.concat([train_df,feat_top_100_stock_before_pd],axis=1)
# # 

# # Week
#     week_df = pd.read_pickle('week.pkl')
#     train_df = pd.concat([train_df,week_df],axis=1)
#     train_df['week'] = train_df['week'].astype('category')

# # #


# # Lagging 1
#     lagging_1_df = pd.read_pickle('lagging_df1.pkl')
#     train_df = pd.concat([train_df,lagging_1_df],axis=1)
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

    train_df = train_df.reset_index(drop=True)
    time_train= train_df['time_id'].copy()
    df_val = train_df[['stock_id','time_id','label']]
    train_df.drop(['stock_id','time_id'],axis=1,inplace=True)
    
    train_columns = list(train_df.columns)
    train_columns.remove('label')
    
    
    X = train_df[train_columns]
    y = train_df['label']
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=10)
    # Lightgbm baseline

    res_vec = np.zeros((nfolds, 1))
    for (ii, (id0, id1)) in enumerate(cv.split(X, groups = pd.DataFrame(time_train)['time_id'])):

        x0, x1 = X.loc[id0], X.loc[id1]
        y0, y1 = y.loc[id0], y.loc[id1]
        
        df_val_tmp = df_val.iloc[id1]
        
        model = linear_model.LinearRegression()
        model.fit(x0, y0)
        val_preds = model.predict(x1)
        # validation score    
        df_val_tmp['pred'] = val_preds
        # validation score    
        score = df_val_tmp.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()

        res_vec[ii] = score
        print("validation score: " + str(score))
        print("test score: " + str(score))

  
    print("overall score " + str(np.mean(res_vec)))

    
if __name__ == '__main__':
    main()