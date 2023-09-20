import pandas as pd
import pickle
import numpy as np
from scipy.stats import pearsonr as p
from tqdm import tqdm
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings('ignore')
def add_mean_feature(group, col):
    # 计算之前所有时刻的"feature"列的平均值
    previous_features = group[col].expanding().mean().shift(1)
    group[f'mean_{col}'] = previous_features
    return group


def add_var_feature(group, col):
    # 计算之前所有时刻的"feature"列的平均值
    previous_features = group[col].expanding().var().shift(1)
    group[f'var_{col}'] = previous_features
    return group

def add_max_feature(group, col):
    # 计算之前所有时刻的"feature"列的平均值
    previous_features = group[col].expanding().max().shift(1)
    group[f'max_{col}'] = previous_features
    return group

def add_min_feature(group, col):
    # 计算之前所有时刻的"feature"列的平均值
    previous_features = group[col].expanding().min().shift(1)
    group[f'min_{col}'] = previous_features
    return group


def scale_by_time_id(df, features):
    
    def lambda_scale(d):
        d = scale(d)
        return d
    
    data = df.groupby("time_id").apply(lambda x: lambda_scale(x[features]))
    data = np.concatenate(data.values)
    
    df[features] = data
    
    return df



def feat_eng(df, train):


    
    scale_columns = list(df.columns)
    if train:
        scale_columns.remove('label')
    scale_columns.remove('time_id')
    scale_columns.remove('stock_id')

    df = scale_by_time_id(df, scale_columns)


    with open('feature_corr_list.pkl', 'rb') as f:
        feature_corr_list =pickle.load(f)

    best_corr = feature_corr_list[:20]


    for col in tqdm(best_corr):
        df = df.groupby("stock_id").apply(add_mean_feature, col=col)
        df = df.groupby("stock_id").apply(add_var_feature, col=col)
        df = df.groupby("stock_id").apply(add_max_feature, col=col)
        df = df.groupby("stock_id").apply(add_min_feature, col=col)


    for col in tqdm(best_corr):
        df[f"lagging_1_{col}_rate"] = df.groupby("stock_id")[col].diff(1)
        df[f"lagging_2_{col}_rate"] = df.groupby("stock_id")[col].diff(2)
        df[f"lagging_3_{col}_rate"] = df.groupby("stock_id")[col].diff(3)
    

    for col in tqdm(best_corr):
        mapper = df.groupby(['time_id'])[col].mean().to_dict()
        df[f'time_id_{col}_mean'] = df['time_id'].map(mapper)
        df[f'time_id_{col}_mean'] = df[f'time_id_{col}_mean']

        mapper = df.groupby(['time_id'])[col].var().to_dict()
        df[f'time_id_{col}_var'] = df['time_id'].map(mapper)
        df[f'time_id_{col}_var'] = df[f'time_id_{col}_var']

        mapper = df.groupby(['time_id'])[col].max().to_dict()
        df[f'time_id_{col}_max'] = df['time_id'].map(mapper)
        df[f'time_id_{col}_max'] = df[f'time_id_{col}_max']

        mapper = df.groupby(['time_id'])[col].min().to_dict()
        df[f'time_id_{col}_min'] = df['time_id'].map(mapper)
        df[f'time_id_{col}_min'] = df[f'time_id_{col}_min']
            
    mapper = df.groupby(['time_id'])['0'].count().to_dict()

    df['new'] =df['time_id'].map(mapper)

    float_columns = list(df.columns)
    if train:
        float_columns.remove('label')

        
    float_columns.remove('time_id')
    float_columns.remove('stock_id')
    float_columns.remove('new')
    if train:
        df[float_columns] = df[float_columns].fillna(0)

    return df

def solve_na_train(df):
    df = df.dropna(axis=0, how='any')
    return df

def solve_na_test(df):
    for i in tqdm(range(300)):
        col = str(i)  
        df[col] = df[col].fillna(df.groupby('time_id')[col].transform('mean'))
    return df

if __name__ == '__main__':
    df_train = pd.read_csv('train.csv')
    df_train = solve_na_train(df_train)
    df_train = feat_eng(df_train, True)
    df_train.to_pickle('train_df_v2.pkl')

    df_test = pd.read_csv('test.csv')
    df_test = solve_na_test(df_test)
    df_test = feat_eng(df_test, False)
    df_test.to_pickle('test_df_v2.pkl')




