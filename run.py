import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import warnings
import pdb
from sklearn.preprocessing import scale
import torch
warnings.filterwarnings('ignore')
import os
import pdb
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#加载测试数据
test_df = pd.read_pickle('./test_df.pkl')

def RMSELoss(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred)) ** 2))

class TestDataset(data.Dataset):
    def __init__(self, df):
        df = df.reset_index(drop=True)
        self.conts = np.stack([c.values for n, c in df.items()], axis=1).astype(np.float32)
    def __len__(self): return len(self.conts)
    def __getitem__(self, idx):
        return self.conts[idx]



    



test_df.drop(['time_id', 'stock_id'],axis=1,inplace=True)   
test_df = test_df.fillna(0)

    
preds = []
X = test_df.values




for seed in [42, 43, 44]:
    for i in range(5):
        model_path = './models/lgb_dart_final' + str(seed) + str(i) + '.pkl'
    # 读取模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        #生成预测
        y_pred = model.predict(X)
        preds.append(y_pred)


        

for seed in [42, 43, 44]:
    for i in range(5):
        model_path = './models/catboost_final' + str(seed) + str(i) + '.pkl'
    # 读取模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        #生成预测
        y_pred = model.predict(X)
        preds.append(y_pred)



weight_catboost = 0.4
weight_lgbm = 0.6



weights = [(weight_lgbm/10) for i in range(15)] + [(weight_catboost/10) for i in range(15)]
final_pred = np.average(preds, axis=0, weights=weights)

# preds = np.array(preds).mean(axis=0)
test_df['pred'] = final_pred

test_label = pd.read_csv('./test_label.csv')
result = pd.concat([test_df, test_label], axis=1)

rank_ic = result.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
print('rank_ic: ', rank_ic)



