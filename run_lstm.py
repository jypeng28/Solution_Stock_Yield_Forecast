import numpy as np
import pandas as pd
import random
import os
import gc

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import KFold

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle
import pdb
device = torch.device("cuda")

class GCF:
    INPUT_ROOT = "./"
    SEED = 0
    MAX_LEN = 1024
    EVAL_MAX_LEN = 1024
    N_FOLDS = 5
    
    BS = 256 * 1
    HIDDEN_SIZE = 256
    N_EPOCHS = 80 * 2
    
    LR = 1e-3
    WEIGHT_DECAY = 1e-5

def set_seed(seed=GCF.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
class UMPLSTM(nn.Module):
    def __init__(self):
        super(UMPLSTM, self).__init__()

        self.noise = AddGaussianNoise(std=0.2)
        
        # self.ae_encoder = nn.Linear(601, 128)
        # self.ae_act = nn.ReLU()
        # self.ae_decoder = nn.Linear(128, 601)
            
        self.rnn = nn.LSTM(601+128, GCF.HIDDEN_SIZE, batch_first=True, num_layers=1, dropout=0.0)
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Sequential(
            nn.Linear(GCF.HIDDEN_SIZE+601, GCF.HIDDEN_SIZE),
            nn.LayerNorm(GCF.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(GCF.HIDDEN_SIZE, 1),
        )
        
    def forward(self, _x, _y=None):
        if self.training:
            h = self.noise(_x)
        else:
            h = _x
            
        # ae_h1 = self.ae_act(self.ae_encoder(h))
        # ae_h2 = self.ae_decoder(ae_h1)
        # ae_loss = nn.MSELoss()(ae_h2, h)
        
        # h = torch.cat([_x, ae_h1], dim=2)
        
        h, _ = self.rnn(h)
        h = self.dropout(h)
        h = torch.cat([_x, h], dim=2)
        regr = self.head(h)
        regr = regr.squeeze(2)
        
        if _y is None:
            return None, regr

        mask = (_y != 999).float()
        loss = nn.MSELoss(reduction='none')(regr, _y)
        loss = (loss * mask).mean()
        
        loss = loss 
        # + ae_loss
        
        return loss, regr






test_df = pd.read_pickle('./test_df.pkl')
test_df = test_df.fillna(0)

test_columns = list(test_df.columns)
test_columns.remove('time_id')
test_columns.remove('stock_id')

sorted_test_df = test_df.sort_values(['stock_id', 'time_id'])
X = sorted_test_df[test_columns].values
stock_id = sorted_test_df['stock_id'].values

all_ids = np.unique(stock_id)
all_index = sorted_test_df.index.values

class UMPDataset(Dataset):
    
    def __init__(self, is_train):
        self.ids = all_ids
        self.l = GCF.MAX_LEN if is_train else GCF.EVAL_MAX_LEN
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, item):
        _id = self.ids[item]
        _x = X[stock_id == _id, :]
        _index = all_index[stock_id == _id]

        if len(_x) > self.l:
            tail_i = len(_x) - 1
            _x = _x[tail_i-self.l:tail_i, :]
            _index = _index[tail_i-self.l:tail_i, :]
        elif len(_x) < self.l:
            pad_len = self.l - len(_x)
            x_pad = np.zeros((pad_len, 601))
            index_pad = np.zeros(pad_len) -1
            _x = np.vstack([x_pad, _x])

            _index = np.hstack([index_pad, _index])
        return _x, _index



test_dset = UMPDataset(False)

preds = np.zeros((5 * 1,len(test_df))).tolist()
for seed in [0]:
    for i in range(5):
        model_path = './models/lstm_v2' + str(seed) + str(i) + '.pkl'
        # 读取模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model.to(device)
        model.eval()

        
        
        test_dloader = DataLoader(test_dset, batch_size=GCF.BS,
                               pin_memory=True, shuffle=False, drop_last=False)
        for _x, _index in test_dloader:

            with torch.no_grad():
                loss, regr = model(_x.float().to(device))

            for j in range(len(_index)):
                for k in range(len(_index[j])):
                    idx = int(_index[j][k])
                    if idx >=0 :
                        preds[i][idx] = regr[j][k].item()
preds = np.mean(preds,axis=0)
test_df['pred'] = preds

test_df.drop(['time_id', 'stock_id'],axis=1,inplace=True)   

test_label = pd.read_csv('./test_label.csv')
result = pd.concat([test_df, test_label], axis=1)
rank_ic = result.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
print('rank_ic: ', rank_ic)

pdb.set_trace()
    