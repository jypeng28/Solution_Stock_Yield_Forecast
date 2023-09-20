import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import warnings
import pdb
from torch.utils import data
import torch.nn as nn
import torch
import torch.nn.functional as F
warnings.filterwarnings('ignore')
name = 'raw_nn_corr_loss_42_'
#加载测试数据
device = torch.device("cuda:2")
class RegModel(nn.Module):
    def __init__(self, n_cont, emb_drop, out_sz, szs, drops,week, use_bn=True):
        super().__init__()
        self.week = week
        if self.week:
            self.emb = nn.embedding(7,32)
        self.n_cont = n_cont
        
        # embeddings are done, now concatatenate 
        szs = [self.n_cont] + szs

        self.lins = nn.ModuleList([nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(sz) for sz in szs[1:]])
        
        

        # simple lines to make sure the weights are initialised in a kaiming distribution
        for o in self.lins: nn.init.kaiming_normal_(o.weight.data)
            
        self.outp = nn.Linear(szs[-1], out_sz) # define output layer
        nn.init.kaiming_normal_(self.outp.weight.data)

        # define dropout layers
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        
        # define batch normalisation layers
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn = use_bn

    def forward(self, x_cat, x_cont):
        # print('initial shape HOW TO GET')
        if self.week:
            x_cat = self.emb(x_cat)
            x_cat = self.emb_drop(x_cat)
            x_cont = self.bn(x_cont)
            x = torch.cat([x_cat, x_cont], 1)
            # print('cat again', x.shape)
            
        else:
            x = self.bn(x_cont)

        for l,d,b in zip(self.lins, self.drops, self.bns):
            # changing order to fc - bn - relu - dropouts
            x = l(x)
            # print('linear', x.shape)
            if self.use_bn: x = b(x)
            x = F.silu(x) 
            x = d(x)

            
        # print('\n')
        x = self.outp(x)
        # print('output', x.shape)
        
            
        return x.squeeze()

class RegressionColumnarDataset(data.Dataset):
    def __init__(self, df,  week=False):
        df = df.reset_index(drop=True)
        self.week = week
        if self.week:
            self.dfcats = df['week']
            self.dfconts = df.drop(['week'], axis=1)
            self.cats = np.stack([c.values for n, c in self.dfcats.items()], axis=1).astype(np.int64)
        else:
            self.dfconts = df
        self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(np.float32)


    def __len__(self): return len(self.conts)

    def __getitem__(self, idx):
        if self.week:
            return self.cats[idx], self.conts[idx]
        else:
            return self.conts[idx]
        
test_df = pd.read_csv('./test.csv')


test_df.drop(['time_id', 'stock_id'],axis=1,inplace=True)   
week = False
testds = RegressionColumnarDataset(test_df,  week)
testdl = data.DataLoader(testds, batch_size = 2048, shuffle = False, num_workers = 2, pin_memory = True)
n_continue_feature = len(test_df.columns)
network = RegModel(n_continue_feature, 0.2, 1, [400, 800, 1000, 600, 400, 128, 8], [0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.01],week, use_bn=False).to(device)

preds = []

for i in range(5):
    pred_vals = []
    model_path = './models/' + name + str(i) + 'fold.pth'
# 读取模型
    model = torch.load(model_path)
    network.load_state_dict(model)
    for cont_x in testdl:
        cont_x = cont_x.to(device)
        pred_val = network(None,cont_x)
        pred_vals.extend(pred_val.cpu().detach().numpy().tolist())
    #生成预测
    preds.append(pred_vals)
    
preds = np.array(preds).mean(axis=0)
test_df['pred'] = preds

test_label = pd.read_csv('./test_label.csv')
result = pd.concat([test_df, test_label], axis=1)

rank_ic = result.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
print('rank_ic: ', rank_ic)




