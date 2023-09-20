import numpy as np
import pandas as pd
import random
import os
import gc
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import torch.optim as optim
from sklearn.model_selection import KFold, GroupKFold

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
scaler = torch.cuda.amp.GradScaler()


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
    WEIGHT_DECAY = 1e-6

def set_seed(seed=GCF.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_df = pd.read_pickle('train_df_1.pkl')

train_df.reset_index(drop=True, inplace=True)
train_df = train_df.fillna(0)
X = train_df.drop(['label','stock_id','time_id'],axis=1).values
y = train_df['label'].values

stock_id = train_df['stock_id'].values
time_id = train_df['time_id'].values

class UMPDataset(Dataset):
    
    def __init__(self, ids, is_train):
        self.ids = ids
        self.is_train = is_train
        self.l = GCF.MAX_LEN if is_train else GCF.EVAL_MAX_LEN
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, item):
        _id = self.ids[item]
        _x = X[stock_id == _id, :]
        _y = y[stock_id == _id]
        
        if len(_x) > self.l:
            if self.is_train:
                tail_i = random.randint(self.l, len(_x) - 1)
            else:
                tail_i = len(_x) - 1
            _x = _x[tail_i-self.l:tail_i, :]
            _y = _y[tail_i-self.l:tail_i]
        elif len(_x) < self.l:
            pad_len = self.l - len(_x)
            x_pad = np.zeros((pad_len, 601))
            y_pad = np.ones(pad_len) * 999
            _x = np.vstack([x_pad, _x])
            _y = np.hstack([y_pad, _y])
        
        return _x, _y
    

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

        self.noise = AddGaussianNoise(std=0.1)
        
        self.ae_encoder = nn.Linear(601, 256)
        self.ae_act = nn.ReLU()
        self.ae_decoder = nn.Linear(256, 601)
            
        self.rnn = nn.LSTM(601 + 256, GCF.HIDDEN_SIZE, batch_first=True, num_layers=1, dropout=0.01)
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Sequential(
            nn.Linear(GCF.HIDDEN_SIZE+601, GCF.HIDDEN_SIZE),
            nn.LayerNorm(GCF.HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(GCF.HIDDEN_SIZE, 1),
        )
        
    def forward(self, _x, _y=None):
        if self.training:
            h = self.noise(_x)
        else:
            h = _x
            
        ae_h1 = self.ae_act(self.ae_encoder(h))
        ae_h2 = self.ae_decoder(ae_h1)
        ae_loss = nn.MSELoss()(ae_h2, h)
        
        h = torch.cat([_x, ae_h1], dim=2)
        
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
        
        loss = loss + ae_loss
        
        return loss, regr
    
def train_loop(dloader, model):
    losses = []
    model.train()
    optimizer.zero_grad()
    for _x, _y in dloader:
        with torch.cuda.amp.autocast(): 
            loss, regr = model(_x.float().to(device), _y.float().to(device))
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update() 
        optimizer.zero_grad()
    return losses

def valid_loop(dloader, model):
    predicts = []
    model.eval()
    for _x, _y in dloader:
        with torch.no_grad():
            loss, regr = model(_x.float().to(device), _y.float().to(device))
        predicts.append(regr.cpu())
    predicts = torch.vstack(predicts)
    return predicts


def calc_score(ids, predicts):
    dfs = []
    for idx, _id in enumerate(ids):
        _time_id = time_id[stock_id == _id]
        _y = y[stock_id == _id]
        _time_id = _time_id[-GCF.EVAL_MAX_LEN:]
        _y = _y[-GCF.EVAL_MAX_LEN:]
        pred = predicts[idx, :].numpy()
        if len(_y) != GCF.EVAL_MAX_LEN:
            n_data = len(_y)
            pred = pred[-n_data:]

        df = pd.DataFrame(np.vstack([_time_id, _y, pred]).T, columns=['time_id', 'target', 'predict'])
        dfs.append(df)
    result_df = pd.concat(dfs, axis=0)
    
    time_count = result_df['time_id'].value_counts()
    result_df = result_df.query(f"time_id in {time_count[time_count > 1].index.tolist()}")
    score = np.mean(result_df.groupby('time_id').apply(lambda df: (df['predict'].rank()).corr(df['target'].rank())))
    return score


all_ids = np.unique(stock_id)

kf = KFold(n_splits=GCF.N_FOLDS, random_state=GCF.SEED, shuffle=True)
for fold, (train_idx, valid_idx) in enumerate(kf.split(all_ids)):
    print(f"Fold-{fold}")
    train_dset = UMPDataset(all_ids[train_idx], True)
    valid_dset = UMPDataset(all_ids[valid_idx], False)
    y_true = np.vstack([y for _, y in valid_dset])
    
    train_dloader = DataLoader(train_dset, batch_size=GCF.BS,
                               pin_memory=True, shuffle=True, drop_last=True,
                               worker_init_fn=lambda x: set_seed())
    valid_dloader = DataLoader(valid_dset, batch_size=GCF.BS,
                               pin_memory=True, shuffle=False, drop_last=False)
    
    model = UMPLSTM()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=GCF.LR, weight_decay=GCF.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, threshold=0.0001, min_lr=1e-8, verbose=True)
    
    set_seed()
    train_losses, valid_rmse, valid_scores = [], [], []
    best_score, best_rmse = float('-inf'), float('inf')
    early_stop = 0
    for epoch in tqdm(range(GCF.N_EPOCHS)):
        if early_stop >= 10:
            print('early stop!!')
            break
        losses = train_loop(train_dloader, model)
        predicts = valid_loop(valid_dloader, model)
        
        train_losses += losses

        rmse = mean_squared_error(y_true[y_true != 999], predicts[y_true != 999].numpy(), squared=False)
        score = calc_score(all_ids[valid_idx], predicts)
        
        scheduler.step(score)
        
        valid_rmse.append(rmse)
        valid_scores.append(score)
        
        print(f"  epoch: {epoch}, RMSE={rmse}, SCORE={score}")


        if best_score < score:
            best_score = score
            model_path = './models/lstm' + str(GCF.SEED) + str(fold) + '.pkl'
            # torch.save(model.state_dict(), model_path)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print('    -> best score update!!')

            early_stop = 0
        else:
            early_stop += 1
        
