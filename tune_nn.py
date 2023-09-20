import pandas as pd
from torch.nn import init
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.optim import lr_scheduler

import random
import numpy as np
from tqdm import tqdm
import optuna
from sklearn import linear_model
import pickle
from split import CombinatorialPurgedGroupKFold, PurgedGroupTimeSeriesSplit
import numpy as np
from scipy.stats import pearsonr as p
from sklearn.preprocessing import scale
import argparse
import logging
import os
import pdb
import warnings 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class RegressionColumnarDataset(data.Dataset):
    def __init__(self, df,  y):
        df = df.reset_index(drop=True)
        self.dfconts = df
        self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(np.float32)
        self.y = y.values

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        
        return [self.conts[idx], self.y[idx]]


class RegModel(nn.Module):
    def __init__(self, n_cont, emb_drop, out_sz, szs, drops):
        super().__init__()

        self.n_cont = n_cont

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

    def forward(self, x):


        for l,d,b in zip(self.lins, self.drops, self.bns):
            # changing order to fc - bn - relu - dropouts
            x = l(x)
            # x = b(x)
            x = F.silu(x) 
            x = d(x)
            
        # print('\n')
        x = self.outp(x)
                 
        return x.squeeze()
    

def correlation_loss(y_pred, y_true):
    x = y_pred.clone()
    y = y_true.clone()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cov = torch.sum(vx * vy)
    corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
    corr = torch.maximum(torch.minimum(corr,torch.tensor(1)), torch.tensor(-1))
    return torch.sub(torch.tensor(1), corr ** 2)
    

def RMSELoss(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred)) ** 2))



def objective(trial, X, Y, df_val, time_train):
    device = torch.device("cuda")
    param_grid = {
    "learning_rate": trial.suggest_float("learning_rate", 0.000001, 0.0001),
    "drop_out": trial.suggest_float("drop_out", 0.0001, 0.01),
    "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2),
    "corr_loss_w": trial.suggest_float("corr_loss_w",0.01, 10),
    }
    print(param_grid)
    cv = CombinatorialPurgedGroupKFold(n_splits = 5, n_test_splits = 1)
    train_columns = list(X.columns)
    n_continue_feature = len(train_columns)
    scores = np.zeros((5, 1))
    for (ii, (id0, id1)) in enumerate(cv.split(X, groups = pd.DataFrame(time_train)['time_id'])):

        x0, x1 = X.loc[id0], X.loc[id1]
        y0, y1 = Y.loc[id0], Y.loc[id1]
        
        df_val_tmp = df_val.iloc[id1]

        trainds = RegressionColumnarDataset(x0, y0)
        valds = RegressionColumnarDataset(x1, y1)

        traindl = data.DataLoader(trainds, batch_size = 2048*5, shuffle = True, num_workers = 2, pin_memory = True)
        valdl = data.DataLoader(valds, batch_size = 2048*5, shuffle = False, num_workers = 2, pin_memory = True)

        network = RegModel(n_continue_feature, param_grid["drop_out"], 1, [1024, 512, 128, 32], [param_grid["drop_out"], param_grid["drop_out"],  param_grid["drop_out"], param_grid["drop_out"]]).to(device)
        
        optimizer = optim.Adam(network.parameters(), lr=param_grid["learning_rate"], weight_decay=param_grid["weight_decay"])

        best_val_score = 0.0
        early_stop = 0
        for i in range(301):
            network.train()
            total_loss_train, total_loss_val = [], []
            pred_vals = []
            for cont_x, y in traindl:
            # loss
                cont_x = cont_x.to(device)
                y = y.to(device).float()
                y_pred = network(cont_x)
                loss = torch.sqrt(F.mse_loss(y_pred, y)) + param_grid["corr_loss_w"] * correlation_loss(y_pred, y)
                total_loss_train.append(loss.item())

                optimizer.zero_grad()
                
                # backprop
                loss.backward()  # update gradients
                optimizer.step() 
            network.eval()
            for cont_x, y in valdl:
                cont_x = cont_x.to(device)
                y = y.to(device).float()
                pred_val = network(cont_x)
                loss = torch.sqrt(F.mse_loss(pred_val, y))  + param_grid["corr_loss_w"] * correlation_loss(pred_val, y)
                total_loss_val.append(loss.item())
                pred_vals.extend(pred_val.cpu().detach().numpy().tolist())
            
            df_val_tmp['pred'] = pred_vals
            score = df_val_tmp.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
            if i % 10 == 0:
                print(f'epoch {i} train loss: {np.mean(total_loss_train)}')
                print(f'epoch {i} val loss: {np.mean(total_loss_val)}')
                print(f'epoch {i} val score: {score}')

            if np.isnan(score):
                scores[ii] = best_val_score
                print(f"fold{ii}:{best_val_score}")
                break

            if score > best_val_score:
                print("Best Epoch")
                best_val_score = score
                # torch.save(network.state_dict(), f'./models/{str(args.seed)}_{ii}.pth')
                early_stop = 0
                print(f'epoch {i} train loss: {np.mean(total_loss_train)}')
                print(f'epoch {i} val loss: {np.mean(total_loss_val)}')
                print(f'epoch {i} val score: {score}')
            else:
                early_stop += 1
                if early_stop >= 15 and i > 50:
                    scores[ii] = best_val_score
                    print(f"fold{ii}:{best_val_score}")
                    break
    print(f"average_score:{np.mean(scores)}")
    return np.mean(scores)




def main():
    nfolds = 5
    parser = argparse.ArgumentParser(description='Validate Performance with Lightgbm baseline')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    train_df = pd.read_pickle('train_df.pkl')
    train_df = train_df.reset_index(drop=True)
    time_train= train_df['time_id'].copy()
    df_val = train_df[['stock_id','time_id','label']]
    train_df.drop(['stock_id','time_id'],axis=1,inplace=True)
    train_columns = list(train_df.columns)
    train_columns.remove('label')
    X = train_df[train_columns]
    Y = train_df['label']
    study = optuna.create_study(direction="maximize", study_name="Catboost Classifier")
    func = lambda trial: objective(trial, X, Y,df_val, time_train)
    study.optimize(func, n_trials=50)
    print(study.best_params)
    print(study.best_value)
    print('Best trial:', study.best_trial.params)

    



if __name__ == '__main__':
    main()