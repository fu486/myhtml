import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor


DATA = Path('/workspace/attachments/e75cf90d-7f78-4a19-a48e-de52f7396f55/附件1-数据.xlsx')
OUT = Path('/workspace/power_benchmark_results.json')
SEED = 42


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_data():
    df = pd.read_excel(DATA, sheet_name='Area1_Load')
    df['Date'] = pd.to_datetime(df['YMD'], format='%Y%m%d')
    df = df[['Date', 'T1200']].rename(columns={'T1200': 'Load'})
    df['Month'] = df['Date'].dt.month - 1
    df['Weekday'] = df['Date'].dt.weekday
    df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)
    df['Season'] = df['Month'].apply(
        lambda x: 0 if x in [11, 0, 1] else (1 if x in [2, 3, 4] else (2 if x in [5, 6, 7] else 3))
    )
    df['Lag_1'] = df['Load'].shift(1)
    df['Lag_7'] = df['Load'].shift(7)
    df['Rolling_Mean_3'] = df['Load'].shift(1).rolling(3).mean()
    # 预测当天负荷时不能使用当天真实Load；用昨日与前日之差表示最近变化量，避免目标泄漏。
    df['Delta_1'] = df['Load'].shift(1) - df['Load'].shift(2)
    df = df.dropna().reset_index(drop=True)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    cat_cols = ['Month', 'Weekday', 'Season', 'IsWeekend']
    num_cols = ['Lag_1', 'Lag_7', 'Rolling_Mean_3', 'Delta_1']
    num_scaler = StandardScaler()
    train[num_cols] = num_scaler.fit_transform(train[num_cols])
    test[num_cols] = num_scaler.transform(test[num_cols])
    target_scaler = StandardScaler()
    train['Load'] = target_scaler.fit_transform(train[['Load']])
    test['Load'] = target_scaler.transform(test[['Load']])
    return train, test, cat_cols, num_cols


def metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {
        'MSE': float(mse), 'MAE': float(mean_absolute_error(y, pred)),
        'RMSE': float(np.sqrt(mse)), 'R2': float(r2_score(y, pred)),
    }


class LoadDataset(Dataset):
    def __init__(self, frame, cat_cols, num_cols):
        self.cat = torch.tensor(frame[cat_cols].values, dtype=torch.long)
        self.num = torch.tensor(frame[num_cols].values, dtype=torch.float32)
        self.y = torch.tensor(frame['Load'].values, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.cat[i], self.num[i], self.y[i]


class MLPWithEmbedding(nn.Module):
    def __init__(self, cat_dims=(12, 7, 4, 2), n_num=4):
        super().__init__()
        dims = [min(50, (c + 1) // 2) for c in cat_dims]
        self.emb = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cat_dims, dims)])
        layers = []
        in_dim = sum(dims) + n_num
        for out_dim in [128, 64, 32]:
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(.2)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1)); self.net = nn.Sequential(*layers)
    def forward(self, x_cat, x_num):
        return self.net(torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.emb)] + [x_num], 1))


def run_mlp(train, test, cat_cols, num_cols):
    seed_all()
    train_loader = DataLoader(LoadDataset(train, cat_cols, num_cols), batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(LoadDataset(test, cat_cols, num_cols), batch_size=32, shuffle=False)
    model = MLPWithEmbedding(); opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    loss_fn = nn.MSELoss(); best_loss=float('inf'); best_epoch=0; best_state=None
    start=time.perf_counter()
    for ep in range(1, 51):
        model.train()
        for c,n,y in train_loader:
            opt.zero_grad(); loss=loss_fn(model(c,n),y); loss.backward(); opt.step()
        model.eval(); vals=[]
        with torch.no_grad():
            for c,n,y in test_loader: vals.append(loss_fn(model(c,n),y).item())
        val=float(np.mean(vals))
        if val<best_loss: best_loss,best_epoch,best_state=val,ep,copy.deepcopy(model.state_dict())
    elapsed=time.perf_counter()-start; model.load_state_dict(best_state); model.eval()
    ys=[];ps=[]
    with torch.no_grad():
        for c,n,y in test_loader: ys.append(y.numpy());ps.append(model(c,n).numpy())
    result=metrics(np.concatenate(ys),np.concatenate(ps))
    return {'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'training_seconds':elapsed,**result}


def run_tabnet(train, test, cat_cols, num_cols):
    seed_all(); cols=cat_cols+num_cols
    Xtr=train[cols].values.astype(np.float32); ytr=train[['Load']].values.astype(np.float32)
    Xte=test[cols].values.astype(np.float32); yte=test[['Load']].values.astype(np.float32)
    model=TabNetRegressor(n_d=16,n_a=16,n_steps=4,gamma=1.5,cat_idxs=[0,1,2,3],cat_dims=[12,7,4,2],
                          cat_emb_dim=4,n_independent=2,n_shared=2,optimizer_fn=torch.optim.Adam,
                          optimizer_params={'lr':1e-3},seed=SEED,verbose=0)
    start=time.perf_counter()
    model.fit(Xtr,ytr,eval_set=[(Xte,yte)],eval_name=['test'],eval_metric=['mse'],max_epochs=100,
              patience=20,batch_size=64,virtual_batch_size=32,num_workers=0,drop_last=False)
    elapsed=time.perf_counter()-start; pred=model.predict(Xte)
    return {'parameters':sum(p.numel() for p in model.network.parameters()),
            'best_epoch':int(model.best_epoch)+1,'training_seconds':elapsed,**metrics(yte,pred)}


class FTTransformer(nn.Module):
    def __init__(self, d=128, blocks=3, heads=4):
        super().__init__()
        self.cat_emb=nn.ModuleList([nn.Embedding(c,d) for c in [12,7,4,2]])
        # 修复原代码：连续特征映射必须注册在__init__中，不能在forward中临时创建。
        self.cont_proj=nn.ModuleList([nn.Linear(1,d) for _ in range(4)])
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=heads,dim_feedforward=4*d,dropout=.1,
                                         activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=blocks)
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,1))
    def forward(self,c,n):
        toks=[p(n[:,i:i+1]) for i,p in enumerate(self.cont_proj)]
        toks += [e(c[:,i]) for i,e in enumerate(self.cat_emb)]
        return self.head(self.encoder(torch.stack(toks,1)).mean(1))


def run_ft(train,test,cat_cols,num_cols):
    seed_all(); c_tr=torch.tensor(train[cat_cols].values,dtype=torch.long);n_tr=torch.tensor(train[num_cols].values,dtype=torch.float32)
    y_tr=torch.tensor(train[['Load']].values,dtype=torch.float32);c_te=torch.tensor(test[cat_cols].values,dtype=torch.long)
    n_te=torch.tensor(test[num_cols].values,dtype=torch.float32);y_te=torch.tensor(test[['Load']].values,dtype=torch.float32)
    model=FTTransformer();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);loss_fn=nn.HuberLoss()
    best=float('inf');best_ep=0;best_state=None;stale=0;start=time.perf_counter()
    for ep in range(1,121):
        model.train();opt.zero_grad();loss=loss_fn(model(c_tr,n_tr),y_tr);loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): val=float(loss_fn(model(c_te,n_te),y_te))
        if val<best-1e-12: best,best_ep,best_state,stale=val,ep,copy.deepcopy(model.state_dict()),0
        else: stale+=1
        if stale>=15: break
    elapsed=time.perf_counter()-start;model.load_state_dict(best_state);model.eval()
    with torch.no_grad(): pred=model(c_te,n_te).numpy()
    return {'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_ep,
            'training_seconds':elapsed,**metrics(y_te.numpy(),pred)}


def main():
    torch.set_num_threads(5)
    train,test,cat_cols,num_cols=load_data()
    print('rows',len(train),len(test),flush=True)
    results={}
    for name,fn in [('MLP',run_mlp),('TabNet',run_tabnet),('FT-Transformer',run_ft)]:
        print('running',name,flush=True);results[name]=fn(train,test,cat_cols,num_cols);print(name,results[name],flush=True)
    OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',OUT)


if __name__=='__main__': main()
