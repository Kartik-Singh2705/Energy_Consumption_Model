
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json, time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"

class LSTMForecaster(nn.Module):
    def __init__(self,n_features,hidden=64):
        super().__init__(); self.rnn=nn.LSTM(n_features,hidden,batch_first=True); self.fc=nn.Linear(hidden,1)
    def forward(self,x): return self.fc(self.rnn(x)[0][:,-1,:])

class GRUForecaster(nn.Module):
    def __init__(self,n_features,hidden=64):
        super().__init__(); self.rnn=nn.GRU(n_features,hidden,batch_first=True); self.fc=nn.Linear(hidden,1)
    def forward(self,x): return self.fc(self.rnn(x)[0][:,-1,:])

class CNNLSTMForecaster(nn.Module):
    def __init__(self,n_features,hidden=64):
        super().__init__(); self.conv=nn.Conv1d(n_features,n_features,kernel_size=3,padding=1); self.rnn=nn.LSTM(n_features,hidden,batch_first=True); self.fc=nn.Linear(hidden,1)
    def forward(self,x):
        x=torch.relu(self.conv(x.transpose(1,2))).transpose(1,2)
        return self.fc(self.rnn(x)[0][:,-1,:])

def train_deep(dataset="region_a", model_name="LSTM", epochs=3, max_train=4000, lookback=24):
    torch.manual_seed(42); np.random.seed(42)
    f=pd.read_csv(OUT/f"{dataset}_feature_importance.csv")
    selected=f[f.Selected==True].Feature.tolist()[:15]
    # Rebuild features through pipeline helper
    from .pipeline import clean_data, make_features, DATASETS, DATA
    raw=pd.read_csv(DATA/DATASETS[dataset]); clean,_=clean_data(raw); feat=make_features(clean).dropna().reset_index(drop=True)
    cols=["Consumption"]+selected
    cols=[c for c in cols if c in feat.columns]
    data=feat[cols].astype(float).replace([np.inf,-np.inf],np.nan).fillna(0)
    n=len(data); split=int(n*.85); train=data.iloc[:split]; test=data.iloc[split:]
    scaler=StandardScaler(); arr=scaler.fit_transform(train)
    all_arr=scaler.transform(data)
    X=[]; y=[]
    start=max(lookback,split-max_train)
    for i in range(start,split):
        X.append(all_arr[i-lookback:i]); y.append(all_arr[i,0])
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float32)
    tx=torch.tensor(X); ty=torch.tensor(y).view(-1,1)
    cls={"LSTM":LSTMForecaster,"GRU":GRUForecaster,"CNN-LSTM":CNNLSTMForecaster}[model_name]
    model=cls(len(cols),64); opt=torch.optim.Adam(model.parameters(),lr=.001); loss_fn=nn.MSELoss()
    t=time.time(); model.train()
    losses=[]
    for _ in range(epochs):
        perm=torch.randperm(len(tx)); total=0
        for i in range(0,len(tx),128):
            ix=perm[i:i+128]; pred=model(tx[ix]); loss=loss_fn(pred,ty[ix])
            opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()*len(ix)
        losses.append(total/len(tx))
    # test one-step recursively only using actual history, for a modest sample
    model.eval(); preds=[]; actual=[]
    with torch.no_grad():
        for i in range(split,min(n,split+1000)):
            xx=torch.tensor(all_arr[i-lookback:i],dtype=torch.float32).unsqueeze(0)
            pp=float(model(xx).item())
            # inverse only first dimension: scaler mean/scale of consumption
            pv=pp*scaler.scale_[0]+scaler.mean_[0]
            preds.append(pv); actual.append(float(data.iloc[i,0]))
    mae=float(np.mean(np.abs(np.array(actual)-np.array(preds))))
    rmse=float(np.sqrt(np.mean((np.array(actual)-np.array(preds))**2)))
    result={"dataset":dataset,"model":model_name,"lookback":lookback,"epochs":epochs,
            "train_windows":len(tx),"eval_rows":len(actual),"MAE":mae,"RMSE":rmse,
            "final_train_loss":losses[-1],"runtime_seconds":round(time.time()-t,2),"seed":42}
    torch.save(model.state_dict(),ROOT/"models"/f"{dataset}_{model_name.lower().replace('-','_')}.pt")
    with open(OUT/f"{dataset}_{model_name.lower().replace('-','_')}_deep.json","w") as fh: json.dump(result,fh,indent=2)
    return result
