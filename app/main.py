from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd, json, subprocess, sys, numpy as np, joblib

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'outputs'; MODELS=ROOT/'models'
app=FastAPI(title='Energy Forecasting Studio', version='2.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')

DATASETS={
 'region_a':{'name':'Region A','raw':'region_a_messy_raw.csv','clean':'region_a_clean_reference.csv','scale':'Grid / Regional'},
 'building_b':{'name':'Building B','raw':'building_b_messy_raw.csv','clean':'building_b_clean_reference.csv','scale':'Commercial Building'},
 'home_c':{'name':'Home C','raw':'home_c_messy_raw.csv','clean':'home_c_clean_reference.csv','scale':'Residential'}
}

@app.get('/')
def home(): return FileResponse(ROOT/'static/index.html')

def read_raw(ds):
    df=pd.read_csv(DATA/DATASETS[ds]['raw'])
    df['Timestamp']=pd.to_datetime(df['Timestamp'], errors='coerce')
    return df.dropna(subset=['Timestamp']).sort_values('Timestamp')

@app.get('/api/datasets')
def datasets():
    out=[]
    for k,v in DATASETS.items():
        df=read_raw(k)
        out.append({'id':k,**v,'rows':len(df),'columns':len(df.columns),'start':str(df.Timestamp.min()),'end':str(df.Timestamp.max()),
                    'missing_cells':int(df.isna().sum().sum()),'duplicate_timestamps':int(df.Timestamp.duplicated().sum())})
    return out

def load_outputs(ds):
    required=[f'{ds}_metrics.csv',f'{ds}_forecast.csv',f'{ds}_feature_importance.csv',f'{ds}_data_quality.csv',f'{ds}_explanation.csv',f'{ds}_run.json']
    if not all((OUT/x).exists() for x in required): return None
    metrics=pd.read_csv(OUT/f'{ds}_metrics.csv'); forecast=pd.read_csv(OUT/f'{ds}_forecast.csv')
    imp=pd.read_csv(OUT/f'{ds}_feature_importance.csv'); quality=pd.read_csv(OUT/f'{ds}_data_quality.csv')
    exp=pd.read_csv(OUT/f'{ds}_explanation.csv'); run=json.load(open(OUT/f'{ds}_run.json'))
    return {'metrics':metrics.to_dict('records'),'forecast':forecast.tail(240).to_dict('records'),'importance':imp.head(15).to_dict('records'),
            'quality':quality.to_dict('records'),'explanation':exp.to_dict('records'),'run':run}

@app.get('/api/results/{ds}')
def results(ds):
    if ds not in DATASETS: raise HTTPException(404,'Unknown dataset')
    data=load_outputs(ds)
    if data is None: return {'ready':False}
    return {'ready':True,**data}

@app.post('/api/run/{ds}')
def run(ds):
    if ds not in DATASETS: raise HTTPException(404,'Unknown dataset')
    p=subprocess.run([sys.executable,'-m','app.pipeline','--dataset',ds,'--quick'],cwd=ROOT,capture_output=True,text=True,timeout=300)
    if p.returncode!=0: raise HTTPException(500,p.stderr[-3000:])
    return {'ready':True,'message':'Pipeline completed','output':p.stdout[-3000:]}

# ---------- User-facing forecast ----------
def clean_for_forecast(ds):
    # Use the same cleaning logic as the pipeline so the dashboard and training data agree.
    from .pipeline import clean_data
    raw=read_raw(ds)
    clean,_=clean_data(raw)
    return clean

def build_feature_row(history, ts):
    """Create one XGBoost feature row. Lag/rolling features use only data before ts."""
    h=history.copy().sort_values('Timestamp').set_index('Timestamp')
    y=h['Consumption'].astype(float)
    row={}
    # Source variables: future values are taken from the same hour one week earlier.
    source_cols=['Temperature_C','Humidity_pct','Wind_Speed_mps','Solar_Radiation_Wm2','Is_Holiday','Occupancy_Index','Dew_Point_C','Cloud_Cover_pct','Rainfall_mm','Pressure_hPa','Electricity_Price']
    ref_ts=ts-pd.Timedelta(hours=168)
    ref=h.loc[ref_ts] if ref_ts in h.index else h.iloc[-1]
    for c in source_cols:
        if c in h.columns: row[c]=float(ref[c]) if pd.notna(ref[c]) else float(h[c].dropna().iloc[-1])
    row['Hour']=ts.hour; row['DayOfWeek']=ts.dayofweek; row['Month']=ts.month; row['Is_Weekend']=int(ts.dayofweek>=5)
    for lag in [1,2,3,24,48,168]:
        t=ts-pd.Timedelta(hours=lag)
        row[f'lag_{lag}']=float(y.loc[t]) if t in y.index else float(y.iloc[-1])
    shifted=y.loc[:ts-pd.Timedelta(hours=1)]
    for w in [6,24,168]:
        vals=shifted.tail(w)
        row[f'roll_mean_{w}']=float(vals.mean()); row[f'roll_std_{w}']=float(vals.std() if len(vals)>1 else 0)
        row[f'roll_min_{w}']=float(vals.min()); row[f'roll_max_{w}']=float(vals.max())
    row['hour_sin']=np.sin(2*np.pi*ts.hour/24); row['hour_cos']=np.cos(2*np.pi*ts.hour/24)
    row['dow_sin']=np.sin(2*np.pi*ts.dayofweek/7); row['dow_cos']=np.cos(2*np.pi*ts.dayofweek/7)
    return row

def demand_band(pred, history):
    y=history['Consumption'].astype(float)
    # Dataset-relative thresholds: bottom 40%, middle 40%, top 20%.
    q40=float(y.quantile(.40)); q80=float(y.quantile(.80))
    if pred>=q80: return 'HIGH', q80, q40
    if pred>=q40: return 'MEDIUM', q80, q40
    return 'LOW', q80, q40

@app.get('/api/forecast-now/{ds}')
def forecast_now(ds, horizon:int=24):
    if ds not in DATASETS: raise HTTPException(404,'Unknown dataset')
    if horizon not in {1,24,168}: raise HTTPException(400,'Horizon must be 1, 24 or 168 hours')
    model_path=MODELS/f'{ds}_xgboost.joblib'
    if not model_path.exists():
        raise HTTPException(409,'Run the pipeline for this dataset first.')
    model=joblib.load(model_path)
    from .pipeline import make_features
    history=clean_for_forecast(ds)
    history['Timestamp']=pd.to_datetime(history['Timestamp'])
    latest=history['Timestamp'].max()
    imp=pd.read_csv(OUT/f'{ds}_feature_importance.csv') if (OUT/f'{ds}_feature_importance.csv').exists() else pd.DataFrame()
    feature_cols=[c for c in make_features(history).columns if c not in {'Timestamp','Consumption'} and pd.api.types.is_numeric_dtype(make_features(history)[c])]
    work=history[['Timestamp','Consumption']+[c for c in history.columns if c not in {'Timestamp','Consumption'}]].copy()
    forecasts=[]
    for step in range(1,horizon+1):
        ts=latest+pd.Timedelta(hours=step)
        row=build_feature_row(work,ts)
        X=pd.DataFrame([row])
        for c in feature_cols:
            if c not in X: X[c]=0
        X=X[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
        pred=float(model.predict(X)[0]); band,high_thr,mid_thr=demand_band(pred,work)
        forecasts.append({'Timestamp':ts.isoformat(),'predicted_kWh':round(pred,2),'demand_level':band,'step':step})
        # recursive prediction becomes the next step's lag input
        new={c:row.get(c,np.nan) for c in work.columns if c not in {'Timestamp','Consumption'}}
        new['Timestamp']=ts; new['Consumption']=pred
        work=pd.concat([work,pd.DataFrame([new])],ignore_index=True)
    first=forecasts[0]
    # Driver explanation: signed deviation of important features from their recent average,
    # weighted by model importance. This is a transparent approximation, not a SHAP value.
    last_row=build_feature_row(work.iloc[:-horizon],pd.Timestamp(first['Timestamp']))
    drivers=[]
    if not imp.empty:
        recent=work.iloc[:-horizon]
        for _,r in imp.head(8).iterrows():
            f=r['Feature']; val=last_row.get(f)
            if val is None: continue
            baseline=float(recent[f].tail(168).mean()) if f in recent.columns else 0
            delta=float(val-baseline)
            drivers.append({'feature':f,'importance_pct':round(float(r['Importance'])*100,2),'direction':'up' if delta>=0 else 'down','relative_change_pct':round((delta/(abs(baseline)+1e-9))*100,1)})
        drivers=sorted(drivers,key=lambda x: abs(x['relative_change_pct'])*x['importance_pct'],reverse=True)[:5]
    return {'dataset':ds,'dataset_name':DATASETS[ds]['name'],'latest_observation':latest.isoformat(),
            'forecast_start':forecasts[0]['Timestamp'],'horizon':horizon,'first':first,'forecasts':forecasts,
            'thresholds':{'medium_from_kWh':round(float(demand_band(0,history)[2]),2),'high_from_kWh':round(float(demand_band(0,history)[1]),2)},
            'drivers':drivers,'note':'Forecast starts after the latest timestamp available in the supplied CSV. Future weather/exogenous values are approximated from the same hour one week earlier for this synthetic-data demo.'}

@app.post('/api/deep-train/{ds}/{model_name}')
def deep_train(ds, model_name):
    if ds not in DATASETS or model_name not in {'LSTM','GRU','CNN-LSTM'}: raise HTTPException(400,'Choose LSTM, GRU or CNN-LSTM')
    try:
        from .deep_models import train_deep
        return train_deep(ds, model_name, epochs=3, max_train=4000)
    except Exception as e: raise HTTPException(500,str(e))

@app.get('/api/deep-results/{ds}')
def deep_results(ds):
    import glob
    files=glob.glob(str(OUT/f'{ds}_*_deep.json'))
    return [json.load(open(f)) for f in files]

@app.get('/api/preview/{ds}')
def preview(ds):
    if ds not in DATASETS: raise HTTPException(404,'Unknown dataset')
    df=pd.read_csv(DATA/DATASETS[ds]['raw'])
    return {'columns':list(df.columns),'rows':df.head(10).fillna('').to_dict('records')}
