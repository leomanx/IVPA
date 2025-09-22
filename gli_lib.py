# gli_lib.py
import numpy as np, pandas as pd, yfinance as yf
from fredapi import Fred
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def get_fred(api_key:str): return Fred(api_key=api_key)

def fred_series(fred, sid, start=None, end=None):
    s = fred.get_series(sid); s.index = pd.to_datetime(s.index)
    if start: s = s[s.index >= pd.to_datetime(start)]
    if end:   s = s[s.index <= pd.to_datetime(end)]
    return s.sort_index()

def to_weekly(s): return s.resample("W-FRI").last()
def yf_close(ticker, start=None, end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty: return pd.Series(dtype=float)
    col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if col is None: return pd.Series(dtype=float)
    s = df[col].copy(); s.index = pd.to_datetime(s.index); return s.sort_index()

def build_gli_proxy(fred, start="2008-01-01", end=None):
    S = {"FED":"WALCL","ECB":"ECBASSETSW","BOJ":"JPNASSETS","TGA":"WTREGEN","ONRRP":"RRPONTSYD","JPY":"DEXJPUS","EUR":"DEXUSEU"}
    raw = {k: fred_series(fred, sid, start, end) for k, sid in S.items()}
    wk = pd.DataFrame({
        "FED": to_weekly(raw["FED"]), "ECB": to_weekly(raw["ECB"]), "BOJ": to_weekly(raw["BOJ"]),
        "TGA": to_weekly(raw["TGA"]), "ONRRP": to_weekly(raw["ONRRP"]),
        "JPY": to_weekly(raw["JPY"]), "EUR": to_weekly(raw["EUR"]),
    }).dropna(how="any")
    wk["ECB_USD"] = wk["ECB"] * wk["EUR"]
    wk["BOJ_USD"] = wk["BOJ"] / wk["JPY"].replace(0,np.nan)
    wk["GLI_USD"] = wk["FED"] + wk["ECB_USD"] + wk["BOJ_USD"] - wk["TGA"] - wk["ONRRP"]
    wk["GLI_INDEX"] = 100*wk["GLI_USD"]/wk["GLI_USD"].iloc[0]
    return wk

def fetch_assets(fred, start="2008-01-01", end=None):
    req = {
        "NASDAQ":{"fred":"NASDAQCOM","yahoo":"^IXIC"},
        "SP500":{"fred":"SP500","yahoo":"^GSPC"},
        "GOLD":{"fred":None,"yahoo":"GC=F"},
        "BTC":{"fred":"CBBTCUSD","yahoo":"BTC-USD"},
        "ETH":{"fred":None,"yahoo":"ETH-USD"},
    }
    out={}
    for name, src in req.items():
        s = pd.Series(dtype=float)
        if src["fred"]:
            try:
                s = fred_series(fred, src["fred"], start, end).asfreq("D").ffill()
            except: s = pd.Series(dtype=float)
        if s.dropna().size < 2:
            s = yf_close(src["yahoo"], start, end)
        if s.dropna().size >= 2:
            out[name]=s
    if "GOLD" not in out:
        g = yf_close("GC=F", start, end)
        if g.dropna().size<2:
            g = yf_close("GLD", start, end)
        if g.dropna().size<2: raise RuntimeError("Gold price not available (GC=F/GLD)")
        out["GOLD"]=g
    return pd.concat(out, axis=1).asfreq("D").ffill()

def monthly_panels(wk, assets_df):
    gli_m = wk["GLI_INDEX"].resample("M").last().rename("GLI_INDEX")
    assets_m = assets_df.resample("M").last()
    monthly = pd.concat([gli_m, assets_m], axis=1).dropna(how="any")
    monthly_rets = monthly.pct_change().dropna()*100.0
    return monthly, monthly_rets

def annual_panel(wk, assets_df):
    gli_y = wk["GLI_INDEX"].resample("A-DEC").last().rename("GLI_INDEX")
    assets_y = assets_df.resample("A-DEC").last()
    return pd.concat([gli_y, assets_y], axis=1).dropna(how="any")

def rebase(s): s = s.dropna(); return 100.0*s/s.iloc[0]

def metrics_tables(monthly, monthly_rets, annual, rf_annual=0.02, years_n=10):
    def years(idx): return (pd.to_datetime(idx[-1])-pd.to_datetime(idx[0])).days/365.25
    def cagr(x): x=x.dropna(); return (x.iloc[-1]/x.iloc[0])**(1/years(x.index))-1 if len(x)>1 else np.nan
    def cagr_n(x,n):
        x=x.dropna(); cut=x.index[-1]-pd.DateOffset(years=n); x=x[x.index>=cut]
        return (x.iloc[-1]/x.iloc[0])**(1/years(x.index))-1 if len(x)>1 else np.nan
    def vol(r): r=r.dropna()/100.0; return r.std()*np.sqrt(12)
    def sharpe(r): r=r.dropna()/100.0; ex=r-(rf_annual/12); sd=r.std(); return np.nan if sd==0 else (ex.mean()/sd)*np.sqrt(12)
    def maxdd(series): s=series.dropna(); dd=s/s.cummax()-1; return dd.min()
    def calmar(c,m): return np.nan if (m is None or np.isnan(m) or m==0) else c/abs(m)

    rows=[]
    for a in [c for c in annual.columns if c!="GLI_INDEX"]:
        c_full=cagr(annual[a]); c_n=cagr_n(annual[a], years_n)
        gli_f=cagr(annual["GLI_INDEX"]); gli_n=cagr_n(annual["GLI_INDEX"], years_n)
        liq_f = c_full-gli_f if pd.notna(c_full) and pd.notna(gli_f) else np.nan
        liq_n = c_n-gli_n   if pd.notna(c_n)   and pd.notna(gli_n) else np.nan
        mret = monthly_rets[a].align(monthly_rets["GLI_INDEX"], join="inner")[0]
        vol_m=vol(mret); shp=sharpe(mret)
        base = monthly[a]/monthly[a].iloc[0]; mdd=maxdd(base); cal=calmar(c_full, mdd)
        rows.append({"Asset":a,"CAGR_full_%":100*c_full,f"CAGR_{years_n}Y_%":100*c_n,
                     "GLI_CAGR_full_%":100*gli_f,f"GLI_CAGR_{years_n}Y_%":100*gli_n,
                     "LiquidityAdj_CAGR_full_%":100*liq_f,f"LiquidityAdj_CAGR_{years_n}Y_%":100*liq_n,
                     "AnnVol_%(monthly)":100*vol_m,"Sharpe(monthly)":shp,"MaxDD_%":100*mdd,"Calmar":cal})
    metrics = pd.DataFrame(rows).round(2)
    corr = monthly_rets.corr().round(3)

    # OLS betas
    betas={}
    for a in [c for c in monthly.columns if c!="GLI_INDEX"]:
        y=monthly_rets[a].dropna(); x=monthly_rets["GLI_INDEX"].reindex(y.index).dropna()
        idx=y.index.intersection(x.index)
        if len(idx)>12:
            Y=y.loc[idx].values; X=add_constant(x.loc[idx].values)
            m=OLS(Y,X).fit(); betas[a]={"Beta_vs_GLI":m.params[1],"Alpha_%/mo":m.params[0],"R2":m.rsquared}
        else: betas[a]={"Beta_vs_GLI":np.nan,"Alpha_%/mo":np.nan,"R2":np.nan}
    return metrics, corr, pd.DataFrame(betas).T.round(3)

def roll_metrics(monthly, monthly_rets, window=12):
    assets=[c for c in monthly.columns if c!="GLI_INDEX"]; g=monthly_rets["GLI_INDEX"]
    rc={a: monthly_rets[a].rolling(window).corr(g) for a in assets}
    rb={}
    for a in assets:
        cov=monthly_rets[a].rolling(window,min_periods=window).cov(g)
        var=g.rolling(window,min_periods=window).var()
        rb[a]=cov/var
    ra={}
    rbdf=pd.DataFrame(rb)
    for a in assets:
        b=rbdf[a].reindex(monthly_rets.index)
        resid=monthly_rets[a]-b*g; ra[a]=resid.rolling(window,min_periods=window).mean()
    return pd.DataFrame(rc), rbdf, pd.DataFrame(ra)

def build_regime(monthly_gli):
    glim = monthly_gli.copy()
    yoy = glim.pct_change(12)*100
    reg = (yoy>0).rename("GLI_Expansion")
    return pd.concat([glim.rename("GLI_INDEX"), yoy.rename("GLI_%YoY"), reg], axis=1).dropna()

def event_study(monthly_rets, regime_df, horizons=(3,6,12)):
    reg = regime_df["GLI_Expansion"].astype(int); sw = reg.diff().fillna(0)
    up   = sw[sw==1].index; down = sw[sw==-1].index
    def calc(events):
        out={}
        for h in horizons:
            vals={}
            for c in monthly_rets.columns:
                if c=="GLI_INDEX": continue
                arr=[]
                for d in events:
                    win=monthly_rets.loc[d:].iloc[1:h+1][c]
                    if len(win)==h: arr.append((1+(win/100)).prod()-1)
                vals[c]=np.mean(arr)*100 if arr else np.nan
            out[f"{h}M_after"]=pd.Series(vals)
        return pd.DataFrame(out).round(2)
    return calc(up), calc(down)
