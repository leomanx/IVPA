# streamlit_app.py
# Inverse Vol / DRP Analyzer (Streamlit)
# - Upload portfolio (CSV/XLSX)
# - Pull prices via yfinance
# - Compute metrics (AnnVol, Sharpe, MaxDD, CAGR, Beta, IVP weights)
# - Trade Plan + execution constraints (lot/min-ticket/force-to-target cash)
# - Market state (FRED optional), VaR/CVaR, CPPI, Multi-benchmark
# - Visuals + Export Excel / ZIP figs / NAV CSV
#
# NOTE: Core logic adapted from your notebook/script and wrapped for Streamlit UI.

import os, io, re, math, time, json, warnings, zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests

# ---------- UI CONFIG ----------
st.set_page_config(page_title="Downside-Protection / IVP Analyzer", layout="wide")

# ---------- GLOBAL DEFAULTS ----------
SHEET_NAME     = "Portfolio"
SYMBOL_COL_D   = "Symbol"
WEIGHT_COL_D   = None                  # If None -> try Qty*Price or Equal
QTY_COL_D      = "Qty"
PRICE_COL_D    = "Mkt Price(USD)"
PRIMARY_BENCH  = "^GSPC"
MORE_BENCHES_D = "^GSPC,GLD,BTC-USD,^NDX"
PERIOD_D       = "5y"
RISK_FREE_PCTD = 4.0
TARGET_VOL_D   = 0.20
TARGET_BETA_D  = 1.00
RC_CAP_D       = 0.15
BAD_MAXDD_D    = -0.40
BAD_SHARPE_D   = 0.0
VAR_ALPHA_D    = 0.95
VAR_METHOD_D   = "hist"                # 'hist'|'normal'|'cornish'
INDEX_MULT     = 200.0
ETF_FALLBACK   = {"^GSPC":"SPY","SPY":"SPY","^NDX":"QQQ","QQQ":"QQQ","^IXIC":"QQQ"}

MIN_HIST_DAYS  = 250
METHOD_ALIGN   = "intersect"
RUN_TRACKER    = True
REBASE_NAV     = 100.0

# ---------- HELPERS ----------
def annualize_factor(freq='D'): return 252.0 if freq=='D' else 12.0

def max_drawdown(idx):
    s = pd.Series(idx).dropna()
    return float((s/s.cummax()-1).min()) if len(s) else np.nan

def sharpe(returns, rf_annual=0.0, freq='D'):
    r = pd.Series(returns).dropna()
    if r.empty: return np.nan
    k = annualize_factor(freq)
    mu = r.mean()*k; sd = r.std(ddof=0)*math.sqrt(k)
    return np.nan if sd==0 else (mu - rf_annual)/sd

def cagr(idx, freq='D'):
    s = pd.Series(idx).dropna().astype(float)
    s = s[s > 0]
    if s.size < 2: return np.nan
    years = (s.index[-1]-s.index[0]).days/365.25 if isinstance(s.index, pd.DatetimeIndex) else len(s)/252.0
    if years <= 0: return np.nan
    base = s.iloc[-1]/s.iloc[0]
    return np.nan if base<=0 else float(base**(1.0/years)-1.0)

def beta_vs(a,b):
    x = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    if len(x)<2: return np.nan
    cov = np.cov(x.iloc[:,0], x.iloc[:,1])[0,1]; var = np.var(x.iloc[:,1])
    return cov/var if var!=0 else np.nan

def clean_symbol(s):
    u = str(s).strip().upper()
    if u in {"TOTAL","TOTALS","SUBTOTAL","PORTFOLIO TOTAL","","NAN","NONE"}: return ""
    return u if re.match(r"^[A-Za-z0-9\.\-^]+$", u) else ""

def ivp_weights(ann_vol):
    inv = 1.0/ann_vol.replace(0, np.nan); w = inv/inv.sum()
    return w.fillna(0.0)

def normalize_pct(x, default):
    try: v = float(x)
    except: return default
    while v > 1.0: v /= 100.0
    return max(0.0, min(1.0, v))

# ---------- DATA IO ----------
@st.cache_data(show_spinner=False)
def read_portfolio(uploaded_file, sym_col, w_col, qty_col, px_col, sheet_name=SHEET_NAME):
    if uploaded_file is None:
        raise ValueError("กรุณาอัปโหลดพอร์ต (CSV/XLSX)")

    if uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    else:
        df = pd.read_csv(uploaded_file)

    if sym_col not in df.columns:
        raise ValueError(f"ไม่พบคอลัมน์ '{sym_col}' ในไฟล์ที่อัปโหลด")

    syms = df[sym_col].map(clean_symbol)
    df = df[syms!=""].copy(); syms = df[sym_col]

    if w_col and w_col in df.columns:
        w_raw = pd.to_numeric(df[w_col], errors="coerce").fillna(0.0)
    elif qty_col in df.columns and px_col in df.columns:
        w_raw = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0) * pd.to_numeric(df[px_col], errors="coerce").fillna(0.0)
    elif "Mkt Val" in df.columns:
        w_raw = pd.to_numeric(df["Mkt Val"], errors="coerce").fillna(0.0)
    else:
        w_raw = pd.Series(1.0, index=df.index)

    g = pd.DataFrame({"Symbol": syms.values, "w_raw": w_raw.values}).groupby("Symbol")["w_raw"].sum()
    g = g[g>0]
    if g.empty: raise ValueError("น้ำหนักรวมเป็นศูนย์")
    weights = g/g.sum()
    return g.index.tolist(), weights

@st.cache_data(show_spinner=True)
def yf_download(tickers, start=None, end=None, period="5y"):
    import yfinance as yf
    if not tickers: return pd.DataFrame()
    df = yf.download(" ".join(tickers), start=start, end=end,
                     period=None if (start or end) else period,
                     auto_adjust=True, progress=False, group_by="ticker", threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        cols=[]
        for t in tickers:
            if t not in df.columns.levels[0]: continue
            sub=df[t]; col="Close" if "Close" in sub.columns else ("Adj Close" if "Adj Close" in sub.columns else None)
            if col: cols.append(sub[col].rename(t))
        px = pd.concat(cols, axis=1) if cols else pd.DataFrame()
    else:
        px = df.copy()
        if "Close" in px.columns: px=px["Close"]
        elif "Adj Close" in px.columns: px=px["Adj Close"]
        if isinstance(px, pd.Series): px = px.to_frame(tickers[0])
    return px.dropna(how="all")

@st.cache_data(show_spinner=False)
def latest_prices(tickers, period="1mo"):
    px = yf_download(tickers, period=period)
    return pd.Series({t: float(px[t].dropna().iloc[-1]) for t in tickers if t in px.columns})

# ---------- METRICS / ANALYTICS ----------
def align_returns(px, method):
    lr = np.log(px/px.shift(1))
    return (lr.dropna(how='any') if method=='intersect' else lr.dropna(how='all'))

def compute_metrics(px_assets, bench_price, file_weights, rf_pct):
    r_all = align_returns(px_assets.join(bench_price), METHOD_ALIGN)
    r_assets = r_all[px_assets.columns]
    r_bench  = r_all[bench_price.name]
    k = annualize_factor('D')

    ann_vol = r_assets.std(ddof=0)*math.sqrt(k)
    betas   = pd.Series({t: beta_vs(r_assets[t], r_bench) for t in r_assets.columns}, name="Beta")
    ivp_w   = ivp_weights(ann_vol).rename("IVP_Weight")
    cov_an  = r_assets.cov(min_periods=30, ddof=0)*k

    w_file = file_weights.reindex(r_assets.columns).fillna(0.0)
    w_file = w_file/(w_file.sum() if w_file.sum()!=0 else 1.0); w_file.name="File_Weight"

    # risk contribution
    def portfolio_vol(weights, cov_an):
        if cov_an.empty or weights.empty: return np.nan
        w = weights.values.reshape(-1,1)
        return float(np.sqrt(np.clip((w.T@cov_an.values@w)[0,0],0,np.inf)))

    def risk_contrib(weights, cov_an):
        if cov_an.empty: return pd.Series(index=weights.index, dtype=float)
        w = weights.values.reshape(-1,1); covw = cov_an.values@w
        sig = portfolio_vol(weights, cov_an); denom = sig if sig and not np.isnan(sig) else 1.0
        return pd.Series((weights.values*covw.flatten())/denom, index=weights.index)

    rc_ivp  = risk_contrib(ivp_w,  cov_an).rename("RC_IVP")
    rc_file = risk_contrib(w_file, cov_an).rename("RC_File")

    idx_assets    = (r_assets.fillna(0).add(1)).cumprod()
    cagr_assets   = idx_assets.apply(cagr).rename("CAGR")
    mdd_assets    = idx_assets.apply(max_drawdown).rename("MaxDD")
    sharpe_assets = r_assets.apply(lambda s: sharpe(s, rf_annual=rf_pct/100.0)).rename("Sharpe")

    metrics = pd.concat([ann_vol.rename("AnnVol"), betas, ivp_w, w_file,
                         rc_ivp, rc_file, cagr_assets, mdd_assets, sharpe_assets], axis=1)

    return metrics, r_assets, r_bench, (r_bench.add(1).cumprod()), cov_an, ivp_w, w_file

def var_cvar(r, alpha=0.95, method="hist"):
    x = pd.Series(r).dropna().values
    if x.size == 0: return np.nan, np.nan
    if method == "hist":
        q = np.quantile(x, 1-alpha, method="lower")
        var = -q
        cvar = -x[x <= q].mean() if (x <= q).any() else np.nan
    elif method == "normal":
        mu, sd = x.mean(), x.std(ddof=0)
        from math import erf, sqrt
        # z for alpha tail:
        import mpmath as mp
        z = float(mp.erfcinv(2*(1-alpha))*math.sqrt(2))  # approx
        var = -(mu + sd*z)
        cvar = -(mu - sd * (math.exp(-0.5*z*z)/math.sqrt(2*math.pi))/(1-alpha))
    elif method == "cornish":
        mu  = x.mean(); sd = x.std(ddof=0)
        if sd==0: return np.nan, np.nan
        s   = ((x - mu)/sd).mean()**3
        k   = ((x - mu)/sd).mean()**4 - 3
        # Use standard normal ppf from numpy
        from scipy.stats import norm
        z   = norm.ppf(alpha)
        zcf = (z + (s/6)*(z**2-1) + (k/24)*(z**3-3*z) - (s**2/36)*(2*z**3-5*z))
        var = -(mu + sd*zcf)
        cvar = -(mu - sd * (math.exp(-0.5*zcf*zcf)/math.sqrt(2*math.pi))/(1-alpha))
    else:
        raise ValueError("method must be 'hist'|'normal'|'cornish'")
    return float(var), float(cvar)

def vol_scaler(rP, target_ann_vol=0.20, lookback=20, clip=(0.5, 1.5)):
    rP = pd.Series(rP)
    sig = rP.rolling(lookback).std(ddof=0) * np.sqrt(252)
    scale = (target_ann_vol / sig).clip(lower=clip[0], upper=clip[1])
    return scale.fillna(1.0)

def build_trade_plan(metrics, rc_cap=0.15, forbid_bad_buy=True, bad_mdd=-0.40, bad_sharpe=0.0):
    df = metrics.copy()
    tgt = df["File_Weight"].copy()

    over = df["RC_File"] > rc_cap
    scale = (rc_cap/df["RC_File"]).clip(upper=1.0)
    tgt.loc[over] = (tgt.loc[over] * scale.loc[over]).values

    bad = (df["Sharpe"]<bad_sharpe) | (df["MaxDD"]<bad_mdd)
    tgt.loc[bad] = tgt.loc[bad]*0.5

    good = (df["Beta"]<0.6) & (df["CAGR"]>0)
    tgt.loc[good] = tgt.loc[good]*1.2

    tgt = tgt + 0.25*(df["IVP_Weight"] - tgt)

    if forbid_bad_buy:
        tgt.loc[bad] = np.minimum(tgt.loc[bad], df.loc[bad,"File_Weight"])

    tgt = tgt.clip(lower=0).fillna(0.0); tgt = tgt/(tgt.sum() if tgt.sum()!=0 else 1.0)

    delta  = tgt - df["File_Weight"]
    action = np.where(delta>0.002, "BUY", np.where(delta<-0.002, "SELL", "KEEP"))
    reason=[]
    for t in df.index:
        rs=[]
        if t in over.index and over.loc[t]: rs.append(f"RC>{int(rc_cap*100)}%")
        if t in bad.index and bad.loc[t]:  rs.append("Sharpe<0/DeepDD")
        if t in good.index and good.loc[t]: rs.append("Diversifier")
        if not rs: rs.append("Rebalance→IVP")
        reason.append(", ".join(rs))

    return pd.DataFrame({"TargetWeight":tgt, "Delta":delta, "Action":action, "Reason":reason}, index=df.index)

def apply_trade_constraints(tp, last_px, pv, cash_now_pct, cash_tgt_pct, min_ticket, lot_func):
    tp = tp.copy()
    tp["Price"] = pd.Series(last_px).reindex(tp.index).astype(float)
    tp["TradeValue_raw"] = tp["Delta"] * pv

    cash_now_pct = max(0.0, min(1.0, float(cash_now_pct)))
    cash_tgt_pct = max(0.0, min(1.0, float(cash_tgt_pct)))
    cash_now  = cash_now_pct * pv
    cash_tgt  = cash_tgt_pct * pv

    net_buys_raw  = tp.loc[tp["TradeValue_raw"]>0,"TradeValue_raw"].sum()
    net_sells_raw = -tp.loc[tp["TradeValue_raw"]<0,"TradeValue_raw"].sum()
    budget        = net_sells_raw + max(0.0, cash_now - cash_tgt)

    scale = 1.0
    if net_buys_raw > budget and net_buys_raw > 0:
        scale = budget/net_buys_raw
        tp.loc[tp["TradeValue_raw"]>0,"TradeValue_raw"] *= scale

    def lot_size_for_symbol(sym: str) -> int:
        return 100 if sym.upper().endswith(".BK") else 1

    lot = tp.index.to_series().map(lot_func or lot_size_for_symbol).astype(int).replace(0,1)
    shares_round=[]; tv_round=[]; note=[]
    for sym, raw, act, px in zip(tp.index, tp["TradeValue_raw"], tp["Action"], tp["Price"]):
        if np.isnan(px) or px<=0:
            shares_round.append(np.nan); tv_round.append(0.0); note.append("no_price"); continue
        L = lot.loc[sym]; sh = raw/px
        if act=="BUY":   sh_r = math.floor(max(sh,0)/L)*L
        elif act=="SELL": sh_r = -math.floor(abs(min(sh,0))/L)*L
        else:            sh_r = 0
        tv_r = sh_r * px
        if abs(tv_r) < min_ticket and sh_r!=0:
            sh_r, tv_r = 0, 0.0; note.append("below_min_ticket")
        else:
            note.append("")
        shares_round.append(sh_r); tv_round.append(tv_r)

    tp["Lot"]               = lot.values
    tp["Shares"]            = pd.Series(shares_round, index=tp.index).astype("Int64")
    tp["TradeValueRounded"] = pd.Series(tv_round, index=tp.index).astype(float)
    tp["Notes"]             = note

    buys_exec  = tp.loc[tp["TradeValueRounded"]>0,"TradeValueRounded"].sum()
    sells_exec = -tp.loc[tp["TradeValueRounded"]<0,"TradeValueRounded"].sum()
    cash_after = cash_now + sells_exec - buys_exec
    cash_after_pct = cash_after / pv if pv>0 else np.nan

    summary = {
        "PV": pv,
        "Cash_now": cash_now,
        "Cash_target": cash_tgt,
        "Budget_buys": budget,
        "Net_buys_raw": net_buys_raw,
        "Net_sells_raw": net_sells_raw,
        "Net_buys_exec": buys_exec,
        "Net_sells_exec": sells_exec,
        "Budget_buys_exec": sells_exec + max(0.0, cash_now - cash_tgt),
        "Cash_after": cash_after,
        "Cash_after_pct": cash_after_pct,
        "Cash_gap_to_target": cash_after_pct - cash_tgt_pct,
        "Scaled_buys_factor": scale
    }
    return tp, summary

# ---------- MARKET STATE (FRED) ----------
def _detect_fred_key(key, env_default="FRED_API_KEY"):
    key = (key or os.environ.get(env_default, "")).strip().strip('"').strip("'")
    return key

@st.cache_data(show_spinner=False)
def fred_fetch(series, start="2005-01-01", key=None, retries=2, timeout=10):
    key = _detect_fred_key(key)
    if not key:
        return pd.Series(dtype=float, name=series), {"ok":False,"reason":"no_api_key"}
    url = "https://api.stlouisfed.org/fred/series/observations"
    p = {"series_id":series,"observation_start":start,"file_type":"json","api_key":key}
    last_err=None
    for _ in range(retries+1):
        try:
            resp = requests.get(url, params=p, timeout=timeout)
            if resp.status_code!=200:
                last_err=f"http_{resp.status_code}:{resp.text[:120]}"
                time.sleep(0.4); continue
            js = resp.json().get("observations",[])
            data = pd.Series({
                pd.to_datetime(i["date"]):(float(i["value"]) if i["value"]!="." else np.nan)
                for i in js
            }).sort_index()
            return data.rename(series), {"ok":True,"reason":"ok"}
        except Exception as e:
            last_err=str(e); time.sleep(0.3)
    return pd.Series(dtype=float, name=series), {"ok":False,"reason":last_err or "unknown"}

def fred_summary(key=None):
    out, meta = {}, {}
    for sid in ["BAMLH0A0HYM2","T10Y2Y","DGS2","FEDFUNDS","NAPM","PCEPILFE","SAHMREALTIME"]:
        s, m = fred_fetch(sid, key=key); out[sid]=s; meta[sid]=m
    df = pd.concat(out.values(), axis=1)

    core_yoy = df["PCEPILFE"].pct_change(12)*100 if "PCEPILFE" in df else pd.Series(dtype=float)
    res = []
    for sid in ["BAMLH0A0HYM2","T10Y2Y","DGS2","FEDFUNDS","NAPM","SAHMREALTIME"]:
        s = df[sid] if sid in df else pd.Series(dtype=float)
        last = s.dropna().iloc[-1] if s.dropna().size else np.nan
        last_dt = s.dropna().index[-1].date().isoformat() if s.dropna().size else ""
        res.append({"Series":sid,"Last":last,"LastDate":last_dt,"Note":("" if meta.get(sid,{}).get("ok") else f"err:{meta.get(sid,{}).get('reason')}")})
    last = core_yoy.dropna().iloc[-1] if core_yoy.dropna().size else np.nan
    last_dt = core_yoy.dropna().index[-1].date().isoformat() if core_yoy.dropna().size else ""
    res.append({"Series":"CorePCE_YoY","Last":last,"LastDate":last_dt,"Note":("" if meta.get("PCEPILFE",{}).get("ok") else f"err:{meta.get('PCEPILFE',{}).get('reason')}" )})

    return pd.DataFrame(res), df, core_yoy

def market_state(primary_bench_price, fred_key=None):
    bench_price = pd.Series(primary_bench_price).dropna()
    ma200 = bench_price.rolling(200).mean()
    above200 = 1.0 if (len(ma200.dropna()) and bench_price.iloc[-1]>ma200.iloc[-1]) else 0.0
    slope200 = 0.0
    if len(ma200.dropna())>22 and ma200.iloc[-1]>0:
        slope200 = float((ma200.iloc[-1]-ma200.shift(20).iloc[-1])/ma200.iloc[-1])

    # breadth (ETF 5 ชุด)
    try:
        etfs = ["SPY","QQQ","IWM","EFA","EEM"]
        px_etf = yf_download(etfs, period="3y")
        breadth = (px_etf.apply(lambda s: s.iloc[-1] > s.rolling(200).mean().iloc[-1])).mean()
    except Exception:
        breadth = np.nan
    trend_score = np.nanmean([above200, np.tanh(10*slope200), breadth])

    # VIX rank
    try:
        vix = yf_download(["^VIX"], period="3y")["^VIX"].dropna()
        vix_pct = float(vix.rank(pct=True).iloc[-1])
    except Exception:
        vix_pct = np.nan

    fred_sum, fred_raw, core_yoy = fred_summary(key=fred_key)

    credit_risk = np.nan
    if "BAMLH0A0HYM2" in fred_raw:
        ser = fred_raw["BAMLH0A0HYM2"].dropna()
        cutoff = ser.index.max() - pd.DateOffset(years=3) if isinstance(ser.index, pd.DatetimeIndex) else None
        hy = ser.loc[ser.index >= cutoff] if cutoff is not None else ser.tail(min(len(ser), 3*252))
        if hy.size > 10 and hy.std() > 0:
            z_hy = float((hy.iloc[-1] - hy.mean()) / hy.std())
            credit_risk = 1.0 / (1.0 + np.exp(-z_hy))

    curve = float(fred_raw["T10Y2Y"].dropna().iloc[-1]) if "T10Y2Y" in fred_raw and fred_raw["T10Y2Y"].dropna().size else np.nan
    curve_risk = 1.0 if (not np.isnan(curve) and curve<0) else 0.0

    pmi_score = np.nan
    if "NAPM" in fred_raw and fred_raw["NAPM"].dropna().size:
        pmi = float(fred_raw["NAPM"].dropna().iloc[-1]); pmi_score = min(max((pmi-50)/10, -1), 1)/2 + 0.5

    infl_risk = np.nan
    if core_yoy.dropna().size:
        corey = float(core_yoy.dropna().iloc[-1]); infl_risk = 1/(1+np.exp(-(corey-2.5)))

    trend_final = np.nanmean([trend_score, pmi_score])
    risk_score  = np.nanmean([vix_pct, credit_risk, curve_risk, infl_risk])

    if trend_final>=0.70 and (not np.isnan(risk_score) and risk_score<=0.30): state="Risk-On Strong"
    elif trend_final>=0.55 and (np.isnan(risk_score) or risk_score<=0.45):    state="Risk-On"
    elif (not np.isnan(risk_score) and risk_score>=0.60) and (not np.isnan(trend_final) and trend_final<=0.45): state="Risk-Off"
    else: state="Mixed"

    detail = {
        "trend_score": round(float(trend_final),3) if not np.isnan(trend_final) else np.nan,
        "risk_score":  round(float(risk_score),3)  if not np.isnan(risk_score)  else np.nan,
        "breadth_5ETF_%>200D": round(float(breadth),3) if not np.isnan(breadth) else np.nan,
        "vix_pctile_3y": round(float(vix_pct),3) if not np.isnan(vix_pct) else np.nan,
        "hy_spread_risk": round(float(credit_risk),3) if not np.isnan(credit_risk) else np.nan,
        "curve_10y2y": curve,
        "pmi_score": round(float(pmi_score),3) if not np.isnan(pmi_score) else np.nan,
        "infl_risk_corePCE": round(float(infl_risk),3) if not np.isnan(infl_risk) else np.nan,
    }
    return state, detail, fred_sum

def hedge_advice(state, betaP, beta_target, pv, bench_price, primary_bench, multiplier=INDEX_MULT):
    hedge_etf = ETF_FALLBACK.get(primary_bench, "SPY")
    try:
        etf_px = float(yf_download([hedge_etf], period="1mo")[hedge_etf].dropna().iloc[-1])
    except Exception:
        etf_px = np.nan
    index_lvl = float(pd.Series(bench_price).dropna().iloc[-1])
    delta_beta = max((betaP or 0) - beta_target, 0.0)
    notional   = delta_beta * pv
    futs = math.ceil(notional / (index_lvl * multiplier)) if index_lvl and multiplier>0 else np.nan
    shares = math.ceil(notional / etf_px) if etf_px and etf_px>0 else np.nan
    if state=="Risk-Off": advice="Active hedge to target β"
    elif state=="Mixed":  advice="Partial hedge 10–20% หรือ hedge เฉพาะ over-beta"
    else:                 advice="Monitor; hedge เฉพาะกรณี β>β*+0.2 หรือ event risk"
    return {"Ticker":hedge_etf,"HedgeNotional":notional,"NotionalPct":(notional/pv if pv else np.nan),
            "FuturesContracts":futs,"ETF_Shares":shares,"Advice":advice}

# ---------- VISUALS ----------
def plot_cum_vs_bench(idx_port, bench_prices):
    fig=plt.figure()
    idx_port=pd.Series(idx_port).dropna()
    (idx_port/idx_port.iloc[0]).rename("Portfolio").plot()
    for name, px in bench_prices.items():
        s=pd.Series(px).dropna()
        if s.empty: continue
        (s/s.iloc[0]).rename(name).plot()
    plt.title("Cumulative Index: Portfolio vs Benchmarks")
    plt.xlabel("Date"); plt.ylabel("Index (rebased=1.0)")
    plt.grid(True, alpha=0.3); plt.legend(ncols=2, fontsize=9)
    return fig

def plot_drawdown(idx_port):
    fig=plt.figure()
    s=pd.Series(idx_port).dropna()
    (s/s.cummax()-1).rename("Drawdown").plot()
    plt.title("Portfolio Drawdown"); plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    return fig

def plot_rc_bar(metrics, col="RC_File", title=None):
    if col not in metrics.columns: return None
    fig=plt.figure()
    metrics[col].sort_values(ascending=False).plot(kind="bar")
    plt.title(title or f"Risk Contribution — {col}")
    plt.xlabel("Symbol"); plt.ylabel("Contribution")
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    return fig

def plot_risk_return_bubble(metrics):
    df=metrics[["AnnVol","CAGR","File_Weight"]].dropna()
    if df.empty: return None
    fig=plt.figure()
    x=df["AnnVol"].values; y=df["CAGR"].values; sz=(df["File_Weight"].values*1500)+10
    plt.scatter(x,y,s=sz,alpha=0.6)
    for sym,xv,yv in zip(df.index,x,y): plt.annotate(sym,(xv,yv))
    plt.title("Risk–Return Map (bubble = current weight)"); plt.xlabel("AnnVol"); plt.ylabel("CAGR")
    plt.grid(True, alpha=0.3)
    return fig

def to_excel_bytes(sheets: dict):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if df is None: continue
            if isinstance(df, (pd.Series, pd.Index)): df = df.to_frame()
            df.to_excel(xw, sheet_name=name[:31], index=True)
    bio.seek(0)
    return bio

def figs_to_zip(figs: dict):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figs.items():
            if fig is None: continue
            png = io.BytesIO()
            fig.savefig(png, dpi=160, bbox_inches="tight")
            png.seek(0)
            zf.writestr(f"{name}.png", png.read())
    bio.seek(0)
    return bio

# ---------- UI: SIDEBAR ----------
st.sidebar.title("⚙️ Settings")
uploaded = st.sidebar.file_uploader("Upload Portfolio (CSV/XLSX)", type=["csv","xlsx"])
sheet_name = st.sidebar.text_input("Sheet name (XLSX)", SHEET_NAME)
sym_col   = st.sidebar.text_input("Symbol column", SYMBOL_COL_D)
weight_col= st.sidebar.text_input("Weight column (optional)", "" if WEIGHT_COL_D is None else WEIGHT_COL_D)
qty_col   = st.sidebar.text_input("Qty column (fallback)", QTY_COL_D)
price_col = st.sidebar.text_input("Price column (fallback)", PRICE_COL_D)

st.sidebar.markdown("---")
period   = st.sidebar.selectbox("Price history period", ["1y","3y","5y","10y","max"], index=2)
more_benches = st.sidebar.text_input("Benchmarks (comma)", MORE_BENCHES_D)
primary_bench= st.sidebar.text_input("Primary Bench (β/hedge)", PRIMARY_BENCH)

st.sidebar.markdown("---")
pv        = st.sidebar.number_input("Portfolio Value (USD)", min_value=1000.0, value=30000.0, step=1000.0)
cash_now  = st.sidebar.text_input("Current Cash %", "10")
cash_tgt  = st.sidebar.text_input("Target  Cash %", "15")
min_ticket= st.sidebar.number_input("Min ticket per order", min_value=0.0, value=500.0, step=100.0)

st.sidebar.markdown("---")
rf_pct    = st.sidebar.number_input("Risk-free %/y", min_value=0.0, value=RISK_FREE_PCTD, step=0.25)
target_vol= st.sidebar.number_input("Target ann. vol", min_value=0.05, value=TARGET_VOL_D, step=0.01, format="%.2f")
target_beta=st.sidebar.number_input("Target β", min_value=0.0, value=TARGET_BETA_D, step=0.05)
rc_cap    = st.sidebar.number_input("RC cap (share)", min_value=0.05, value=RC_CAP_D, step=0.01)
bad_mdd   = st.sidebar.number_input("Bad MaxDD (≤)", value=BAD_MAXDD_D, step=0.05, format="%.2f")
bad_shp   = st.sidebar.number_input("Bad Sharpe (≤)", value=BAD_SHARPE_D, step=0.1, format="%.2f")

st.sidebar.markdown("---")
var_alpha = st.sidebar.slider("VaR/CVaR α", 0.80, 0.995, VAR_ALPHA_D, 0.005)
var_method= st.sidebar.selectbox("VaR method", ["hist","normal","cornish"], index=["hist","normal","cornish"].index(VAR_METHOD_D))

st.sidebar.markdown("---")
fred_key  = st.sidebar.text_input("FRED API Key (optional)", value=os.environ.get("FRED_API_KEY",""))

run_btn = st.sidebar.button("▶️ Run Analysis", use_container_width=True)

st.title("Downside-Protection / IVP Portfolio Analyzer (Streamlit)")
st.caption("Upload → Analyze → Visualize → Export")

# ---------- MAIN FLOW ----------
if run_btn:
    try:
        with st.spinner("Loading portfolio..."):
            tickers, w_file = read_portfolio(uploaded, sym_col, (weight_col or None), qty_col, price_col, sheet_name)

        st.success(f"Loaded {len(tickers)} symbols")
        st.write(pd.DataFrame({"Weight":w_file}).rename_axis("Symbol"))

        benches = [s.strip() for s in more_benches.split(",") if s.strip()]
        all_syms = list(dict.fromkeys(tickers + ([primary_bench] if primary_bench not in benches else []) + benches))

        with st.spinner("Downloading prices..."):
            px_all = yf_download(all_syms, period=period)
        if primary_bench not in px_all.columns:
            st.error(f"ไม่พบ {primary_bench} ในข้อมูลราคา"); st.stop()

        # filter missing
        miss = [t for t in tickers if t not in px_all.columns]
        if miss:
            st.warning(f"ตัดออก (ไม่มีราคา): {', '.join(miss)}")
            tickers = [t for t in tickers if t in px_all.columns]
            w_file  = w_file.reindex(tickers).fillna(0.0)
            w_file  = w_file/(w_file.sum() if w_file.sum()!=0 else 1.0)

        px_assets = px_all[tickers]
        primary_price = px_all[primary_bench].rename(primary_bench)

        # metrics
        metrics, r_assets, r_bench, idx_bench, cov_an, ivp_w, w_file = compute_metrics(px_assets, primary_price, w_file, rf_pct)
        idx_file = (r_assets.mul(w_file, axis=1).sum(axis=1)).add(1).cumprod()
        rP = idx_file.pct_change().dropna()
        betaP = beta_vs(rP, r_bench.loc[rP.index]) if len(rP) else np.nan

        VaR, CVaR = var_cvar(rP, alpha=var_alpha, method=var_method)
        scaler = vol_scaler(rP, target_ann_vol=target_vol, lookback=20, clip=(0.5,1.5))

        tp = build_trade_plan(metrics, rc_cap=rc_cap, forbid_bad_buy=True, bad_mdd=bad_mdd, bad_sharpe=bad_shp)
        last_px = latest_prices(tp.index.tolist())
        tp_exec, exec_summary = apply_trade_constraints(
            tp, last_px, pv,
            normalize_pct(cash_now, 0.10),
            normalize_pct(cash_tgt, 0.15),
            min_ticket,
            lot_func=None
        )

        # market state + hedge
        state, state_detail, fred_sum = market_state(primary_price, fred_key if fred_key else None)
        hedge = hedge_advice(state, betaP, target_beta, pv, primary_price, primary_bench, INDEX_MULT)

        # multi-benchmark
        bench_prices = {b:px_all[b] for b in benches if b in px_all.columns}

        # NAV tracker (in-memory)
        nav_series = (pd.Series(idx_file)/pd.Series(idx_file).dropna().iloc[0])*REBASE_NAV
        nav_df = nav_series.rename("NAV").to_frame()
        nav_df.index.name = "Date"

        # ---------- LAYOUT ----------
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics & Plan","📈 Charts","🧭 Market & Risk","📦 Export"])

        with tab1:
            st.subheader("Asset Metrics")
            st.dataframe(metrics.round(4), use_container_width=True)

            st.subheader("Trade Plan (Executed Constraints)")
            st.dataframe(tp_exec.round(4), use_container_width=True)

            st.caption(f"Exec Summary: { {k:(round(v,4) if isinstance(v,(int,float)) and not pd.isna(v) else v) for k,v in exec_summary.items()} }")

        with tab2:
            st.subheader("Portfolio vs Benchmarks")
            fig1 = plot_cum_vs_bench(idx_file, bench_prices)
            st.pyplot(fig1, use_container_width=True)

            st.subheader("Drawdown")
            fig2 = plot_drawdown(idx_file)
            st.pyplot(fig2, use_container_width=True)

            st.subheader("Risk Contribution (current weights)")
            fig3 = plot_rc_bar(metrics, "RC_File", "Risk Contribution — File Weights")
            if fig3: st.pyplot(fig3, use_container_width=True)

            st.subheader("Risk–Return Bubble")
            fig4 = plot_risk_return_bubble(metrics)
            if fig4: st.pyplot(fig4, use_container_width=True)

        with tab3:
            st.subheader("Market State")
            st.json({"State":state, **state_detail})
            if fred_key:
                st.write("FRED summary (last values)")
                st.dataframe(fred_sum, use_container_width=True)
            st.subheader("Hedge Advice")
            st.json({"Beta_P": betaP, "Target_Beta": target_beta, **hedge})

            st.subheader("VaR / CVaR")
            st.json({"alpha":var_alpha, "method":var_method, "VaR":VaR, "CVaR":CVaR})

        with tab4:
            st.subheader("Download files")
            # Excel
            sheets = {
                "Inputs": pd.DataFrame({"Param":["PrimaryBench","RF(%)","TargetVol","TargetBeta","RC_cap","Period","VaR_method","VaR_alpha"],
                                        "Value":[primary_bench,rf_pct,target_vol,target_beta,rc_cap,period,var_method,var_alpha]}).set_index("Param"),
                "Asset_Metrics": metrics,
                "TradePlan_Targets": tp,
                "TradePlan_Exec": tp_exec,
                "Exec_Summary": pd.DataFrame([exec_summary]).T.rename(columns={0:"Value"}),
                "MarketState": pd.DataFrame([{"State":state, **state_detail}]).T.rename(columns={0:"Value"}),
                "Hedge": pd.DataFrame([hedge]).T.rename(columns={0:"Value"}),
                "Cumulative_Index": pd.Series(idx_file, name="Port_File").to_frame(),
                "VaR_CVaR": pd.DataFrame([{"alpha":var_alpha,"method":var_method,"VaR":VaR,"CVaR":CVaR}]).T.rename(columns={0:"Value"}),
                "Vol_Scaler": vol_scaler(rP).rename("Vol_Scale").to_frame(),
                "NAV_Tracker": nav_df.reset_index()
            }
            excel_bytes = to_excel_bytes(sheets)
            st.download_button("⬇️ Download Excel Report", data=excel_bytes.getvalue(),
                               file_name=f"DRP_Full_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ZIP figs
            figs_zip = figs_to_zip({
                "cum_port_vs_bench": fig1,
                "drawdown_port": fig2,
                "rc_file": fig3,
                "risk_return_bubble": fig4
            })
            st.download_button("⬇️ Download Charts (ZIP)", data=figs_zip.getvalue(),
                               file_name=f"figs_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                               mime="application/zip")

            # NAV CSV
            nav_csv = nav_df.to_csv(index=True).encode()
            st.download_button("⬇️ Download NAV CSV", data=nav_csv,
                               file_name=f"NAV_history_{datetime.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv")

        st.success("Finished ✅")

    except Exception as e:
        st.exception(e)
