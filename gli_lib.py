# gli_lib.py
import os, math
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# ---------- Public knobs (ค่าเริ่มต้น) ----------
DEFAULT_START = "2008-01-01"
DEFAULT_END   = None
PER_YEAR_M = 12

# ---------- Utils ----------
def get_fred(key: str | None):
    key = (key or "").strip().strip('"').strip("'")
    if not key:
        raise ValueError("FRED_API_KEY ไม่ถูกต้องหรือว่าง")
    return Fred(api_key=key)

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["|".join([str(x) for x in tup if x not in (None,"")]) for tup in df.columns.to_list()]
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df

def rebase(s: pd.Series) -> pd.Series:
    s = pd.Series(s).dropna()
    return 100.0 * s / s.iloc[0] if len(s) else s

def _years_span(idx):
    if hasattr(idx[0], "to_timestamp"):
        start = idx[0].to_timestamp(); end = idx[-1].to_timestamp()
        return (end - start).days / 365.25
    if hasattr(idx[0], "year"):
        start, end = pd.to_datetime(idx[0]), pd.to_datetime(idx[-1])
        return (end - start).days / 365.25
    return float(idx[-1] - idx[0])

def cagr_from_series(series):
    s = series.dropna()
    if len(s) < 2: return np.nan
    yrs = _years_span(s.index)
    return (s.iloc[-1]/s.iloc[0])**(1.0/yrs) - 1.0

def cagr_last_n_years(series, n):
    s = series.dropna()
    if len(s) < 2: return np.nan
    if not isinstance(s.index, pd.DatetimeIndex): s.index = pd.to_datetime(s.index)
    cutoff = s.index[-1] - pd.DateOffset(years=n)
    s = s[s.index >= cutoff]
    if len(s) < 2: return np.nan
    yrs = _years_span(s.index)
    return (s.iloc[-1]/s.iloc[0])**(1.0/yrs) - 1.0

def ann_vol_from_returns(ret_series, periods_per_year=PER_YEAR_M):
    r = ret_series.dropna() / 100.0
    return r.std() * np.sqrt(periods_per_year)

def sharpe(ret_series, rf_annual, periods_per_year=PER_YEAR_M):
    r = ret_series.dropna() / 100.0
    rf = rf_annual / periods_per_year
    ex = r - rf
    sd = r.std()
    return np.nan if sd == 0 else (ex.mean() / sd) * np.sqrt(periods_per_year)

def sortino(ret_series, rf_annual, periods_per_year=PER_YEAR_M):
    r = ret_series.dropna() / 100.0
    rf = rf_annual / periods_per_year
    ex = r - rf
    downside = ex[ex < 0]
    dd = downside.std()
    return np.nan if dd == 0 else (ex.mean() / dd) * np.sqrt(periods_per_year)

def max_drawdown(series):
    s = pd.Series(series).dropna().astype(float)
    peak = s.cummax()
    dd = s/peak - 1.0
    return float(dd.min()) if len(dd) else np.nan

# ---------- 1) GLI proxy ----------
_SERIES = {
    "FED_WALCL":  "WALCL",
    "ECB_ASSETS": "ECBASSETSW",
    "BOJ_ASSETS": "JPNASSETS",
    "TGA":        "WTREGEN",
    "ONRRP":      "RRPONTSYD",
    "USDJPY":     "DEXJPUS",
    "USDEUR":     "DEXUSEU",
}

def _fred_series(fred: Fred, sid, start=DEFAULT_START, end=DEFAULT_END):
    s = fred.get_series(sid)
    s.index = pd.to_datetime(s.index)
    if start: s = s[s.index >= pd.to_datetime(start)]
    if end:   s = s[s.index <= pd.to_datetime(end)]
    return s.sort_index()

def _to_weekly(s): return s.resample("W-FRI").last()

def build_gli_proxy(fred: Fred, start=DEFAULT_START, end=DEFAULT_END,
                    include_pboc=False, pboc_series=None):
    raw = {}
    for k, sid in _SERIES.items():
        raw[k] = _fred_series(fred, sid, start, end)

    if include_pboc and pboc_series:
        try:
            raw["PBOC_USD"] = _fred_series(fred, pboc_series, start, end)
        except Exception:
            pass

    wk = pd.DataFrame({
        "FED_WALCL":  _to_weekly(raw["FED_WALCL"]),
        "ECB_ASSETS": _to_weekly(raw["ECB_ASSETS"]),
        "BOJ_ASSETS": _to_weekly(raw["BOJ_ASSETS"]),
        "TGA":        _to_weekly(raw["TGA"]),
        "ONRRP":      _to_weekly(raw["ONRRP"]),
        "USDJPY":     _to_weekly(raw["USDJPY"]),
        "USDEUR":     _to_weekly(raw["USDEUR"]),
    }).dropna(how="any")

    wk["ECB_USD"] = wk["ECB_ASSETS"] * wk["USDEUR"]
    wk["BOJ_USD"] = wk["BOJ_ASSETS"] / wk["USDJPY"].replace(0, np.nan)
    wk["PBOC_USD"] = 0.0
    if include_pboc and "PBOC_USD" in raw:
        wk["PBOC_USD"] = _to_weekly(raw["PBOC_USD"]).reindex(wk.index).interpolate()

    wk["GLI_USD"]   = wk["FED_WALCL"] + wk["ECB_USD"] + wk["BOJ_USD"] + wk["PBOC_USD"] - wk["TGA"] - wk["ONRRP"]
    wk["GLI_INDEX"] = 100 * wk["GLI_USD"] / wk["GLI_USD"].iloc[0]
    return wk

# ---------- 2) Fetch assets ----------
_REQ = {
    "NASDAQ": {"fred": "NASDAQCOM", "yahoo": "^IXIC"},
    "SP500":  {"fred": "SP500",     "yahoo": "^GSPC"},
    "GOLD":   {"fred": None,        "yahoo": "GC=F"},
    "BTC":    {"fred": "CBBTCUSD",  "yahoo": "BTC-USD"},
    "ETH":    {"fred": None,        "yahoo": "ETH-USD"},
}

def _fred_daily(fred: Fred, sid, start=DEFAULT_START, end=DEFAULT_END):
    s = _fred_series(fred, sid, start, end)
    return s.sort_index().asfreq("D").ffill()

def _yf_close(ticker, start=DEFAULT_START, end=DEFAULT_END):
    df_y = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df_y is None or df_y.empty or "Close" not in df_y: return pd.Series(dtype=float)
    s = df_y["Close"].copy(); s.index = pd.to_datetime(s.index)
    return s.sort_index()

def fetch_assets(fred: Fred, start=DEFAULT_START, end=DEFAULT_END) -> pd.DataFrame:
    assets_daily, ok, fail = {}, [], []
    for name, src in _REQ.items():
        ser = pd.Series(dtype=float)
        if src["fred"]:
            try:
                ser = _fred_daily(fred, src["fred"], start, end)
                if len(ser.dropna()) > 1:
                    assets_daily[name] = ser; ok.append(f"{name}: FRED ({src['fred']})"); continue
            except Exception as e:
                fail.append(f"{name}: FRED fail ({src['fred']}) -> {e}")
        try:
            ser = _yf_close(src["yahoo"], start, end)
            if len(ser.dropna()) > 1:
                assets_daily[name] = ser; ok.append(f"{name}: Yahoo ({src['yahoo']})")
            else:
                fail.append(f"{name}: Yahoo empty ({src['yahoo']})")
        except Exception as e:
            fail.append(f"{name}: Yahoo fail ({src['yahoo']}) -> {e}")

    def _concat(sd):
        keep = {k:v for k,v in sd.items() if isinstance(v, pd.Series) and len(v.dropna())>0}
        return pd.concat(keep, axis=1) if keep else pd.DataFrame()

    df = _concat(assets_daily)
    if "GOLD" not in df.columns:
        gold = _yf_close("GC=F", start, end)
        if len(gold.dropna())>1: df["GOLD"]=gold
        else:
            gld = _yf_close("GLD", start, end)
            if len(gld.dropna())>1: df["GOLD"]=gld
            else: raise RuntimeError("ไม่พบราคาทองคำจาก GC=F/GLD")
    if df.empty: raise RuntimeError("ดึงราคา assets ไม่สำเร็จ")
    return df.asfreq("D").ffill()

# ---------- 3) Monthly/Annual panels ----------
def monthly_panels(wk: pd.DataFrame, assets_df: pd.DataFrame):
    gli_m = wk["GLI_INDEX"].resample("M").last().rename("GLI_INDEX")
    assets_m = assets_df.resample("M").last()
    monthly = pd.concat([gli_m, assets_m], axis=1).dropna(how="any")
    monthly = _flatten_cols(monthly)
    mrets = monthly.pct_change().dropna() * 100.0
    return monthly, mrets

def annual_panel(wk, assets_df):
    assets_yr = assets_df.resample("A-DEC").last()
    gli_yr = wk["GLI_INDEX"].resample("A-DEC").last().to_frame("GLI_INDEX")
    annual = assets_yr.join(gli_yr, how="inner")
    return _flatten_cols(annual)

# ---------- 4) Metrics tables ----------
def metrics_tables(monthly, mrets, annual, rf_annual=0.02, years_n=10):
    assets_cols = [c for c in annual.columns if c != "GLI_INDEX"]
    rows = []
    for a in assets_cols:
        cagr_full = cagr_from_series(annual[a])
        cagr_n    = cagr_last_n_years(annual[a], years_n)
        gli_full  = cagr_from_series(annual["GLI_INDEX"])
        gli_n     = cagr_last_n_years(annual["GLI_INDEX"], years_n)
        liq_full  = (cagr_full - gli_full) if (pd.notna(cagr_full) and pd.notna(gli_full)) else np.nan
        liq_n     = (cagr_n - gli_n) if (pd.notna(cagr_n) and pd.notna(gli_n)) else np.nan

        mret = mrets[a].align(mrets["GLI_INDEX"], join="inner")[0]
        vol   = ann_vol_from_returns(mret, PER_YEAR_M)
        shp   = sharpe(mret, rf_annual, PER_YEAR_M)
        srt   = sortino(mret, rf_annual, PER_YEAR_M)

        mseries = monthly[a]
        mdd = max_drawdown(mseries/mseries.iloc[0]) if len(mseries.dropna())>1 else np.nan
        cal = ( (cagr_full/abs(mdd)) if (pd.notna(cagr_full) and pd.notna(mdd) and mdd!=0) else np.nan )

        rows.append({
            "Asset": a,
            "CAGR_full_%": round(100*cagr_full, 2) if pd.notna(cagr_full) else np.nan,
            f"CAGR_{years_n}Y_%": round(100*cagr_n, 2) if pd.notna(cagr_n) else np.nan,
            "GLI_CAGR_full_%": round(100*gli_full, 2) if pd.notna(gli_full) else np.nan,
            f"GLI_CAGR_{years_n}Y_%": round(100*gli_n, 2) if pd.notna(gli_n) else np.nan,
            "LiquidityAdj_CAGR_full_%": round(100*liq_full, 2) if pd.notna(liq_full) else np.nan,
            f"LiquidityAdj_CAGR_{years_n}Y_%": round(100*liq_n, 2) if pd.notna(liq_n) else np.nan,
            "AnnVol_%(monthly)": round(100*vol, 2) if pd.notna(vol) else np.nan,
            "Sharpe(monthly)": round(shp, 2) if pd.notna(shp) else np.nan,
            "Sortino(monthly)": round(srt, 2) if pd.notna(srt) else np.nan,
            "MaxDD_%": round(100*mdd, 2) if pd.notna(mdd) else np.nan,
            "Calmar": round(cal, 2) if pd.notna(cal) else np.nan,
        })
    metrics = pd.DataFrame(rows)

    corr = mrets.corr()

    # betas (OLS, monthly)
    betas = {}
    for a in [c for c in mrets.columns if c != "GLI_INDEX"]:
        y = mrets[a].dropna(); x = mrets["GLI_INDEX"].reindex(y.index).dropna()
        common = y.index.intersection(x.index)
        if len(common) > 12:
            Y = y.loc[common].values; X = add_constant(x.loc[common].values)
            m = OLS(Y, X).fit()
            betas[a] = {"Beta_vs_GLI": m.params[1], "Alpha_%/mo": m.params[0], "R2": m.rsquared}
        else:
            betas[a] = {"Beta_vs_GLI": np.nan, "Alpha_%/mo": np.nan, "R2": np.nan}
    betas_df = pd.DataFrame(betas).T
    return metrics, corr, betas_df

# ---------- 5) Rolling (monthly) ----------
def rolling_beta(asset_ret, bench_ret, window=12):
    cov = asset_ret.rolling(window, min_periods=window).cov(bench_ret)
    var = bench_ret.rolling(window, min_periods=window).var()
    return cov / var

def roll_metrics(monthly, mrets, window=12):
    # corr
    roll_corr = {a: mrets[a].rolling(window).corr(mrets["GLI_INDEX"]) for a in mrets.columns if a!="GLI_INDEX"}
    roll_corr = pd.DataFrame(roll_corr).dropna(how="all")

    # beta
    g = mrets["GLI_INDEX"]
    roll_beta = {a: rolling_beta(mrets[a], g, window) for a in mrets.columns if a!="GLI_INDEX"}
    roll_beta = pd.DataFrame(roll_beta).dropna(how="all")

    # alpha (approx)
    alpha = {}
    for a in [c for c in mrets.columns if c!="GLI_INDEX"]:
        beta_a = roll_beta[a].reindex(mrets.index)
        resid = mrets[a] - beta_a * g
        alpha[a] = resid.rolling(window, min_periods=window).mean()
    roll_alpha = pd.DataFrame(alpha).dropna(how="all")

    return _flatten_cols(roll_corr), _flatten_cols(roll_beta), _flatten_cols(roll_alpha)

# ---------- 6) Regime / Event study ----------
def build_regime(gli_m_series: pd.Series):
    gli_m = pd.Series(gli_m_series).dropna()
    gli_ret_m = gli_m.pct_change()*100
    gli_yoy   = gli_m.pct_change(12)*100
    regime = (gli_yoy > 0).rename("GLI_Expansion")
    return pd.concat([gli_m, gli_ret_m.rename("GLI_%MoM"), gli_yoy.rename("GLI_%YoY"), regime], axis=1).dropna()

def event_study(mrets: pd.DataFrame, regime_df: pd.DataFrame, horizons=[3,6,12]):
    reg = regime_df["GLI_Expansion"].astype(int)
    switch = reg.diff().fillna(0)
    upturns   = switch[switch == 1].index
    downturns = switch[switch == -1].index

    def cum_after(events):
        out = {}
        for h in horizons:
            res = {}
            for col in [c for c in mrets.columns if c != "GLI_INDEX"]:
                vals=[]
                for d in events:
                    win = mrets.loc[d:].iloc[1:h+1][col]
                    if len(win)==h:
                        comp = (1 + (win/100)).prod() - 1
                        vals.append(comp*100)
                res[col] = np.mean(vals) if len(vals)>0 else np.nan
            out[f"{h}M_after"] = pd.Series(res)
        return pd.DataFrame(out)
    return cum_after(upturns), cum_after(downturns)
