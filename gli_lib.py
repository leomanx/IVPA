# gli_lib.py — Robust helpers for GLI Streamlit page
import time
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, Iterable, Optional

# Optional imports; keep local to avoid hard crashes if absent
try:
    from fredapi import Fred
except Exception:
    Fred = None  # handled downstream

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
except Exception:
    OLS = None
    def add_constant(x): return np.c_[np.ones(len(x)), x]

# ----------------- General utils -----------------
def _to_dt_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated()].sort_index()
    return s

def _first_nonzero(v: pd.Series) -> Optional[float]:
    v = pd.Series(v).dropna()
    nz = v[np.abs(v) > 0]
    return None if nz.empty else float(nz.iloc[0])

def _safe_div(a, b):
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float).replace({0: np.nan})
    return a / b

# ----------------- FRED helpers -----------------
def get_fred(api_key: str):
    if not api_key or Fred is None:
        raise RuntimeError("FRED API key missing or fredapi not installed.")
    return Fred(api_key=api_key)

def fred_series(fred, sid: str, start=None, end=None, retries=2, sleep=0.4) -> pd.Series:
    last_err = None
    for _ in range(retries + 1):
        try:
            s = fred.get_series(sid)
            s = _to_dt_index(pd.Series(s, name=sid))
            if start: s = s[s.index >= pd.to_datetime(start)]
            if end:   s = s[s.index <= pd.to_datetime(end)]
            return s.sort_index()
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"FRED fetch failed for {sid}: {last_err}")

def to_weekly_last(s: pd.Series) -> pd.Series:
    if s is None or s.empty: return pd.Series(dtype=float)
    s = _to_dt_index(s)
    return s.resample("W-FRI").last()

# ----------------- Yahoo helpers -----------------
def yf_close(ticker: str, start=None, end=None) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=True)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if col is None:
        return pd.Series(dtype=float, name=ticker)
    s = df[col].copy()
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return _to_dt_index(s)

# ----------------- GLI proxy -----------------
def build_gli_proxy(fred, start="2008-01-01", end=None) -> pd.DataFrame:
    """GLI = Fed + ECB(USD) + BoJ(USD) - TGA - ONRRP"""
    if fred is None:
        raise RuntimeError("FRED object is None — cannot build GLI without FRED.")

    S = {
        "FED": "WALCL",        # USD
        "ECB": "ECBASSETSW",   # EUR
        "BOJ": "JPNASSETS",    # JPY
        "TGA": "WTREGEN",      # USD
        "ONRRP": "RRPONTSYD",  # USD
        "JPY": "DEXJPUS",      # JPY per USD
        "EUR": "DEXUSEU",      # USD per EUR
    }

    raw = {}
    for k, sid in S.items():
        raw[k] = fred_series(fred, sid, start=start, end=end)

    df = pd.DataFrame({
        "FED":   to_weekly_last(raw["FED"]),
        "ECB":   to_weekly_last(raw["ECB"]),
        "BOJ":   to_weekly_last(raw["BOJ"]),
        "TGA":   to_weekly_last(raw["TGA"]),
        "ONRRP": to_weekly_last(raw["ONRRP"]),
        "JPY":   to_weekly_last(raw["JPY"]),
        "EUR":   to_weekly_last(raw["EUR"]),
    }).dropna(how="any")

    # EUR->USD, JPY->USD
    df["ECB_USD"] = df["ECB"] * df["EUR"]                     # (EUR) * (USD/EUR) = USD
    df["BOJ_USD"] = _safe_div(df["BOJ"], df["JPY"])           # (JPY) / (JPY/USD) = USD

    gli_usd = df["FED"] + df["ECB_USD"] + df["BOJ_USD"] - df["TGA"] - df["ONRRP"]
    base = _first_nonzero(gli_usd)
    if base is None or np.isnan(base):
        raise RuntimeError("GLI base is zero/NaN — check FRED series availability.")
    gli_index = 100.0 * (gli_usd / base)

    out = pd.DataFrame({"GLI_USD": gli_usd, "GLI_INDEX": gli_index}).dropna()
    return out

# ----------------- Assets panel -----------------
def fetch_assets(fred, start="2008-01-01", end=None) -> pd.DataFrame:
    """
    GOLD = GC=F (fall back GLD), ETH from Yahoo.
    Others try FRED first then Yahoo.
    """
    req = {
        "NASDAQ": {"fred": "NASDAQCOM", "yahoo": "^IXIC"},
        "SP500":  {"fred": "SP500",     "yahoo": "^GSPC"},
        "GOLD":   {"fred": None,        "yahoo": "GC=F"},
        "BTC":    {"fred": "CBBTCUSD",  "yahoo": "BTC-USD"},
        "ETH":    {"fred": None,        "yahoo": "ETH-USD"},
    }
    cols: Dict[str, pd.Series] = {}

    for name, src in req.items():
        s = pd.Series(dtype=float)
        # FRED first (if defined)
        if fred is not None and src["fred"]:
            try:
                s = fred_series(fred, src["fred"], start, end).asfreq("D").ffill()
            except Exception:
                s = pd.Series(dtype=float)
        # Yahoo fallback
        if s.dropna().size < 2:
            s = yf_close(src["yahoo"], start, end)
        if s.dropna().size >= 2:
            s.name = name
            cols[name] = s

    # Ensure GOLD exists
    if "GOLD" not in cols:
        g = yf_close("GC=F", start, end)
        if g.dropna().size < 2:
            g = yf_close("GLD", start, end)
        if g.dropna().size < 2:
            raise RuntimeError("Gold price not available (GC=F/GLD).")
        g.name = "GOLD"
        cols["GOLD"] = g

    if not cols:
        raise RuntimeError("No asset price series fetched. Check internet/API.")

    df = pd.concat(cols, axis=1).sort_index()
    df = df.asfreq("D").ffill()
    return df

# ----------------- Panels & Metrics -----------------
def monthly_panels(wk: pd.DataFrame, assets_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gli_m = _to_dt_index(wk["GLI_INDEX"]).resample("M").last().rename("GLI_INDEX")
    assets_m = _to_dt_index(assets_df).resample("M").last()
    monthly = pd.concat([gli_m, assets_m], axis=1).dropna(how="any")
    monthly_rets = monthly.pct_change().dropna() * 100.0
    return monthly, monthly_rets

def annual_panel(wk: pd.DataFrame, assets_df: pd.DataFrame) -> pd.DataFrame:
    gli_y = _to_dt_index(wk["GLI_INDEX"]).resample("A-DEC").last().rename("GLI_INDEX")
    assets_y = _to_dt_index(assets_df).resample("A-DEC").last()
    return pd.concat([gli_y, assets_y], axis=1).dropna(how="any")

def rebase(s: pd.Series) -> pd.Series:
    s = pd.Series(s).dropna()
    base = _first_nonzero(s)
    if base is None or np.isnan(base): return s*np.nan
    return 100.0 * s / base

def metrics_tables(monthly: pd.DataFrame, monthly_rets: pd.DataFrame, annual: pd.DataFrame,
                   rf_annual=0.02, years_n=10):
    def _years(idx): 
        i = pd.to_datetime(idx)
        return (i[-1] - i[0]).days / 365.25

    def cagr(x):
        x = pd.Series(x).dropna()
        if x.size < 2: return np.nan
        yrs = _years(x.index)
        return (x.iloc[-1]/x.iloc[0])**(1/yrs) - 1.0

    def cagr_n(x, n):
        x = pd.Series(x).dropna()
        if x.size < 2: return np.nan
        cutoff = pd.to_datetime(x.index[-1]) - pd.DateOffset(years=n)
        x = x[x.index >= cutoff]
        if x.size < 2: return np.nan
        yrs = _years(x.index)
        return (x.iloc[-1]/x.iloc[0])**(1/yrs) - 1.0

    def ann_vol_m(r):
        r = pd.Series(r).dropna()/100.0
        return r.std(ddof=0) * np.sqrt(12)

    def sharpe_m(r):
        r = pd.Series(r).dropna()/100.0
        ex = r - (rf_annual/12)
        sd = r.std(ddof=0)
        return np.nan if sd == 0 else (ex.mean()/sd) * np.sqrt(12)

    def maxdd(series):
        s = pd.Series(series).dropna().astype(float)
        if s.empty: return np.nan
        peak = s.cummax()
        return (s/peak - 1.0).min()

    def calmar(cagr_val, mdd):
        if mdd is None or np.isnan(mdd) or mdd == 0: return np.nan
        return cagr_val / abs(mdd)

    rows = []
    assets = [c for c in annual.columns if c != "GLI_INDEX"]

    for a in assets:
        c_full = cagr(annual[a]); c_n = cagr_n(annual[a], years_n)
        gli_f = cagr(annual["GLI_INDEX"]); gli_n = cagr_n(annual["GLI_INDEX"], years_n)
        liq_f = c_full - gli_f if (pd.notna(c_full) and pd.notna(gli_f)) else np.nan
        liq_n = c_n - gli_n     if (pd.notna(c_n) and pd.notna(gli_n)) else np.nan

        if a in monthly_rets.columns and "GLI_INDEX" in monthly_rets.columns:
            mret = monthly_rets[a].align(monthly_rets["GLI_INDEX"], join="inner")[0]
        else:
            mret = pd.Series(dtype=float)

        vol_m = ann_vol_m(mret); shp = sharpe_m(mret)
        mser = monthly[a] if a in monthly.columns else pd.Series(dtype=float)
        base = _safe_div(mser, mser.iloc[0] if len(mser)>0 else np.nan)
        mdd  = maxdd(base); cal = calmar(c_full, mdd)

        rows.append({
            "Asset": a,
            "CAGR_full_%": round(100*c_full, 2) if pd.notna(c_full) else np.nan,
            f"CAGR_{years_n}Y_%": round(100*c_n, 2) if pd.notna(c_n) else np.nan,
            "GLI_CAGR_full_%": round(100*gli_f, 2) if pd.notna(gli_f) else np.nan,
            f"GLI_CAGR_{years_n}Y_%": round(100*gli_n, 2) if pd.notna(gli_n) else np.nan,
            "LiquidityAdj_CAGR_full_%": round(100*liq_f, 2) if pd.notna(liq_f) else np.nan,
            f"LiquidityAdj_CAGR_{years_n}Y_%": round(100*liq_n, 2) if pd.notna(liq_n) else np.nan,
            "AnnVol_%(monthly)": round(100*vol_m, 2) if pd.notna(vol_m) else np.nan,
            "Sharpe(monthly)": round(shp, 2) if pd.notna(shp) else np.nan,
            "MaxDD_%": round(100*mdd, 2) if pd.notna(mdd) else np.nan,
            "Calmar": round(cal, 2) if pd.notna(cal) else np.nan,
        })

    metrics = pd.DataFrame(rows)
    corr = monthly_rets.corr()

    # Beta/Alpha vs GLI using OLS (if statsmodels is available)
    betas = {}
    for a in [c for c in monthly.columns if c != "GLI_INDEX"]:
        y = monthly_rets.get(a, pd.Series(dtype=float)).dropna()
        x = monthly_rets.get("GLI_INDEX", pd.Series(dtype=float)).reindex(y.index).dropna()
        common = y.index.intersection(x.index)
        if len(common) > 12 and OLS is not None:
            Y = y.loc[common].values
            X = add_constant(x.loc[common].values)
            try:
                model = OLS(Y, X).fit()
                betas[a] = {"Beta_vs_GLI": float(model.params[1]),
                            "Alpha_%/mo": float(model.params[0]),
                            "R2": float(model.rsquared)}
            except Exception:
                betas[a] = {"Beta_vs_GLI": np.nan, "Alpha_%/mo": np.nan, "R2": np.nan}
        else:
            betas[a] = {"Beta_vs_GLI": np.nan, "Alpha_%/mo": np.nan, "R2": np.nan}
    betas_df = pd.DataFrame(betas).T
    return metrics.round(2), corr.round(3), betas_df.round(3)

# ----------------- Rolling & Regime -----------------
def roll_metrics(monthly: pd.DataFrame, monthly_rets: pd.DataFrame, window=12):
    assets = [c for c in monthly.columns if c != "GLI_INDEX"]
    g = monthly_rets.get("GLI_INDEX", pd.Series(dtype=float))
    # Rolling Corr
    roll_corr = {a: monthly_rets[a].rolling(window).corr(g) for a in assets if a in monthly_rets}
    roll_corr_df = pd.DataFrame(roll_corr).dropna(how="all")
    # Rolling Beta
    def rbeta(a):
        ar = monthly_rets.get(a, pd.Series(dtype=float))
        cov = ar.rolling(window, min_periods=window).cov(g)
        var = g.rolling(window, min_periods=window).var()
        return cov / var
    roll_beta = {a: rbeta(a) for a in assets}
    roll_beta_df = pd.DataFrame(roll_beta).dropna(how="all")
    # Rolling Alpha (approx)
    roll_alpha = {}
    for a in assets:
        b = roll_beta_df.get(a, pd.Series(dtype=float)).reindex(monthly_rets.index)
        roll_alpha[a] = (monthly_rets.get(a, pd.Series(dtype=float)) - b * g).rolling(window, min_periods=window).mean()
    roll_alpha_df = pd.DataFrame(roll_alpha).dropna(how="all")
    return roll_corr_df, roll_beta_df, roll_alpha_df

def build_regime(monthly_gli: pd.Series) -> pd.DataFrame:
    glim = _to_dt_index(monthly_gli)
    yoy = glim.pct_change(12)*100
    reg = (yoy > 0).rename("GLI_Expansion")
    return pd.concat([glim.rename("GLI_INDEX"), yoy.rename("GLI_%YoY"), reg], axis=1).dropna()

def event_study(monthly_rets: pd.DataFrame, regime_df: pd.DataFrame, horizons=(3,6,12)):
    if "GLI_Expansion" not in regime_df:
        return pd.DataFrame(), pd.DataFrame()
    reg = regime_df["GLI_Expansion"].astype(int)
    sw = reg.diff().fillna(0)
    up = sw[sw == 1].index
    down = sw[sw == -1].index

    def calc(events: Iterable[pd.Timestamp]) -> pd.DataFrame:
        out = {}
        for h in horizons:
            vals = {}
            for c in monthly_rets.columns:
                if c == "GLI_INDEX": continue
                acc = []
                for d in events:
                    if d not in monthly_rets.index: continue
                    win = monthly_rets.loc[d:].iloc[1:h+1].get(c, pd.Series(dtype=float))
                    if len(win) == h:
                        comp = (1 + (win/100)).prod() - 1
                        acc.append(comp * 100)
                vals[c] = np.mean(acc) if acc else np.nan
            out[f"{h}M_after"] = pd.Series(vals)
        return pd.DataFrame(out).round(2)
    return calc(up), calc(down)
