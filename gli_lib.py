# gli_lib.py
import os, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# fredapi เป็น optional: ถ้าไม่มี key จะอธิบาย error ชัดเจน
try:
    from fredapi import Fred
except Exception:
    Fred = None

# ---------- small utils ----------
def _years_span(idx):
    if hasattr(idx[0], "to_timestamp"):
        start = idx[0].to_timestamp(); end = idx[-1].to_timestamp()
        return (end - start).days / 365.25
    if hasattr(idx[0], "to_pydatetime") or hasattr(idx[0], "year"):
        start, end = pd.to_datetime(idx[0]), pd.to_datetime(idx[-1])
        return (end - start).days / 365.25
    return float(idx[-1] - idx[0])

def cagr_from_series(series):
    s = pd.Series(series).dropna()
    if len(s) < 2: return np.nan
    yrs = _years_span(s.index)
    return (s.iloc[-1]/s.iloc[0])**(1.0/yrs) - 1.0

def cagr_last_n_years(series, n):
    s = pd.Series(series).dropna()
    if len(s) < 2: return np.nan
    if not isinstance(s.index, pd.DatetimeIndex): s.index = pd.to_datetime(s.index)
    cutoff = s.index[-1] - pd.DateOffset(years=n)
    s = s[s.index >= cutoff]
    if len(s) < 2: return np.nan
    yrs = _years_span(s.index)
    return (s.iloc[-1]/s.iloc[0])**(1.0/yrs) - 1.0

def ann_vol_from_returns(ret_series, periods_per_year):
    r = pd.Series(ret_series).dropna() / 100.0
    return r.std(ddof=0) * np.sqrt(periods_per_year)

def sharpe(ret_series, rf_annual, periods_per_year):
    r = pd.Series(ret_series).dropna() / 100.0
    if r.empty: return np.nan
    rf = rf_annual / periods_per_year
    ex = r - rf
    sd = r.std(ddof=0)
    return np.nan if sd == 0 else (ex.mean() / sd) * np.sqrt(periods_per_year)

def max_drawdown(series):
    s = pd.Series(series).dropna().astype(float)
    if s.empty: return np.nan
    dd = s/s.cummax() - 1.0
    return float(dd.min())

def rebase_to_100(s):
    s = pd.Series(s).dropna()
    return 100.0 * s / s.iloc[0]

# ---------- FRED / Yahoo helpers ----------
def _fred_series(fred, sid, start=None, end=None, force_daily=False):
    s = fred.get_series(sid)
    s.index = pd.to_datetime(s.index)
    if start: s = s[s.index >= pd.to_datetime(start)]
    if end:   s = s[s.index <= pd.to_datetime(end)]
    s = s.sort_index()
    if force_daily: s = s.asfreq("D").ffill()
    return s

def _yf_close(ticker, start=None, end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty or "Close" not in df: return pd.Series(dtype=float)
    s = df["Close"].copy(); s.index = pd.to_datetime(s.index); return s.sort_index()

# ---------- PUBLIC APIS ----------
def load_all(
    fred_api_key: str,
    start="2008-01-01",
    end=None,
    years_for_cagr=10,
    risk_free_annual=0.02,
    include_pboc=False,
    pboc_series_id=None
):
    """
    สร้าง GLI proxy + ดึง NASDAQ/SP500/GOLD/BTC/ETH
    คืน dict พร้อม dataframes/fig ที่หน้า Dashboard ใช้
    """
    if Fred is None:
        raise RuntimeError("ไม่พบ fredapi — โปรดเพิ่ม fredapi ใน requirements.txt")
    if not fred_api_key:
        raise RuntimeError("FRED_API_KEY ไม่ถูกตั้งค่า (ใส่ใน secrets หรือ env)")

    fred = Fred(api_key=fred_api_key)

    SERIES = {
        "FED_WALCL":  "WALCL",
        "ECB_ASSETS": "ECBASSETSW",
        "BOJ_ASSETS": "JPNASSETS",
        "TGA":        "WTREGEN",
        "ONRRP":      "RRPONTSYD",
        "USDJPY":     "DEXJPUS",
        "USDEUR":     "DEXUSEU",
    }

    raw = {k: _fred_series(fred, sid, start, end) for k, sid in SERIES.items()}

    def to_weekly(s): return s.resample("W-FRI").last()

    wk = pd.DataFrame({
        "FED_WALCL":  to_weekly(raw["FED_WALCL"]),
        "ECB_ASSETS": to_weekly(raw["ECB_ASSETS"]),
        "BOJ_ASSETS": to_weekly(raw["BOJ_ASSETS"]),
        "TGA":        to_weekly(raw["TGA"]),
        "ONRRP":      to_weekly(raw["ONRRP"]),
        "USDJPY":     to_weekly(raw["USDJPY"]),
        "USDEUR":     to_weekly(raw["USDEUR"]),
    }).dropna(how="any")

    wk["ECB_USD"] = wk["ECB_ASSETS"] * wk["USDEUR"]                    # EUR→USD
    wk["BOJ_USD"] = wk["BOJ_ASSETS"] / wk["USDJPY"].replace(0, np.nan) # JPY→USD
    if include_pboc and pboc_series_id:
        try:
            wk["PBOC_USD"] = to_weekly(_fred_series(fred, pboc_series_id, start, end)).reindex(wk.index).interpolate()
        except Exception:
            wk["PBOC_USD"] = 0.0
    else:
        wk["PBOC_USD"] = 0.0

    wk["GLI_USD"]   = wk["FED_WALCL"] + wk["ECB_USD"] + wk["BOJ_USD"] + wk["PBOC_USD"] - wk["TGA"] - wk["ONRRP"]
    wk["GLI_INDEX"] = 100 * wk["GLI_USD"] / wk["GLI_USD"].iloc[0]

    # Assets (prefer FRED; Gold/ETH via Yahoo)
    REQUESTS = {
        "NASDAQ": {"fred": "NASDAQCOM", "yahoo": "^IXIC"},
        "SP500":  {"fred": "SP500",     "yahoo": "^GSPC"},
        "GOLD":   {"fred": None,        "yahoo": "GC=F"},
        "BTC":    {"fred": "CBBTCUSD",  "yahoo": "BTC-USD"},
        "ETH":    {"fred": None,        "yahoo": "ETH-USD"},
    }
    assets = {}
    for name, src in REQUESTS.items():
        s = pd.Series(dtype=float)
        if src["fred"]:
            try:
                s = _fred_series(fred, src["fred"], start, end, force_daily=True)
            except Exception:
                s = pd.Series(dtype=float)
        if s.dropna().size < 2:
            s = _yf_close(src["yahoo"], start, end)
        if s.dropna().size < 2 and name == "GOLD":  # last fallback
            s = _yf_close("GLD", start, end)
        assets[name] = s

    assets_df = pd.concat({k:v for k,v in assets.items() if v.dropna().size}, axis=1)
    assets_df = assets_df.asfreq("D").ffill()

    # Monthly & Annual
    gli_m    = wk["GLI_INDEX"].resample("M").last().rename("GLI_INDEX")
    assets_m = assets_df.resample("M").last()
    monthly  = pd.concat([gli_m, assets_m], axis=1).dropna(how="any")
    monthly_rets = monthly.pct_change().dropna() * 100.0

    assets_yr = assets_df.resample("A-DEC").last()
    gli_yr    = wk["GLI_INDEX"].resample("A-DEC").last().to_frame("GLI_INDEX")
    
    # บังคับให้เป็น DatetimeIndex แบบ normalized
    assets_yr.index = pd.to_datetime(assets_yr.index).tz_localize(None).normalize()
    gli_yr.index    = pd.to_datetime(gli_yr.index).tz_localize(None).normalize()
    
    # รวมด้วย concat บน index แล้วทิ้งปีที่ GLI ไม่มีค่า
    annual = pd.concat([assets_yr, gli_yr], axis=1).dropna(subset=["GLI_INDEX"])

    # Metrics
    rows=[]; PER_YEAR_M=12
    for a in [c for c in annual.columns if c!="GLI_INDEX"]:
        cagr_full = cagr_from_series(annual[a])
        cagr_n    = cagr_last_n_years(annual[a], years_for_cagr)
        gli_full  = cagr_from_series(annual["GLI_INDEX"])
        gli_n     = cagr_last_n_years(annual["GLI_INDEX"], years_for_cagr)
        liq_full  = cagr_full - gli_full if (pd.notna(cagr_full) and pd.notna(gli_full)) else np.nan
        liq_n     = cagr_n - gli_n if (pd.notna(cagr_n) and pd.notna(gli_n)) else np.nan

        mret = monthly_rets[a].align(monthly_rets["GLI_INDEX"], join="inner")[0]
        vol_m = ann_vol_from_returns(mret, PER_YEAR_M)
        shrp  = sharpe(mret, risk_free_annual, PER_YEAR_M)

        series_m = monthly[a]
        mdd = max_drawdown(series_m/series_m.iloc[0])

        rows.append({
            "Asset": a,
            "CAGR_full_%": round(100*cagr_full,2) if pd.notna(cagr_full) else np.nan,
            f"CAGR_{years_for_cagr}Y_%": round(100*cagr_n,2) if pd.notna(cagr_n) else np.nan,
            "GLI_CAGR_full_%": round(100*gli_full,2) if pd.notna(gli_full) else np.nan,
            f"GLI_CAGR_{years_for_cagr}Y_%": round(100*gli_n,2) if pd.notna(gli_n) else np.nan,
            "LiquidityAdj_CAGR_full_%": round(100*liq_full,2) if pd.notna(liq_full) else np.nan,
            f"LiquidityAdj_CAGR_{years_for_cagr}Y_%": round(100*liq_n,2) if pd.notna(liq_n) else np.nan,
            "AnnVol_%(monthly)": round(100*vol_m,2) if pd.notna(vol_m) else np.nan,
            "Sharpe(monthly)": round(shrp,2) if pd.notna(shrp) else np.nan,
            "MaxDD_%": round(100*mdd,2) if pd.notna(mdd) else np.nan,
        })
    metrics_table = pd.DataFrame(rows)

    # Corr & Beta (monthly)
    corr_matrix = monthly_rets.corr()
    betas={}
    for a in [c for c in monthly_rets.columns if c!="GLI_INDEX"]:
        y = monthly_rets[a].dropna()
        x = monthly_rets["GLI_INDEX"].reindex(y.index).dropna()
        idx = y.index.intersection(x.index)
        if len(idx) > 12:
            Y = y.loc[idx].values; X = add_constant(x.loc[idx].values)
            m = OLS(Y, X).fit()
            betas[a] = {"Beta_vs_GLI": m.params[1], "Alpha_%/mo": m.params[0], "R2": m.rsquared}
        else:
            betas[a] = {"Beta_vs_GLI": np.nan, "Alpha_%/mo": np.nan, "R2": np.nan}
    betas_df = pd.DataFrame(betas).T

    # Rebased panel
    rebased_m = pd.DataFrame({"GLI": rebase_to_100(monthly["GLI_INDEX"])})
    for c in [col for col in monthly.columns if col!="GLI_INDEX"]:
        rebased_m[c] = rebase_to_100(monthly[c])

    # Annual YoY figure
    annual_rets_plot = annual.pct_change().dropna()*100.0
    annual_rets_plot = annual_rets_plot.rename(columns={"GLI_INDEX":"GLI_%YoY"})
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=annual_rets_plot.index, y=annual_rets_plot["GLI_%YoY"],
                              mode="lines+markers", name="GLI%YoY"))
    for c in [col for col in annual_rets_plot.columns if col!="GLI_%YoY"]:
        fig2.add_trace(go.Bar(x=annual_rets_plot.index, y=annual_rets_plot[c], name=f"{c}_%YoY"))
    fig2.update_layout(title="Annual YoY: GLI (line) vs Assets (bars)",
                       barmode="group", hovermode="x unified",
                       legend=dict(orientation="h", y=1.05),
                       xaxis=dict(rangeslider=dict(visible=True)))

    return {
        "wk": wk,
        "monthly": monthly,
        "monthly_rets": monthly_rets,
        "annual": annual,
        "metrics_table": metrics_table,
        "corr_matrix": corr_matrix,
        "betas_df": betas_df,
        "rebased_m": rebased_m,
        "annual_yoy_fig": fig2,
    }

def rolling_corr_beta_alpha(monthly_rets: pd.DataFrame, window=12):
    g = monthly_rets["GLI_INDEX"]
    # corr
    rc = {a: monthly_rets[a].rolling(window).corr(g) for a in monthly_rets.columns if a!="GLI_INDEX"}
    roll_corr = pd.DataFrame(rc).dropna(how="all")
    # beta
    cov = {a: monthly_rets[a].rolling(window, min_periods=window).cov(g) for a in monthly_rets.columns if a!="GLI_INDEX"}
    var = g.rolling(window, min_periods=window).var()
    rb  = {a: pd.Series(cov[a]/var, index=roll_corr.index.union(var.index)).reindex(monthly_rets.index) for a in cov}
    roll_beta = pd.DataFrame(rb).dropna(how="all")
    # alpha (approx)
    alpha={}
    for a in rb:
        beta_a = roll_beta[a].reindex(monthly_rets.index)
        resid  = monthly_rets[a] - beta_a*g
        alpha[a] = resid.rolling(window, min_periods=window).mean()
    roll_alpha = pd.DataFrame(alpha).dropna(how="all")
    return {"corr": roll_corr, "beta": roll_beta, "alpha": roll_alpha}

def regime_and_events(monthly, monthly_rets):
    gli_m   = monthly["GLI_INDEX"]
    gli_ret = gli_m.pct_change()*100
    gli_yoy = gli_m.pct_change(12)*100
    regime  = (gli_yoy > 0).rename("GLI_Expansion")
    regime_df = pd.concat([gli_m, gli_ret.rename("GLI_%MoM"), gli_yoy.rename("GLI_%YoY"), regime], axis=1).dropna()
    # contiguous expansions
    exp_periods=[]; in_block=False; start=None
    for t, is_exp in regime_df["GLI_Expansion"].items():
        if is_exp and not in_block: in_block=True; start=t
        if ((not is_exp) or t==regime_df.index[-1]) and in_block:
            exp_periods.append((start, t)); in_block=False
    # event study
    r = regime_df["GLI_Expansion"].astype(int); sw = r.diff().fillna(0)
    upturn   = sw[sw==1].index
    downturn = sw[sw==-1].index
    def _cum_after(returns_df, dates, horizons=[3,6,12]):
        out={}
        for h in horizons:
            res={}
            for col in returns_df.columns:
                if col=="GLI_INDEX": continue
                vals=[]
                for d in dates:
                    win = returns_df.loc[d:].iloc[1:h+1][col]
                    if len(win)==h: vals.append((1+(win/100)).prod()-1)
                res[col]= (np.mean(vals)*100) if vals else np.nan
            out[f"{h}M_after"]=pd.Series(res)
        return pd.DataFrame(out)
    evt_up   = _cum_after(monthly_rets, upturn)
    evt_down = _cum_after(monthly_rets, downturn)
    return {"regime_df": regime_df, "expansion_periods": exp_periods, "evt_up": evt_up, "evt_down": evt_down}

def perf_regime_table(monthly_rets, regime_df):
    align = monthly_rets.join(regime_df["GLI_Expansion"], how="inner")
    avg = align.groupby("GLI_Expansion").mean()
    std = align.groupby("GLI_Expansion").std()
    return pd.concat({"Avg_%/mo": avg, "Std_%/mo": std}, axis=1).round(2)

def gli_yoy_vs_gold(monthly, monthly_rets, regime_df, exp_periods):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=regime_df.index, y=regime_df["GLI_%YoY"], mode="lines", name="GLI %YoY", yaxis="y1"))
    if "GOLD" in monthly_rets.columns:
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly_rets["GOLD"], mode="lines", name="GOLD %/mo", yaxis="y2", opacity=0.6))
    for s, e in exp_periods:
        fig.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
    fig.update_layout(title="GLI YoY (ซ้าย) vs GOLD %/เดือน (ขวา) + Expansion",
                      hovermode="x unified",
                      legend=dict(orientation="h", y=1.05),
                      xaxis=dict(rangeslider=dict(visible=True)),
                      yaxis=dict(title="GLI %YoY"),
                      yaxis2=dict(title="GOLD %/mo", overlaying="y", side="right"))
    return fig

def auto_summary(metrics_table, betas_df, evt_up, evt_down, perf_regime_tbl):
    lines=[]
    lines.append("สรุปย่อ (วัดเป็นรายเดือน):")
    try:
        liq_cols=[c for c in metrics_table.columns if "LiquidityAdj_CAGR" in c]
        liq_mean=metrics_table.set_index("Asset")[liq_cols].mean(axis=1).sort_values(ascending=False)
        past_top=", ".join(liq_mean.head(2).index.tolist()); past_bot=", ".join(liq_mean.tail(1).index.tolist())
        lines.append(f"- อดีต: เมื่อเทียบกับ GLI ระยะยาว เด่นสุด: {past_top}; อ่อนสุด: {past_bot}")
    except Exception:
        lines.append("- อดีต: (สรุป Liquidity-Adj CAGR ไม่สำเร็จ)")
    try:
        latest_beta = betas_df["Beta_vs_GLI"].sort_values(ascending=False)
        lines.append(f"- ปัจจุบัน: Beta ต่อ GLI สูงสุด: {latest_beta.index[0]} | ต่ำสุด: {latest_beta.index[-1]}")
    except Exception:
        lines.append("- ปัจจุบัน: (ข้อมูลเบต้าไม่ครบ)")
    try:
        avg = perf_regime_tbl["Avg_%/mo"]
        exp_winners = avg.loc[True].sort_values(ascending=False).head(2).index.tolist()
        con_winners = avg.loc[False].sort_values(ascending=False).head(2).index.tolist()
        lines.append(f"- อนาคต (ตามสถานการณ์): หาก GLI ขยาย → เน้น {', '.join(exp_winners)}; ถ้า GLI หด → เน้น {', '.join(con_winners)}")
    except Exception:
        lines.append("- อนาคต: (สรุป regime ไม่สำเร็จ)")
    return "\n".join(lines)
