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

# advanced stats (graceful fallback ถ้าไม่มี package)
try:
    from scipy import stats as _sp_stats
    _HAS_SCIPY = True
except ImportError:
    _sp_stats = None; _HAS_SCIPY = False

try:
    from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
    _HAS_SM_TSA = True
except ImportError:
    _HAS_SM_TSA = False

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

# ---------- FRED key helpers / errors ----------
class FredKeyError(RuntimeError):
    """ข้อผิดพลาดเกี่ยวกับ FRED API key"""
    pass

def sanitize_fred_key(raw) -> str:
    if raw is None: return ""
    s = str(raw)
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        s = s.replace(ch, "")
    return s.strip().strip("'").strip('"').strip()

def validate_fred_key_format(key: str) -> bool:
    import re
    return bool(re.fullmatch(r"[0-9a-zA-Z]{32}", key or ""))

def mask_key(key: str) -> str:
    if not key: return "(empty)"
    if len(key) <= 6: return key[0] + "***"
    return f"{key[:4]}…{key[-2:]} (len={len(key)})"

# ---------- FRED / Yahoo helpers ----------
def _fred_series(fred, sid, start=None, end=None, force_daily=False):
    try:
        s = fred.get_series(sid)
    except Exception as e:
        msg = str(e).lower()
        if "does not exist" in msg:
            raise RuntimeError(f"ดึง FRED series '{sid}' ไม่สำเร็จ: {e}") from e
        if any(k in msg for k in ("api key", "api_key", "400", "bad request", "unauthorized")):
            raise FredKeyError(
                f"FRED ปฏิเสธคำขอ (series={sid}): {e}\n"
                "ตรวจสอบว่า FRED_API_KEY ถูกต้องและยังใช้งานได้"
            ) from e
        raise RuntimeError(f"ดึง FRED series '{sid}' ไม่สำเร็จ: {e}") from e
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index)
    if start: s = s[s.index >= pd.to_datetime(start)]
    if end:   s = s[s.index <= pd.to_datetime(end)]
    s = s.sort_index()
    if force_daily: s = s.asfreq("D").ffill()
    return s

def _yf_close(ticker, start=None, end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # yfinance ≥0.2.x อาจคืน MultiIndex columns: ("Close", "BTC-USD") ฯลฯ
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" not in lvl0:
            return pd.Series(dtype=float)
        s = df["Close"]
        # df["Close"] บน MultiIndex คืน DataFrame ที่มีคอลัมน์เป็น ticker
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]   # เอาคอลัมน์แรก (single ticker)
    else:
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Close"]

    s = s.copy().squeeze()     # บังคับเป็น Series ถ้ายังเป็น DataFrame 1 col
    if not isinstance(s, pd.Series):
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.dropna().sort_index()

# ---------- PUBLIC APIS ----------
def load_all(
    fred_api_key: str,
    start="2008-01-01",
    end=None,
    years_for_cagr=10,
    risk_free_annual=0.02,
    include_pboc=False,
    pboc_series_id=None,
    normalize=False,          # เพิ่ม GLI/M2 normalization ใน return dict
    extra_assets=False,       # เพิ่ม Copper, DXY เข้า asset universe
):
    """
    สร้าง GLI proxy + ดึง NASDAQ/SP500/GOLD/BTC/ETH
    คืน dict พร้อม dataframes/fig ที่หน้า Dashboard ใช้

    normalize=True   : เรียก gli_normalize() → return['wk_norm']
    extra_assets=True: เพิ่ม Copper (HG=F) + DXY (DTWEXBGS) เข้า universe
    """
    if Fred is None:
        raise RuntimeError("ไม่พบ fredapi — โปรดเพิ่ม fredapi ใน requirements.txt")

    key = sanitize_fred_key(fred_api_key)
    if not key:
        raise FredKeyError(
            "FRED_API_KEY ไม่ถูกตั้งค่า\n"
            "- รันท้องถิ่น: .streamlit/secrets.toml\n"
            "- Cloud: App → Settings → Secrets"
        )
    if not validate_fred_key_format(key):
        raise FredKeyError(
            f"รูปแบบ FRED_API_KEY ไม่ถูกต้อง ({mask_key(key)}) — "
            "ต้องเป็น alphanumeric 32 ตัว"
        )

    fred = Fred(api_key=key)

    # Probe key ก่อนยิงทั้งชุด
    try:
        _probe = fred.get_series("WALCL")
        if _probe is None or len(_probe) == 0:
            raise FredKeyError("เชื่อมต่อ FRED ได้แต่ไม่มีข้อมูลคืนกลับ")
    except FredKeyError:
        raise
    except Exception as e:
        raise FredKeyError(f"ตรวจสอบ FRED_API_KEY ไม่ผ่าน ({mask_key(key)}): {e}") from e

    SERIES = {
        "FED_WALCL":  "WALCL",
        "ECB_ASSETS": "ECBASSETSW",
        "BOJ_ASSETS": "JPNASSETS",
        "TGA":        "WTREGEN",
        "ONRRP":      "RRPONTSYD",
        "USDJPY":     "DEXJPUS",
        "USDEUR":     "DEXUSEU",
    }

    raw = {}
    _failed = []
    for k, sid in SERIES.items():
        try:
            raw[k] = _fred_series(fred, sid, start, end)
        except FredKeyError:
            raise
        except Exception:
            raw[k] = pd.Series(dtype=float)
            _failed.append(f"{k}({sid})")

    _essential = ["FED_WALCL", "ECB_ASSETS", "BOJ_ASSETS", "TGA", "ONRRP", "USDJPY", "USDEUR"]
    _missing = [k for k in _essential if raw.get(k, pd.Series(dtype=float)).dropna().size < 2]
    if _missing:
        raise RuntimeError(
            "ดึงข้อมูล FRED ที่จำเป็นต่อ GLI ไม่ครบ: " + ", ".join(_missing)
            + (f" | ล้มเหลว: {', '.join(_failed)}" if _failed else "")
        )

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
    if extra_assets:
        REQUESTS.update({
            # Copper: leading demand indicator, correlates with GLI expansion 3-6M ahead
            "COPPER": {"fred": None,        "yahoo": "HG=F"},
            # DXY: inverse of global dollar liquidity (stronger USD = tighter global conditions)
            "DXY":    {"fred": "DTWEXBGS",  "yahoo": "DX-Y.NYB"},
        })
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
    # บังคับชื่อคอลัมน์เป็น string เสมอ (ป้องกัน tuple จาก yfinance MultiIndex)
    assets_df.columns = [str(c) if not isinstance(c, str) else c
                         for c in assets_df.columns]
    assets_df = assets_df.asfreq("D").ffill()

    # Monthly & Annual
    gli_m    = wk["GLI_INDEX"].resample("ME").last().rename("GLI_INDEX")
    assets_m = assets_df.resample("ME").last()
    monthly  = pd.concat([gli_m, assets_m], axis=1).dropna(how="any")
    monthly_rets = monthly.pct_change().dropna() * 100.0

    assets_yr = assets_df.resample("YE-DEC").last()
    gli_yr    = wk["GLI_INDEX"].resample("YE-DEC").last().to_frame("GLI_INDEX")
    
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
        "wk_norm": gli_normalize(wk, key, start=start) if normalize else None,
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
    """สรุปย่อรายเดือนแบบอ่านง่าย พร้อมทำความสะอาดชื่อสินทรัพย์"""
    def _pretty_name(x: object) -> str:
        s = str(x)
        # ตัดเศษซ้อน เช่น tuple/list แอบถูก cast เป็นสตริง
        s = s.strip().strip("()[]")
        # ตัดเศษคอมมา/ช่องว่างปลายหาง
        s = s.replace("'", "").replace('"', "").strip()
        # map ชื่อยอดฮิตให้เป็นมาตรฐาน
        aliases = {
            "^GSPC": "SP500",
            "^IXIC": "NASDAQ",
            "GC=F": "GOLD",
            "GLD": "GOLD",
            "BTC-USD": "BTC",
            "ETH-USD": "ETH",
        }
        if s in aliases: 
            return aliases[s]
        # ตัด suffix ตลาดสกุลเงินยอดนิยม
        for suf in ("-USD", "-USDT", "-THB"):
            if s.endswith(suf):
                return s.replace(suf, "")
        return s

    lines = []
    lines.append("สรุปย่อ (วัดเป็นรายเดือน):")

    # ===== อดีต: Liquidity-Adj CAGR =====
    try:
        liq_cols = [c for c in metrics_table.columns if "LiquidityAdj_CAGR" in c]
        liq = metrics_table.set_index("Asset")[liq_cols].dropna(how="all")
        # เฉลี่ยข้ามคอลัมน์ LiquidityAdj_CAGR_* แล้วจัดอันดับ
        liq_mean = liq.mean(axis=1).dropna()
        if liq_mean.empty:
            raise ValueError("empty liq_mean")
        liq_mean = liq_mean.sort_values(ascending=False)
        past_top = ", ".join(_pretty_name(n) for n in liq_mean.head(2).index.tolist())
        past_bot = ", ".join(_pretty_name(n) for n in liq_mean.tail(1).index.tolist())
        lines.append(f"- อดีต: เมื่อเทียบกับ GLI ระยะยาว เด่นสุด: {past_top}; อ่อนสุด: {past_bot}")
    except Exception:
        lines.append("- อดีต: (สรุป Liquidity-Adj CAGR ไม่สำเร็จ)")

    # ===== ปัจจุบัน: Beta vs GLI =====
    try:
        beta_ser = betas_df["Beta_vs_GLI"].dropna()
        if beta_ser.empty:
            raise ValueError("empty beta")
        beta_ser = beta_ser.sort_values(ascending=False)
        beta_hi = _pretty_name(beta_ser.index[0])
        beta_lo = _pretty_name(beta_ser.index[-1])
        lines.append(f"- ปัจจุบัน: Beta ต่อ GLI สูงสุด: {beta_hi} | ต่ำสุด: {beta_lo}")
    except Exception:
        lines.append("- ปัจจุบัน: (ข้อมูลเบต้าไม่ครบ)")

    # ===== อนาคต: ชนะในแต่ละระบอบ (Expansion / Contraction) =====
    try:
        # perf_regime_tbl มี MultiIndex คอลัมน์: ('Avg_%/mo', Asset)
        avg = perf_regime_tbl["Avg_%/mo"].copy()
        # กัน GLI_INDEX ออก
        if "GLI_INDEX" in avg.columns:
            avg = avg.drop(columns=["GLI_INDEX"])
        # เลือกแถวตาม regime
        exp_row = avg.loc[True].dropna()
        con_row = avg.loc[False].dropna()
        # จัดอันดับจากมากไปน้อย
        exp_winners = [ _pretty_name(x) for x in exp_row.sort_values(ascending=False).head(2).index.tolist() ]
        con_winners = [ _pretty_name(x) for x in con_row.sort_values(ascending=False).head(2).index.tolist() ]
        if len(exp_winners)==0 and len(con_winners)==0:
            raise ValueError("no winners")
        exp_txt = ", ".join(exp_winners) if exp_winners else "—"
        con_txt = ", ".join(con_winners) if con_winners else "—"
        lines.append(f"- อนาคต (ตามสถานการณ์): หาก GLI ขยาย → เน้น {exp_txt}; ถ้า GLI หด → เน้น {con_txt}")
    except Exception:
        lines.append("- อนาคต: (สรุป regime ไม่สำเร็จ)")

    return "\n".join(lines)


# ================================================================
# SECTION 2 : ADVANCED GLI ANALYSIS  (v2 additions)
# ================================================================

# ── A. LEAD / LAG CROSS-CORRELATION ─────────────────────────────

def lead_lag_analysis(monthly_rets: pd.DataFrame, max_lag: int = 12) -> dict:
    """
    Cross-Correlation Function (CCF) ระหว่าง GLI%MoM กับแต่ละสินทรัพย์

    การตีความ lag:
      lag > 0  → GLI นำ (leading indicator): ดีที่สุดสำหรับ timing
      lag = 0  → contemporaneous
      lag < 0  → asset นำ GLI (GLI เป็น lagging)

    คืน:
      ccf_df       : DataFrame (index=lag, cols=assets) — ค่า Pearson r ทุก lag
      optimal_lags : DataFrame สรุป best lag, max|r|, p-value ต่อ asset
      ci95         : float — แนว 95% confidence interval
      fig          : Plotly CCF chart พร้อม CI band
    """
    if not _HAS_SCIPY:
        raise RuntimeError("ต้องการ scipy — pip install scipy")

    g = monthly_rets["GLI_INDEX"].dropna()
    assets = [c for c in monthly_rets.columns if c != "GLI_INDEX"]
    lags   = list(range(-max_lag, max_lag + 1))
    N_full = int(monthly_rets.dropna().shape[0])
    ci95   = 1.96 / math.sqrt(max(N_full, 1))

    ccf_dict, opt_rows = {}, []

    for a in assets:
        x = monthly_rets[a].dropna()
        aligned = pd.concat([g.rename("G"), x.rename("X")], axis=1).dropna()
        g_a, x_a = aligned["G"].values, aligned["X"].values

        corrs, pvals = [], []
        for lag in lags:
            if lag == 0:
                g_sl, x_sl = g_a, x_a
            elif lag > 0:   # GLI at t, asset at t+lag  → GLI leads
                g_sl, x_sl = g_a[:-lag], x_a[lag:]
            else:           # lag < 0  → asset leads
                g_sl, x_sl = g_a[(-lag):], x_a[:lag]

            if len(g_sl) < 12:
                corrs.append(np.nan); pvals.append(np.nan); continue
            r, p = _sp_stats.pearsonr(g_sl, x_sl)
            corrs.append(r); pvals.append(p)

        ccf_dict[a] = pd.Series(corrs, index=lags)

        abs_c  = np.abs(np.array(corrs, dtype=float))
        best_i = int(np.nanargmax(abs_c)) if not np.all(np.isnan(abs_c)) else max_lag
        opt_lag  = lags[best_i]
        opt_corr = corrs[best_i]
        opt_p    = pvals[best_i]
        r0       = corrs[max_lag]; p0 = pvals[max_lag]   # lag=0

        if isinstance(opt_lag, (int, float)):
            direction = ("GLI leads ▶" if opt_lag > 0
                         else ("Asset leads ◀" if opt_lag < 0
                         else "Contemporaneous"))
        else:
            direction = "—"

        opt_rows.append({
            "Asset":            a,
            "Optimal_Lag(mo)":  opt_lag,
            "Direction":        direction,
            "Max_|Corr|":       round(float(opt_corr), 3) if pd.notna(opt_corr) else np.nan,
            "p_value(opt)":     round(float(opt_p),    4) if pd.notna(opt_p)    else np.nan,
            "Sig_95%":          "✅" if (pd.notna(opt_p) and opt_p < 0.05) else "—",
            "Corr_lag0":        round(float(r0), 3)       if pd.notna(r0)       else np.nan,
            "p_lag0":           round(float(p0), 4)       if pd.notna(p0)       else np.nan,
        })

    ccf_df = pd.DataFrame(ccf_dict)

    # Figure
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, a in enumerate(assets):
        fig.add_trace(go.Scatter(
            x=lags, y=ccf_df[a].round(3).values,
            mode="lines+markers", name=str(a),
            line=dict(color=colors[i % len(colors)]),
            hovertemplate=f"<b>{a}</b><br>Lag=%{{x}} mo<br>r=%{{y:.3f}}<extra></extra>",
        ))
    # Zero line
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(160,160,160,0.7)",
                  annotation_text="Contemporaneous", annotation_position="top")
    # 95% CI band
    for y_ci, pos in [(ci95, "right"), (-ci95, None)]:
        ann = dict(text="95% CI", showarrow=False, x=max_lag, y=y_ci,
                   xanchor="left", font=dict(color="red", size=10)) if pos else {}
        fig.add_hline(y=y_ci, line_dash="dot", line_color="rgba(220,0,0,0.45)", **({} if not pos else {}))
        if pos:
            fig.add_annotation(text="95% CI", x=max_lag, y=y_ci, showarrow=False,
                               xanchor="left", font=dict(color="red", size=10))
    fig.update_layout(
        title="Cross-Correlation: GLI%MoM ↔ Asset%MoM<br>"
              "<sup>lag > 0 = GLI leads (นำ)  |  lag < 0 = GLI lags (ตาม)</sup>",
        xaxis=dict(title="Lag (months)", tickmode="linear", dtick=2),
        yaxis=dict(title="Pearson r", range=[-1, 1]),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.20),
    )

    return {
        "ccf_df":       ccf_df,
        "optimal_lags": pd.DataFrame(opt_rows),
        "ci95":         round(ci95, 4),
        "fig":          fig,
    }


# ── B. STATISTICAL VALIDITY TESTS ───────────────────────────────

def statistical_tests(monthly: pd.DataFrame, monthly_rets: pd.DataFrame,
                      max_lag_granger: int = 6) -> dict:
    """
    ทดสอบความแม่นเชิงสถิติของความสัมพันธ์ GLI-สินทรัพย์:

    1. ADF (Augmented Dickey-Fuller)
       • ทดสอบ stationarity ทั้ง level และ return
       • ถ้า level เป็น non-stationary → เปรียบเทียบระดับอาจ spurious

    2. Engle-Granger Cointegration
       • ถ้า GLI และสินทรัพย์ cointegrated → มี long-run equilibrium
       • สามารถสร้าง Error Correction Model (ECM) ได้

    3. Granger Causality (GLI → Asset)
       • ทดสอบว่า GLI%MoM ช่วยพยากรณ์ Asset%MoM ได้จริงหรือไม่
       • p < 0.05 = GLI Granger-cause สินทรัพย์นั้น

    คืน: {'adf_table', 'coint_table', 'granger_table'}
    """
    if not _HAS_SM_TSA:
        raise RuntimeError("ต้องการ statsmodels — pip install statsmodels")

    assets = [c for c in monthly.columns if c != "GLI_INDEX"]

    # ── 1. ADF ──────────────────────────────────────────────────
    adf_rows = []
    _cols_to_test = [("GLI_INDEX", monthly["GLI_INDEX"])] + [(a, monthly[a]) for a in assets]
    for name, lvl_raw in _cols_to_test:
        ret_raw = (monthly_rets[name] if name in monthly_rets.columns
                   else pd.Series(dtype=float))
        for suffix, s in [("_level", lvl_raw), ("_%ret", ret_raw)]:
            s = s.dropna()
            if len(s) < 20: continue
            try:
                stat, p, _, _, cv, _ = adfuller(s, autolag="AIC")
                adf_rows.append({
                    "Series":     name + suffix,
                    "ADF_stat":   round(stat, 3),
                    "p_value":    round(p, 4),
                    "CV_1%":      round(cv["1%"], 3),
                    "CV_5%":      round(cv["5%"], 3),
                    "Stationary": "✅" if p < 0.05 else "❌",
                    "Note": ("I(0) — ใช้ได้โดยตรง" if p < 0.05
                             else ("I(1) — ควร first-diff ก่อน regress"
                                   if suffix == "_level"
                                   else "⚠️ return ยัง non-stationary")),
                })
            except Exception as e:
                adf_rows.append({"Series": name + suffix, "ADF_stat": np.nan,
                                  "p_value": np.nan, "CV_1%": np.nan, "CV_5%": np.nan,
                                  "Stationary": "?", "Note": str(e)})

    adf_table = pd.DataFrame(adf_rows)

    # ── 2. Engle-Granger cointegration ──────────────────────────
    coint_rows = []
    gli_lvl = monthly["GLI_INDEX"].dropna()
    for a in assets:
        x_lvl = monthly[a].dropna()
        idx = gli_lvl.index.intersection(x_lvl.index)
        if len(idx) < 30: continue
        try:
            stat, p, cv = coint(gli_lvl.loc[idx].values, x_lvl.loc[idx].values)
            coint_rows.append({
                "GLI_vs":       a,
                "EG_stat":      round(stat, 3),
                "p_value":      round(p, 4),
                "CV_5%":        round(cv[1], 3),
                "Cointegrated": "✅" if p < 0.05 else "—",
                "Implication":  ("Long-run equilibrium → ECM applicable"
                                  if p < 0.05
                                  else "No long-run level tie → use returns only"),
            })
        except Exception as e:
            coint_rows.append({"GLI_vs": a, "EG_stat": np.nan, "p_value": np.nan,
                                "CV_5%": np.nan, "Cointegrated": "?", "Implication": str(e)})

    coint_table = pd.DataFrame(coint_rows)

    # ── 3. Granger causality ─────────────────────────────────────
    granger_rows = []
    gli_r = (monthly_rets["GLI_INDEX"]
             if "GLI_INDEX" in monthly_rets.columns
             else pd.Series(dtype=float)).dropna()
    for a in assets:
        x_r = (monthly_rets[a] if a in monthly_rets.columns
                else pd.Series(dtype=float)).dropna()
        idx  = gli_r.index.intersection(x_r.index)
        data2 = pd.concat([x_r.loc[idx].rename("Asset"),
                           gli_r.loc[idx].rename("GLI")], axis=1).dropna()
        if len(data2) < max_lag_granger + 15: continue
        try:
            res = grangercausalitytests(data2[["Asset", "GLI"]],
                                         maxlag=max_lag_granger, verbose=False)
            best_lag = min(res, key=lambda l: res[l][0]["ssr_ftest"][1])
            fstat, pval = res[best_lag][0]["ssr_ftest"][:2]
            all_p = {lag: round(res[lag][0]["ssr_ftest"][1], 3) for lag in res}
            granger_rows.append({
                "GLI → Asset":      a,
                "Best_Lag(mo)":     best_lag,
                "F_stat":           round(fstat, 3),
                "p_value":          round(pval, 4),
                "GLI_causes_Asset": "✅" if pval < 0.05 else "—",
                "All_lag_p":        str(all_p),
            })
        except Exception as e:
            granger_rows.append({"GLI → Asset": a, "Best_Lag(mo)": np.nan,
                                   "F_stat": np.nan, "p_value": np.nan,
                                   "GLI_causes_Asset": "?", "All_lag_p": str(e)})

    granger_table = pd.DataFrame(granger_rows)

    return {
        "adf_table":     adf_table,
        "coint_table":   coint_table,
        "granger_table": granger_table,
    }


# ── C. PREDICTIVE: FORWARD RETURN BY GLI REGIME/QUANTILE ────────

def forward_return_analysis(monthly: pd.DataFrame, monthly_rets: pd.DataFrame,
                             horizons: list = None, n_quantiles: int = 5) -> dict:
    """
    Predictive analysis: ถ้า GLI อยู่ใน quantile ไหน
    สินทรัพย์ให้ผลตอบแทนเฉลี่ยกี่ % ในอีก h เดือนข้างหน้า?

    Signal ที่ใช้:
      YoY signal: GLI %YoY แบ่ง Q1-Q5 (cycle position)
      MoM signal: GLI %MoM แบ่ง M1-M5 (momentum ระยะสั้น)

    คืน:
      fwd_yoy           : {f"{h}M": DataFrame(index=Qi, cols=assets)} avg fwd return
      fwd_mom           : {f"{h}M": DataFrame(index=Mi, cols=assets)}
      hit_rate_yoy      : {f"{h}M": DataFrame} % ที่ fwd return > 0
      hit_rate_mom      : {f"{h}M": DataFrame}
      sample_counts_yoy : Series — จำนวน obs ต่อ quantile
      sample_counts_mom : Series
      fig_heatmap_yoy   : Plotly heatmap 3M YoY
      fig_heatmap_mom   : Plotly heatmap 3M MoM
      horizons, n_quantiles, assets
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    assets = [c for c in monthly.columns if c != "GLI_INDEX"]
    gli    = monthly["GLI_INDEX"].dropna()

    def _qcut_safe(s, n, prefix):
        s = s.dropna()
        labels = [f"{prefix}{i+1}" for i in range(n)]
        try:
            return pd.qcut(s, n, labels=labels, duplicates="drop")
        except Exception:
            return pd.Series(np.nan, index=s.index, dtype="object")

    gli_yoy_q = _qcut_safe(gli.pct_change(12) * 100, n_quantiles, "Q")
    gli_mom_q = _qcut_safe(gli.pct_change(1)  * 100, n_quantiles, "M")

    def _build_fwd(signal_q, prefix):
        labels = [f"{prefix}{i+1}" for i in range(n_quantiles)]
        fwd_out, hit_out = {}, {}
        for h in horizons:
            rows_f, rows_h = {}, {}
            for a in assets:
                price = monthly[a].dropna()
                fwd = price.pct_change(h).shift(-h) * 100      # h-month fwd return
                df = pd.concat([signal_q.rename("Q"), fwd.rename("F")], axis=1).dropna()
                df["Q"] = df["Q"].astype(str)
                rows_f[a] = df.groupby("Q")["F"].mean().reindex(labels)
                rows_h[a] = (df.groupby("Q")["F"]
                               .apply(lambda x: round((x > 0).mean() * 100, 1))
                               .reindex(labels))
            fwd_out[f"{h}M"] = pd.DataFrame(rows_f).round(2)
            hit_out[f"{h}M"] = pd.DataFrame(rows_h)
        counts = (signal_q.astype(str).value_counts()
                  .reindex(labels).fillna(0).astype(int))
        return fwd_out, hit_out, counts

    fwd_yoy, hit_yoy, cnt_yoy = _build_fwd(gli_yoy_q, "Q")
    fwd_mom, hit_mom, cnt_mom = _build_fwd(gli_mom_q, "M")

    def _heatmap(fwd_dict, h_plot, title, q_axis_label):
        df = fwd_dict.get(f"{h_plot}M")
        if df is None or df.dropna(how="all").empty:
            return go.Figure()
        y_labels = [f"{idx} ← {q_axis_label}" for idx in df.index]
        z = df.values.tolist()
        text = [[f"{v:.1f}%" if pd.notna(v) else "—" for v in row] for row in z]
        _max = max(abs(df.stack().dropna().max()), abs(df.stack().dropna().min()), 1)
        fig = go.Figure(go.Heatmap(
            z=z, x=df.columns.tolist(), y=y_labels,
            colorscale="RdYlGn", zmid=0, zmin=-_max, zmax=_max,
            text=text, texttemplate="%{text}",
            colorbar=dict(title=f"{h_plot}M Fwd Ret %"),
            hovertemplate="Asset: %{x}<br>GLI Bin: %{y}<br>Avg Fwd Return: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{title} — {h_plot}-Month Forward Return by GLI Quantile<br>"
                  f"<sup>Q1=GLI weakest / Q{n_quantiles}=GLI strongest</sup>",
            xaxis_title="Asset",
            yaxis_title="GLI Quantile",
            margin=dict(l=180),
        )
        return fig

    return {
        "fwd_yoy":           fwd_yoy,
        "fwd_mom":           fwd_mom,
        "hit_rate_yoy":      hit_yoy,
        "hit_rate_mom":      hit_mom,
        "sample_counts_yoy": cnt_yoy,
        "sample_counts_mom": cnt_mom,
        "fig_heatmap_yoy":   _heatmap(fwd_yoy, 3, "GLI YoY Signal", "GLI%YoY"),
        "fig_heatmap_mom":   _heatmap(fwd_mom, 3, "GLI MoM Signal", "GLI%MoM"),
        "horizons":          horizons,
        "n_quantiles":       n_quantiles,
        "assets":            assets,
    }


# ── D. NORMALIZED GLI: M2 Ratio + Z-Score + Acceleration ────────

def gli_normalize(wk: pd.DataFrame, fred_api_key: str, start: str = None) -> pd.DataFrame:
    """
    เพิ่ม normalized GLI metrics เข้าใน DataFrame ที่มีโครงสร้างเดียวกับ wk จาก load_all():

    GLI_M2_RATIO   : GLI_USD / US_M2 (millions USD) — วัดสัดส่วน GLI ต่อ M2
    GLI_M2_INDEX   : GLI_M2_RATIO rebased to 100 — adjusted trend ที่ไม่ถูก inflate
    GLI_ZSCORE_36M : rolling 36-month z-score ของ GLI_USD
                     > +1.5 = extremely expansive | < -1.5 = extremely tight
    GLI_ACC        : GLI Acceleration (MoM change of YoY%) — 2nd derivative
                     > 0 = liquidity ขยายตัวเร็วขึ้น | < 0 = ชะลอตัว

    คืน: wk_out — DataFrame ที่มีทุกคอลัมน์เดิม + 4 คอลัมน์ใหม่
    """
    key = sanitize_fred_key(fred_api_key)
    if not key or not validate_fred_key_format(key):
        raise FredKeyError("FRED_API_KEY ไม่ถูกต้อง — ไม่สามารถดึง M2SL")
    if Fred is None:
        raise RuntimeError("ต้องการ fredapi")
    if "GLI_USD" not in wk.columns:
        raise ValueError("wk ต้องมีคอลัมน์ 'GLI_USD' — ใช้ DataFrame จาก load_all()['wk']")

    fred = Fred(api_key=key)

    # M2SL: billions USD, monthly, SA
    try:
        m2_raw = fred.get_series("M2SL")
    except Exception as e:
        raise RuntimeError(f"ดึง M2SL ไม่สำเร็จ: {e}") from e

    m2_raw.index = pd.to_datetime(m2_raw.index)
    if start:
        m2_raw = m2_raw[m2_raw.index >= pd.to_datetime(start)]
    # convert billions → millions (same unit as WALCL)
    m2_weekly = m2_raw.resample("W-FRI").last().ffill() * 1_000

    wk_out = wk.copy()
    common = wk_out.index.intersection(m2_weekly.index)
    wk_out.loc[common, "US_M2_Mln"] = m2_weekly.loc[common]
    wk_out["US_M2_Mln"] = wk_out["US_M2_Mln"].ffill()

    valid = wk_out["US_M2_Mln"].notna() & wk_out["GLI_USD"].notna()
    wk_out.loc[valid, "GLI_M2_RATIO"] = (
        wk_out.loc[valid, "GLI_USD"] / wk_out.loc[valid, "US_M2_Mln"]
    )
    first = wk_out["GLI_M2_RATIO"].first_valid_index()
    if first is not None:
        base = wk_out.loc[first, "GLI_M2_RATIO"]
        wk_out["GLI_M2_INDEX"] = 100.0 * wk_out["GLI_M2_RATIO"] / base

    # Rolling z-score (≈36 months = 36*52/12 ≈ 156 weeks)
    WIN_W = 156
    roll_mean = wk_out["GLI_USD"].rolling(WIN_W, min_periods=WIN_W // 2).mean()
    roll_std  = wk_out["GLI_USD"].rolling(WIN_W, min_periods=WIN_W // 2).std()
    wk_out["GLI_ZSCORE_36M"] = ((wk_out["GLI_USD"] - roll_mean) / roll_std).round(3)

    # Acceleration: 2nd derivative  d(YoY%)/dt monthly, resampled back to weekly
    gli_m   = wk_out["GLI_USD"].resample("ME").last()
    yoy_m   = gli_m.pct_change(12) * 100
    acc_m   = yoy_m.diff()
    acc_w   = acc_m.resample("W-FRI").last().reindex(wk_out.index).ffill()
    wk_out["GLI_ACC"] = acc_w.round(3)

    return wk_out


# ── E. SUMMARY HELPER ────────────────────────────────────────────

def advanced_summary(lead_lag_res: dict, stat_res: dict,
                     fwd_res: dict) -> str:
    """
    สรุปผลการวิเคราะห์ขั้นสูงแบบอ่านง่าย สำหรับแสดงใน Dashboard

    Parameters:
      lead_lag_res : output จาก lead_lag_analysis()
      stat_res     : output จาก statistical_tests()
      fwd_res      : output จาก forward_return_analysis()
    """
    lines = ["═══ Advanced GLI Analysis Summary ═══", ""]

    # 1. Lead/lag
    opt = lead_lag_res.get("optimal_lags", pd.DataFrame())
    if not opt.empty:
        leaders = opt[opt["Optimal_Lag(mo)"] > 0].sort_values("Max_|Corr|", ascending=False)
        laggers = opt[opt["Optimal_Lag(mo)"] < 0]
        lines.append("📡 Lead/Lag:")
        if not leaders.empty:
            top = leaders.iloc[0]
            lines.append(f"  • GLI นำ {top['Asset']} ที่ดีที่สุด: "
                         f"{int(top['Optimal_Lag(mo)'])} เดือน (r={top['Max_|Corr|']})")
        if not laggers.empty:
            lines.append(f"  • Asset ที่นำ GLI: "
                         + ", ".join(laggers["Asset"].tolist()))
        lines.append("")

    # 2. Granger causality
    gr = stat_res.get("granger_table", pd.DataFrame())
    if not gr.empty:
        sig = gr[gr.get("GLI_causes_Asset", pd.Series()) == "✅"]
        lines.append("📊 Granger Causality (GLI → Asset):")
        if not sig.empty:
            lines.append("  • GLI มีนัยสำคัญเชิงสาเหตุต่อ: "
                         + ", ".join(sig["GLI → Asset"].tolist()))
        else:
            lines.append("  • ไม่มี asset ใดที่ GLI Granger-cause อย่างมีนัยสำคัญ")
        lines.append("")

    # 3. Predictive: best GLI quantile
    fwd = fwd_res.get("fwd_yoy", {}).get("3M")
    if fwd is not None and not fwd.dropna(how="all").empty:
        lines.append("🔮 Predictive (3M fwd return หาก GLI อยู่ Q5):")
        top_q = fwd.index[-1]
        best_assets = fwd.loc[top_q].dropna().sort_values(ascending=False).head(2)
        for a, v in best_assets.items():
            lines.append(f"  • {a}: avg {v:+.1f}%")
        lines.append("")

    lines.append("(ผลข้างต้นเป็นค่าเฉลี่ยในอดีต ไม่ใช่การรับประกันอนาคต)")
    return "\n".join(lines)

# ================================================================
# SECTION 3 : FED PLUMBING & STEALTH LIQUIDITY ANALYSIS
# ================================================================

# ── Fed Plumbing FRED Series ─────────────────────────────────────
FED_PLUMBING_SERIES = {
    # ── Stealth Liquidity Injections ──
    "RESERVES":   "WRESBAL",        # Reserve Balances (weekly, billions USD)
                                     # = สภาพคล่องจริงในระบบธนาคาร
                                     # สูตร: Reserves ≈ WALCL − TGA − RRP
    "BTFP_PROXY": "WLCFLPCL",       # Other Credit Extensions (weekly, millions)
                                     # ≈ 0 ก่อน Mar 2023; พุ่ง $165B หลัง SVB
                                     # = BTFP + Discount Window (stealth QE)
    "MMF_ASSETS": "WRMFSL",         # Money Market Fund Assets (weekly, billions)
                                     # $6T+ ที่จอดรอ; ลดลง = เงินไหลสู่ risk assets
    # ── Market Stress / Liquidity Effectiveness ──
    "HY_SPREAD":  "BAMLH0A0HYM2",   # HY OAS Spread (daily, bps)
                                     # แคบ = liquidity ส่งถึงตลาดได้ดี
                                     # กว้าง = ตลาดตึง แม้ GLI สูง ก็ไม่ส่งผ่าน
    "YLD_CURVE":  "T10Y2Y",          # 10Y − 2Y Spread (daily, %)
                                     # < 0 = inversion = tightening cycle
                                     # un-invert ≈ turning point สำหรับ risk assets
    # ── Global Dollar Liquidity ──
    "DXY_BROAD":  "DTWEXBGS",        # Broad USD Index (daily)
                                     # แข็ง = ดูด global dollar liquidity
                                     # อ่อน = ปล่อย liquidity ออกสู่โลก (inverse GLI)
    # ── PBOC Proxy ──
    "CHINA_FX":   "RRFXRBCNM",       # China FX Reserves (monthly, billions USD)
                                     # ลด = PBOC ขาย USD ป้องหยวน = ดูด global liquidity
                                     # เพิ่ม = inject liquidity เข้าระบบ
}

# Copper via Yahoo (no FRED series for spot copper)
_COPPER_YAHOO = "HG=F"


def fed_plumbing(fred_api_key: str,
                 start: str = "2020-01-01",
                 end=None) -> dict:
    """
    ดึงและวิเคราะห์ Fed Liquidity Plumbing Tools:

    Stealth Injections  : Reserve Balances, BTFP/Discount Window, MMF Assets
    Market Stress       : HY Credit Spread, 10Y-2Y Yield Curve
    Global Dollar Flow  : DXY Broad, China FX Reserves
    Real Economy        : Copper (Dr. Copper leading indicator)

    คืน dict:
      'df_weekly'    : DataFrame weekly — Reserves, BTFP_PROXY, MMF_ASSETS, HY_SPREAD, YLD_CURVE, DXY
      'df_monthly'   : DataFrame monthly — China FX + Copper resampled
      'net_fed_liq'  : Series — Net Fed Liquidity = Reserves + BTFP_PROXY − MMF_ASSETS*0
      'fig_inject'   : Plotly stacked area — Stealth Injections
      'fig_stress'   : Plotly dual-axis — HY Spread + Yield Curve
      'fig_global'   : Plotly dual-axis — DXY + China FX Reserves
      'fig_copper'   : Plotly line — Copper vs GLI proxy
      'summary_tbl'  : DataFrame สรุปค่าล่าสุด + เทียบปีก่อน
    """
    key = sanitize_fred_key(fred_api_key)
    if not key or not validate_fred_key_format(key):
        raise FredKeyError("FRED_API_KEY ไม่ถูกต้อง")
    if Fred is None:
        raise RuntimeError("ต้องการ fredapi")

    fred = Fred(api_key=key)

    # ── Fetch all series ─────────────────────────────────────────
    raw = {}
    for name, sid in FED_PLUMBING_SERIES.items():
        try:
            s = _fred_series(fred, sid, start, end)
            raw[name] = s
        except FredKeyError:
            raise
        except Exception:
            raw[name] = pd.Series(dtype=float)

    # Copper via Yahoo
    try:
        raw["COPPER"] = _yf_close(_COPPER_YAHOO, start=start, end=end)
    except Exception:
        raw["COPPER"] = pd.Series(dtype=float)

    # ── Resample to weekly ───────────────────────────────────────
    def _to_w(s):
        """Convert any frequency to weekly (W-FRI), forward-fill."""
        if s.dropna().empty:
            return pd.Series(dtype=float)
        return s.resample("W-FRI").last().ffill()

    wk = pd.DataFrame({
        "RESERVES":   _to_w(raw.get("RESERVES",   pd.Series(dtype=float))),
        "BTFP_PROXY": _to_w(raw.get("BTFP_PROXY", pd.Series(dtype=float))),
        "MMF_ASSETS": _to_w(raw.get("MMF_ASSETS", pd.Series(dtype=float))),
        "HY_SPREAD":  _to_w(raw.get("HY_SPREAD",  pd.Series(dtype=float))),
        "YLD_CURVE":  _to_w(raw.get("YLD_CURVE",  pd.Series(dtype=float))),
        "DXY_BROAD":  _to_w(raw.get("DXY_BROAD",  pd.Series(dtype=float))),
        "COPPER":     _to_w(raw.get("COPPER",      pd.Series(dtype=float))),
    })

    # CHINA_FX stays monthly
    china_fx = raw.get("CHINA_FX", pd.Series(dtype=float))
    if not china_fx.dropna().empty:
        china_m = china_fx.resample("ME").last()
    else:
        china_m = pd.Series(dtype=float)

    # ── Net Fed Liquidity ─────────────────────────────────────────
    # Reserves (B) + BTFP_PROXY (M→B) = net banking system liquidity
    # MMF represents WHERE idle cash hides (high = risk-off = not injected into market)
    net_fed = pd.Series(dtype=float)
    if "RESERVES" in wk.columns and wk["RESERVES"].dropna().size > 0:
        r = wk["RESERVES"].dropna()
        b = (wk["BTFP_PROXY"] / 1000).reindex(r.index).fillna(0)  # millions → billions
        net_fed = (r + b).rename("Net_Fed_Liq_B")

    # ── Figures ──────────────────────────────────────────────────

    # Fig 1: Stealth Injections (stacked area)
    fig_inject = go.Figure()
    _inject_series = [
        ("RESERVES",   "#1f77b4", "Reserve Balances (B USD)"),
        ("BTFP_PROXY", "#d62728", "Emergency Loans / BTFP (M USD → scaled)"),
        ("MMF_ASSETS", "#ff7f0e", "MMF Assets (B USD) — risk-off parking"),
    ]
    for col, color, label in _inject_series:
        s = wk.get(col, pd.Series(dtype=float)).dropna()
        if s.empty:
            continue
        # BTFP is in millions; scale to billions for chart
        vals = s / 1000 if col == "BTFP_PROXY" else s
        fig_inject.add_trace(go.Scatter(
            x=vals.index, y=vals.round(1).values,
            mode="lines", name=label, line=dict(color=color, width=1.8),
            fill=("tonexty" if col != "RESERVES" else "tozeroy"),
            fillcolor=color.replace("#", "rgba(").replace(")", ",0.12)")
                       if "#" in color else f"rgba(0,0,0,0.08)",
        ))
    fig_inject.update_layout(
        title="🏦 Fed Plumbing: Stealth Liquidity (Reserves + BTFP + MMF)",
        hovermode="x unified", height=380,
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis_title="Billions USD",
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="⬆ Reserves & BTFP = inject  |  ⬆ MMF = risk-off (money not in market)",
            showarrow=False, font=dict(size=10, color="gray"),
        )],
    )

    # Fig 2: Market Stress
    fig_stress = go.Figure()
    hy  = wk.get("HY_SPREAD",  pd.Series(dtype=float)).dropna()
    yc  = wk.get("YLD_CURVE",  pd.Series(dtype=float)).dropna()
    if not hy.empty:
        fig_stress.add_trace(go.Scatter(x=hy.index, y=hy.round(2).values,
            mode="lines", name="HY OAS Spread (bps)", yaxis="y1",
            line=dict(color="#d62728", width=1.8)))
    if not yc.empty:
        fig_stress.add_trace(go.Scatter(x=yc.index, y=yc.round(3).values,
            mode="lines", name="10Y−2Y Curve (%)", yaxis="y2",
            line=dict(color="#2ca02c", width=1.8, dash="dot")))
        # Shade inversion (YC < 0)
        neg = yc[yc < 0]
        if not neg.empty:
            for seg_start, seg_end in zip(
                neg.index[np.concatenate([[True], np.diff(neg.index.astype(np.int64)) > 7*24*3600*1e9])],
                neg.index[np.concatenate([np.diff(neg.index.astype(np.int64)) > 7*24*3600*1e9, [True]])],
            ):
                fig_stress.add_vrect(x0=seg_start, x1=seg_end,
                    fillcolor="rgba(214,39,40,0.08)", line_width=0)
    fig_stress.add_hline(y=400, line_dash="dot", line_color="#d62728", opacity=0.4,
                          annotation_text="HY Stress >400bps", yref="y1")
    fig_stress.update_layout(
        title="📉 Market Stress: HY Spread (ซ้าย) + Yield Curve (ขวา)",
        hovermode="x unified", height=350,
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(title="HY OAS Spread (bps)", side="left"),
        yaxis2=dict(title="10Y−2Y (%)", overlaying="y", side="right"),
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="พื้นที่แดง = Yield Curve Inversion (tightening cycle)",
            showarrow=False, font=dict(size=10, color="gray"),
        )],
    )

    # Fig 3: Global Dollar
    fig_global = go.Figure()
    dxy = wk.get("DXY_BROAD", pd.Series(dtype=float)).dropna()
    if not dxy.empty:
        dxy_rb = 100 * dxy / dxy.iloc[0]
        fig_global.add_trace(go.Scatter(x=dxy_rb.index, y=dxy_rb.round(2).values,
            mode="lines", name="DXY Broad (rebased 100)", yaxis="y1",
            line=dict(color="#7f7f7f", width=1.8)))
    if not china_m.dropna().empty:
        fig_global.add_trace(go.Scatter(x=china_m.index, y=china_m.round(1).values,
            mode="lines+markers", name="China FX Reserves (B USD)", yaxis="y2",
            line=dict(color="#c5000b", width=1.8, dash="dash")))
    fig_global.update_layout(
        title="💵 Global Dollar Flow: DXY (ซ้าย) + China FX Reserves (ขวา)",
        hovermode="x unified", height=350,
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(title="DXY (rebased 100)", side="left"),
        yaxis2=dict(title="China FX Reserves (B USD)", overlaying="y", side="right"),
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="⬆ DXY = dollar แข็ง = ดูด global liquidity  |  ⬇ China FX = PBOC ป้องหยวน = ดูด liquidity",
            showarrow=False, font=dict(size=10, color="gray"),
        )],
    )

    # Fig 4: Copper (Dr. Copper)
    fig_copper = go.Figure()
    cu = wk.get("COPPER", pd.Series(dtype=float)).dropna()
    if not cu.empty:
        cu_rb = 100 * cu / cu.iloc[0]
        fig_copper.add_trace(go.Scatter(x=cu_rb.index, y=cu_rb.round(2).values,
            mode="lines", name="Copper (rebased 100)",
            line=dict(color="#b5541c", width=2.0)))
    fig_copper.update_layout(
        title="🔶 Dr. Copper — Leading Demand Indicator (3-6M ahead)",
        hovermode="x unified", height=300,
        yaxis_title="Index (Base=100)",
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="Copper มักนำ GDP และ GLI expansion ประมาณ 1 ไตรมาส",
            showarrow=False, font=dict(size=10, color="gray"),
        )],
    )

    # ── Summary Table ────────────────────────────────────────────
    summary_rows = []
    label_map = {
        "RESERVES":   ("Reserve Balances",   "B USD",  "⬆ inject"),
        "BTFP_PROXY": ("BTFP/Emergency Loans","M USD", "⬆ stealth QE"),
        "MMF_ASSETS": ("Money Mkt Fund AUM",  "B USD",  "⬆ risk-off"),
        "HY_SPREAD":  ("HY OAS Spread",       "bps",    "⬇ tight = ok"),
        "YLD_CURVE":  ("10Y−2Y Curve",        "%",      "⬆ steepen = ok"),
        "DXY_BROAD":  ("DXY Broad",           "index",  "⬇ = inject"),
        "COPPER":     ("Copper",              "$/lb",   "⬆ = demand ok"),
    }
    for col, (name, unit, note) in label_map.items():
        s = wk.get(col, pd.Series(dtype=float)).dropna()
        if s.empty:
            continue
        latest = s.iloc[-1]
        prev_y = s[s.index <= s.index[-1] - pd.DateOffset(years=1)]
        yoy    = ((latest / prev_y.iloc[-1] - 1) * 100) if not prev_y.empty else np.nan
        prev_m = s[s.index <= s.index[-1] - pd.DateOffset(months=1)]
        mom    = ((latest / prev_m.iloc[-1] - 1) * 100) if not prev_m.empty else np.nan
        summary_rows.append({
            "Instrument":   name,
            "Unit":         unit,
            "Latest":       round(latest, 2),
            "MoM %":        round(mom, 2) if pd.notna(mom) else np.nan,
            "YoY %":        round(yoy, 2) if pd.notna(yoy) else np.nan,
            "Signal Note":  note,
            "As of":        s.index[-1].strftime("%d %b %Y"),
        })
    # China FX monthly
    if not china_m.dropna().empty:
        s = china_m.dropna()
        latest = s.iloc[-1]
        prev_y = s.iloc[-13] if len(s) >= 13 else np.nan
        yoy    = ((latest / prev_y - 1) * 100) if pd.notna(prev_y) else np.nan
        summary_rows.append({
            "Instrument":   "China FX Reserves",
            "Unit":         "B USD",
            "Latest":       round(latest, 1),
            "MoM %":        round((latest / s.iloc[-2] - 1) * 100, 2) if len(s) >= 2 else np.nan,
            "YoY %":        round(yoy, 2) if pd.notna(yoy) else np.nan,
            "Signal Note":  "⬇ = PBOC ดูด liquidity",
            "As of":        s.index[-1].strftime("%b %Y"),
        })

    summary_tbl = pd.DataFrame(summary_rows)

    return {
        "df_weekly":    wk,
        "df_monthly":   china_m.to_frame("CHINA_FX") if not china_m.empty else pd.DataFrame(),
        "net_fed_liq":  net_fed,
        "fig_inject":   fig_inject,
        "fig_stress":   fig_stress,
        "fig_global":   fig_global,
        "fig_copper":   fig_copper,
        "summary_tbl":  summary_tbl,
    }

# ================================================================
# SECTION 4 : YEN CARRY TRADE & UNWIND ANALYSIS
# ================================================================

def yen_carry_analysis(wk: pd.DataFrame,
                       fred_api_key: str = None,
                       start: str = None) -> dict:
    """
    Yen Carry Trade + Unwind Detection

    ═══ Logic ════════════════════════════════════════════════════
    Carry Trade:
      • กู้ JPY ดอกเบี้ย ~0% → แปลงเป็น USD → ลงทุนใน US risk assets
      • USDJPY ขึ้น (JPY อ่อน) = carry expanding = risk-on
      • USDJPY ลงเร็ว (JPY แข็ง) = UNWIND = ขาย risk assets บังคับ

    GLI Connection (crucial):
      BOJ_USD = BOJ_Assets_JPY ÷ USDJPY
      • USDJPY ลง (JPY แข็ง) → BOJ_USD เพิ่มขึ้น → GLI ขึ้น  ← mechanical
      • แต่ตลาดร่วงพร้อมกัน เพราะ carry unwind = forced selling
      • Contradiction: GLI (formula) ขึ้น แต่ตลาดลง = สัญญาณ CARRY UNWIND

    Unwind Severity Thresholds (% drop of USDJPY):
      Minor  : > 3% in 4 weeks
      Major  : > 7% in 8 weeks
      Severe : > 12% in 12 weeks  (Aug 2024: 157→141 = −10.2%)

    Parameters:
      wk            : DataFrame จาก load_all()['wk'] (ต้องมี USDJPY, BOJ_USD, GLI_USD)
      fred_api_key  : optional — ถ้าให้จะดึง JGB 10Y yield
      start         : optional filter

    Returns dict:
      carry_df      : DataFrame weekly metrics
      unwind_events : DataFrame of detected unwind events
      fig_usdjpy    : USDJPY + unwind markers + carry state shading
      fig_boj_impact: BOJ_USD vs GLI — แสดง mechanical effect
      fig_vix       : VIX (ถ้าดึงได้) vs USDJPY
      fig_matrix    : Scatter matrix carry state vs asset return (ถ้า monthly_rets ให้มา)
      current_state : dict สรุปสถานะปัจจุบัน
      jgb_10y       : Series JGB yield (หรือ None)
    """
    if "USDJPY" not in wk.columns:
        raise ValueError("wk ต้องมีคอลัมน์ 'USDJPY' — ใช้ DataFrame จาก load_all()['wk']")

    df = wk[["USDJPY"]].copy()
    if "BOJ_USD" in wk.columns:
        df["BOJ_USD"] = wk["BOJ_USD"]
    if "GLI_USD" in wk.columns:
        df["GLI_USD"] = wk["GLI_USD"]
    if "BOJ_ASSETS" in wk.columns:
        df["BOJ_JPY"] = wk["BOJ_ASSETS"]

    if start:
        df = df[df.index >= pd.to_datetime(start)]

    usdjpy = df["USDJPY"].dropna()

    # ── Carry Metrics ─────────────────────────────────────────
    df["USDJPY_WoW_pct"]  = usdjpy.pct_change(1)  * 100   # weekly %
    df["USDJPY_MoM_pct"]  = usdjpy.pct_change(4)  * 100   # 4-week %
    df["USDJPY_3M_pct"]   = usdjpy.pct_change(13) * 100   # 13-week %
    df["USDJPY_MA26"]     = usdjpy.rolling(26).mean()      # 6-month MA (trend)
    df["USDJPY_MA52"]     = usdjpy.rolling(52).mean()      # 1-year MA

    # Carry Regime:
    #   EXPANDING  = USDJPY above MA26 and rising (JPY weakening = carry profitable)
    #   CONTRACTING = USDJPY below MA26 or falling
    df["CARRY_EXPANDING"] = (
        (usdjpy > df["USDJPY_MA26"]) &
        (df["USDJPY_MoM_pct"] > 0)
    )

    # ── BOJ Impact on GLI ─────────────────────────────────────
    # If USDJPY drops X%, BOJ_USD rises by roughly X% (inverse relationship)
    # This mechanically RAISES GLI even though carry is unwinding
    if "BOJ_USD" in df.columns and "GLI_USD" in df.columns:
        boj_share = (df["BOJ_USD"] / df["GLI_USD"]).rename("BOJ_GLI_share")
        df["BOJ_GLI_share"] = boj_share
        # Hypothetical GLI change from USDJPY move alone:
        # dGLI_from_BOJ ≈ −(BOJ_share × USDJPY_WoW_pct)
        df["GLI_mechanical_boost"] = -(df["BOJ_GLI_share"] * df["USDJPY_WoW_pct"])

    # ── Unwind Event Detection ────────────────────────────────
    THRESHOLDS = {
        "Minor":  {"window": 4,  "pct_drop": -3.0},
        "Major":  {"window": 8,  "pct_drop": -7.0},
        "Severe": {"window": 13, "pct_drop": -12.0},
    }
    df["UNWIND_SEVERITY"] = "None"
    for sev, cfg in THRESHOLDS.items():
        chg = usdjpy.pct_change(cfg["window"]) * 100
        mask = chg < cfg["pct_drop"]
        df.loc[mask, "UNWIND_SEVERITY"] = sev

    # Collect peak unwind events (contiguous → take most severe point)
    unwind_rows = []
    in_event, evt_start, evt_peak_date, evt_peak_val, evt_sev = False, None, None, 0.0, "None"
    _sev_rank = {"None": 0, "Minor": 1, "Major": 2, "Severe": 3}
    for dt, row in df.iterrows():
        sev = row["UNWIND_SEVERITY"]
        if sev != "None":
            if not in_event:
                in_event, evt_start = True, dt
            chg_13w = df.loc[dt, "USDJPY_3M_pct"] if "USDJPY_3M_pct" in df.columns else np.nan
            if _sev_rank.get(sev, 0) > _sev_rank.get(evt_sev, 0):
                evt_sev = sev; evt_peak_date = dt; evt_peak_val = chg_13w
        else:
            if in_event:
                unwind_rows.append({
                    "Start": evt_start, "Peak": evt_peak_date,
                    "Severity": evt_sev,
                    "USDJPY_at_Peak": usdjpy.get(evt_peak_date, np.nan),
                    "USDJPY_drop_%": round(evt_peak_val, 2) if pd.notna(evt_peak_val) else np.nan,
                })
                in_event = False; evt_sev = "None"

    unwind_events = pd.DataFrame(unwind_rows) if unwind_rows else pd.DataFrame(
        columns=["Start","Peak","Severity","USDJPY_at_Peak","USDJPY_drop_%"])

    # ── Fetch VIX (Yahoo) ─────────────────────────────────────
    vix = pd.Series(dtype=float)
    try:
        _start_str = str(start) if start else "2008-01-01"
        vix = _yf_close("^VIX", start=_start_str, end=None)
    except Exception:
        pass
    if not vix.dropna().empty:
        vix_w = vix.resample("W-FRI").last().reindex(df.index).ffill()
        df["VIX"] = vix_w

    # ── Fetch JGB 10Y yield (FRED, optional) ─────────────────
    jgb_10y = pd.Series(dtype=float)
    if fred_api_key:
        _key = sanitize_fred_key(fred_api_key)
        if _key and validate_fred_key_format(_key) and Fred is not None:
            try:
                fred = Fred(api_key=_key)
                jgb_raw = _fred_series(fred, "IRLTLT01JPM156N", start, None)
                jgb_10y = jgb_raw.resample("W-FRI").last().reindex(df.index).ffill()
                df["JGB_10Y"] = jgb_10y
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════
    _SEV_COLORS = {"Minor": "rgba(255,200,0,0.18)", "Major": "rgba(255,127,14,0.22)",
                   "Severe": "rgba(214,39,40,0.28)"}

    def _add_unwind_shading(fig):
        for _, ev in unwind_events.iterrows():
            color = _SEV_COLORS.get(ev["Severity"], "rgba(200,0,0,0.1)")
            s = ev["Start"]; e = ev["Peak"]
            if pd.notna(s) and pd.notna(e):
                fig.add_vrect(x0=s, x1=e, fillcolor=color, line_width=0)
                fig.add_annotation(x=s, y=0.98, yref="paper",
                    text=f"⚠️{ev['Severity']}", showarrow=False,
                    font=dict(size=9, color="#c00"), xanchor="left")
        return fig

    # ── Fig 1: USDJPY + Carry State ───────────────────────────
    fig_usdjpy = go.Figure()

    # Shade carry-expanding periods (green)
    expanding = df["CARRY_EXPANDING"].fillna(False)
    in_exp = False; exp_start = None
    for dt, is_exp in expanding.items():
        if is_exp and not in_exp:
            in_exp = True; exp_start = dt
        if (not is_exp or dt == expanding.index[-1]) and in_exp:
            fig_usdjpy.add_vrect(x0=exp_start, x1=dt,
                fillcolor="rgba(44,160,44,0.07)", line_width=0)
            in_exp = False

    fig_usdjpy.add_trace(go.Scatter(
        x=usdjpy.index, y=usdjpy.values, mode="lines",
        name="USD/JPY (JPY per USD)", line=dict(color="#1f77b4", width=2.0)))
    if "USDJPY_MA26" in df.columns:
        fig_usdjpy.add_trace(go.Scatter(
            x=df.index, y=df["USDJPY_MA26"].values, mode="lines",
            name="26W MA", line=dict(color="#ff7f0e", dash="dot", width=1.2)))
    if "JGB_10Y" in df.columns and df["JGB_10Y"].dropna().size > 0:
        fig_usdjpy.add_trace(go.Scatter(
            x=df.index, y=df["JGB_10Y"].values, mode="lines",
            name="JGB 10Y Yield (%)", yaxis="y2",
            line=dict(color="#9467bd", dash="dash", width=1.2)))

    _add_unwind_shading(fig_usdjpy)
    fig_usdjpy.update_layout(
        title="🇯🇵 USD/JPY + Carry Trade State<br>"
              "<sup>🟢 = Carry Expanding (JPY weak)  |  🔴 shading = Unwind Event</sup>",
        hovermode="x unified", height=400,
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(title="JPY per USD", side="left"),
        yaxis2=dict(title="JGB 10Y Yield (%)", overlaying="y", side="right"),
    )

    # ── Fig 2: BOJ Impact on GLI ──────────────────────────────
    fig_boj = go.Figure()
    if "GLI_USD" in df.columns:
        gli_rb = 100 * df["GLI_USD"] / df["GLI_USD"].dropna().iloc[0]
        fig_boj.add_trace(go.Scatter(
            x=df.index, y=gli_rb.values, mode="lines",
            name="GLI Index (rebased 100)", line=dict(color="#1f77b4", width=2)))
    if "BOJ_USD" in df.columns:
        boj_rb = 100 * df["BOJ_USD"] / df["BOJ_USD"].dropna().iloc[0]
        fig_boj.add_trace(go.Scatter(
            x=df.index, y=boj_rb.values, mode="lines",
            name="BOJ_USD Component (rebased)", line=dict(color="#d62728", width=1.5, dash="dot")))

    usdjpy_inv = 1 / usdjpy * (usdjpy.dropna().iloc[0])  # inverse scaled
    fig_boj.add_trace(go.Scatter(
        x=usdjpy_inv.index,
        y=(100 * usdjpy_inv / usdjpy_inv.dropna().iloc[0]).values,
        mode="lines", name="1/USDJPY (JPY strength, rebased)",
        line=dict(color="#2ca02c", width=1.5, dash="dash")))

    _add_unwind_shading(fig_boj)
    fig_boj.update_layout(
        title="🔗 GLI ↔ BOJ Component ↔ USDJPY Relationship<br>"
              "<sup>JPY แข็ง → BOJ_USD ขึ้น → GLI ขึ้น (mechanical) — แต่ตลาดอาจลง!</sup>",
        hovermode="x unified", height=370,
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Index (Base=100)",
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper", showarrow=False,
            text="⚠️ Contradiction: GLI (formula) ขึ้นจาก JPY แข็ง ≠ Liquidity จริงเพิ่ม",
            font=dict(size=10, color="darkred"))],
    )

    # ── Fig 3: VIX vs USDJPY ─────────────────────────────────
    fig_vix = go.Figure()
    if "VIX" in df.columns and df["VIX"].dropna().size > 0:
        fig_vix.add_trace(go.Scatter(
            x=df.index, y=df["VIX"].values, mode="lines",
            name="VIX (Fear Index)", yaxis="y1",
            line=dict(color="#d62728", width=1.5)))
        fig_vix.add_hline(y=20, line_dash="dot", line_color="orange", yref="y1",
                           annotation_text="VIX 20 = เริ่มผันผวน")
        fig_vix.add_hline(y=30, line_dash="dot", line_color="red", yref="y1",
                           annotation_text="VIX 30 = Panic zone")
    fig_vix.add_trace(go.Scatter(
        x=usdjpy.index, y=usdjpy.values, mode="lines",
        name="USD/JPY", yaxis="y2",
        line=dict(color="#1f77b4", width=1.5, dash="dot")))
    _add_unwind_shading(fig_vix)
    fig_vix.update_layout(
        title="😱 VIX vs USD/JPY — Carry Unwind = VIX พุ่ง + JPY แข็ง พร้อมกัน",
        hovermode="x unified", height=350,
        legend=dict(orientation="h", y=1.08),
        yaxis=dict(title="VIX", side="left"),
        yaxis2=dict(title="JPY per USD", overlaying="y", side="right"),
    )

    # ── Fig 4: USDJPY MoM% ───────────────────────────────────
    mom = df["USDJPY_MoM_pct"].dropna()
    colors_mom = np.where(mom.values >= 0, "#2ca02c", "#d62728")
    fig_mom = go.Figure()
    fig_mom.add_trace(go.Bar(x=mom.index, y=mom.values,
        name="USDJPY MoM %", marker_color=colors_mom))
    fig_mom.add_hline(y=-3,  line_dash="dot", line_color="orange",
                       annotation_text="Minor Unwind threshold (−3%)")
    fig_mom.add_hline(y=-7,  line_dash="dot", line_color="red",
                       annotation_text="Major Unwind threshold (−7%)")
    fig_mom.update_layout(
        title="📉 USDJPY 4-Week Change % — Carry Unwind Detection",
        hovermode="x unified", height=300,
        yaxis_title="USDJPY MoM % (4W)",
        annotations=[dict(x=0.01, y=0.97, xref="paper", yref="paper",
            text="ลงผ่าน −3% = Minor | −7% = Major Unwind",
            showarrow=False, font=dict(size=10, color="darkred"))],
    )

    # ── Current State Summary ─────────────────────────────────
    now_usdjpy   = usdjpy.iloc[-1]
    now_mom      = df["USDJPY_MoM_pct"].dropna().iloc[-1]
    now_3m       = df["USDJPY_3M_pct"].dropna().iloc[-1] if df["USDJPY_3M_pct"].dropna().size else np.nan
    now_carry    = "🟢 Expanding" if df["CARRY_EXPANDING"].iloc[-1] else "🔴 Contracting/Unwind Risk"
    now_sev      = df["USDJPY_SEVERITY"] if "USDJPY_SEVERITY" in df.columns else df["UNWIND_SEVERITY"].iloc[-1]

    if pd.notna(now_mom):
        if now_mom < -7:     unwind_status = "🚨 MAJOR UNWIND in progress"
        elif now_mom < -3:   unwind_status = "⚠️ Minor Unwind signal"
        elif now_mom > 3:    unwind_status = "✅ Carry Expanding (JPY weakening)"
        else:                unwind_status = "🟡 Neutral / Consolidating"
    else:
        unwind_status = "—"

    if "BOJ_GLI_share" in df.columns:
        boj_share_now = df["BOJ_GLI_share"].dropna().iloc[-1] * 100
    else:
        boj_share_now = np.nan

    current_state = {
        "USDJPY":           round(now_usdjpy, 2),
        "MoM_%":            round(float(now_mom), 2) if pd.notna(now_mom) else np.nan,
        "3M_%":             round(float(now_3m), 2)  if pd.notna(now_3m) else np.nan,
        "Carry_State":      now_carry,
        "Unwind_Status":    unwind_status,
        "BOJ_GLI_share_%":  round(float(boj_share_now), 1) if pd.notna(boj_share_now) else np.nan,
        "VIX_latest":       round(float(df["VIX"].dropna().iloc[-1]), 1)
                            if "VIX" in df.columns and df["VIX"].dropna().size > 0 else np.nan,
        "JGB_10Y_latest":   round(float(df["JGB_10Y"].dropna().iloc[-1]), 3)
                            if "JGB_10Y" in df.columns and df["JGB_10Y"].dropna().size > 0 else np.nan,
    }

    return {
        "carry_df":       df,
        "unwind_events":  unwind_events,
        "fig_usdjpy":     fig_usdjpy,
        "fig_boj_impact": fig_boj,
        "fig_vix":        fig_vix,
        "fig_mom":        fig_mom,
        "current_state":  current_state,
        "jgb_10y":        jgb_10y,
    }
