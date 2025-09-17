# streamlit_app.py
# IVP / DRP Analyzer — Streamlit (Plotly + HTML Export)
# - Fix: keep analysis after downloads via st.session_state.auto_run
# - New: How-to tab + templates, extra charts, econ indicator explanations
# - New: VaR/CVaR PASS/FAIL with user-set daily limit

import os, io, re, math, time, warnings, zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="IVP / DRP Analyzer", layout="wide")

# -------------------- Styling --------------------
DEFAULT_PADDING_PX = 24
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: {DEFAULT_PADDING_PX}px;
        padding-bottom: {DEFAULT_PADDING_PX}px;
        padding-left: {DEFAULT_PADDING_PX}px;
        padding-right: {DEFAULT_PADDING_PX}px;
        max-width: 1600px;
      }}
      .stPlotlyChart {{ background: transparent; }}
    </style>
    """,
    unsafe_allow_html=True
)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Defaults --------------------
SHEET_NAME     = "Portfolio"
SYMBOL_COL_D   = "Symbol"
WEIGHT_COL_D   = None
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
VAR_METHOD_D   = "hist"  # 'hist'|'normal'|'cornish'
INDEX_MULT     = 200.0
ETF_FALLBACK   = {"^GSPC":"SPY","SPY":"SPY","^NDX":"QQQ","QQQ":"QQQ","^IXIC":"QQQ"}
METHOD_ALIGN   = "intersect"
REBASE_NAV     = 100.0

# -------------------- Session init --------------------
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False  # keep analysis after reruns (e.g., download click)

# -------------------- Helpers --------------------
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

# -------------------- Data IO --------------------
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
    px = px.dropna(how="all").ffill()
    return px.dropna(how="all")

@st.cache_data(show_spinner=False)
def latest_prices(tickers, period="1mo"):
    px = yf_download(tickers, period=period)
    return pd.Series({t: float(px[t].dropna().iloc[-1]) for t in tickers if t in px.columns})

# -------------------- Analytics --------------------
def align_returns(px, method):
    px = px.astype(float)
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
        from scipy.stats import norm
        z = norm.ppf(1-alpha)
        var = -(mu + sd*z)
        cvar = -(mu - sd * (np.exp(-0.5*z*z)/np.sqrt(2*np.pi))/(1-alpha))
    elif method == "cornish":
        mu  = x.mean(); sd = x.std(ddof=0)
        if sd==0: return np.nan, np.nan
        s   = pd.Series(x).skew()
        k   = pd.Series(x).kurt()
        from scipy.stats import norm
        z   = norm.ppf(alpha)
        zcf = (z + (s/6)*(z**2-1) + (k/24)*(z**3-3*z) - (s**2/36)*(2*z**3-5*z))
        var = -(mu + sd*zcf)
        cvar = -(mu - sd * (np.exp(-0.5*zcf*zcf)/np.sqrt(2*np.pi))/(1-alpha))
    else:
        raise ValueError("method must be 'hist'|'normal'|'cornish'")
    return float(var), float(cvar)

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
        if t in over.index and bool(over.loc[t]): rs.append(f"RC>{int(rc_cap*100)}%")
        if t in bad.index and bool(bad.loc[t]):  rs.append("Sharpe<0/DeepDD")
        if t in good.index and bool(good.loc[t]): rs.append("Diversifier")
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

# -------------------- Plotly --------------------
import plotly.graph_objects as go
import plotly.io as pio

def _layout_base(title, h=420, rangeslider=False):
    lay = dict(
        title=title,
        margin=dict(l=40, r=28, t=50, b=40),
        xaxis=dict(rangeslider=dict(visible=rangeslider)),
        yaxis=dict(automargin=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    if h: lay["height"] = int(h)
    return lay

def render_plotly(fig, fallback_msg="ยังไม่พอสำหรับพล็อต"):
    if fig is None:
        st.info(fallback_msg); return
    try:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except Exception as e:
        st.warning("แสดงกราฟไม่สำเร็จ"); 
        with st.expander("รายละเอียด error"): st.exception(e)

# ---- Core charts
def p_cum_vs_bench(idx_port, bench_prices, h=420, rangeslider=True):
    s = pd.Series(idx_port).dropna()
    if s.size < 2: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=(s/s.iloc[0]).values, mode="lines", name="Portfolio",
                             hovertemplate="%{x|%Y-%m-%d}<br>Idx=%{y:.3f}<extra></extra>"))
    for name, px in bench_prices.items():
        p = pd.Series(px).dropna()
        if p.size >= 2:
            fig.add_trace(go.Scatter(x=p.index, y=(p/p.iloc[0]).values, mode="lines", name=name,
                                     hovertemplate="%{x|%Y-%m-%d}<br>Idx=%{y:.3f}<extra></extra>"))
    fig.update_layout(**_layout_base("Cumulative Index: Portfolio vs Benchmarks", h=h, rangeslider=rangeslider))
    fig.update_yaxes(title="Index (rebased=1.0)")
    return fig

def p_drawdown(idx_port, h=380):
    s = pd.Series(idx_port).dropna()
    if s.size < 2: return None
    dd = s/s.cummax() - 1.0
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown",
                               hovertemplate="%{x|%Y-%m-%d}<br>DD=%{y:.2%}<extra></extra>"))
    fig.update_layout(**_layout_base("Portfolio Drawdown", h=h, rangeslider=False))
    fig.update_yaxes(title="Drawdown")
    return fig

def p_rc_bar(metrics, col="RC_File", title=None, h=360):
    if metrics is None or col not in metrics.columns: return None
    ser = metrics[col].copy().astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    if ser.empty: return None
    ser = ser.sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=ser.index, y=ser.values, hovertemplate="%{x}<br>RC=%{y:.4f}<extra></extra>"))
    fig.update_layout(**_layout_base(title or f"Risk Contribution — {col}", h=h, rangeslider=False))
    fig.update_yaxes(title="Contribution")
    return fig

def p_rc_compare(metrics, h=360, cols=("RC_File","RC_IVP")):
    if metrics is None or not set(cols).issubset(metrics.columns): return None
    df = metrics[list(cols)].copy().astype(float).replace([np.inf,-np.inf], np.nan).dropna(how="all")
    if df.empty: return None
    order = df["RC_File"].fillna(-999).sort_values(ascending=False).index
    df = df.reindex(order)
    fig = go.Figure()
    for c in cols:
        ser = df[c].dropna()
        if ser.empty: continue
        fig.add_trace(go.Bar(x=ser.index, y=ser.values, name=("Current RC" if c=="RC_File" else "IVP RC"),
                             hovertemplate="%{x}<br>"+c+"=%{y:.4f}<extra></extra>"))
    fig.update_layout(**_layout_base("Risk Contribution — Current vs IVP (side-by-side)", h=h),
                      barmode="group")
    fig.update_yaxes(title="Contribution")
    return fig

def p_risk_return_bubble(metrics, h=420, show_labels=True):
    if metrics is None: return None
    cols = {"AnnVol","CAGR","File_Weight"}
    if not cols.issubset(metrics.columns): return None
    df = metrics[list(cols)].replace([np.inf,-np.inf], np.nan).dropna()
    if df.empty: return None
    size = (np.clip(df["File_Weight"], 0, 1) * 40) + 10
    mode = "markers+text" if show_labels else "markers"
    fig = go.Figure(go.Scatter(
        x=df["AnnVol"], y=df["CAGR"], mode=mode,
        text=df.index if show_labels else None, textposition="top center",
        marker=dict(size=size),
        hovertemplate="Sym=%{text}<br>AnnVol=%{x:.3f}<br>CAGR=%{y:.3f}<extra></extra>"
    ))
    fig.update_layout(**_layout_base("Risk–Return Map (bubble = current weight)", h=h))
    fig.update_xaxes(title="AnnVol"); fig.update_yaxes(title="CAGR")
    return fig

def p_corr_heatmap(r_assets, h=480):
    if r_assets is None or r_assets.empty: return None
    if r_assets.shape[1] > 80: r_assets = r_assets.iloc[:, :80]
    corr = r_assets.corr().replace([np.inf,-np.inf], np.nan)
    corr = corr.dropna(how="all").dropna(how="all", axis=1)
    if corr.empty: return None
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        hovertemplate="(%{y}, %{x})=%{z:.2f}<extra></extra>"
    ))
    fig.update_layout(**_layout_base("Correlation Heatmap", h=h))
    return fig

# ---- Extra charts
def p_rolling_vol(rP, rB, h=360, win=60):
    if rP is None or rP.empty: return None
    k = 252.0
    rvP = (rP.rolling(win).std(ddof=0) * np.sqrt(k)).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rvP.index, y=rvP.values, mode="lines", name="Port 60D Vol"))
    if rB is not None and not rB.empty:
        rB = rB.reindex(rvP.index).dropna()
        rvB = (rB.rolling(win).std(ddof=0) * np.sqrt(k)).dropna()
        if not rvB.empty:
            fig.add_trace(go.Scatter(x=rvB.index, y=rvB.values, mode="lines", name="Bench 60D Vol"))
    fig.update_layout(**_layout_base("Rolling 60D Annualized Volatility", h=h))
    fig.update_yaxes(title="Ann. Vol")
    return fig

def p_rolling_corr(rP, rB, h=360, win=60):
    if rP is None or rB is None or rP.empty or rB.empty: return None
    rB = rB.reindex(rP.index).dropna()
    rP = rP.reindex(rB.index).dropna()
    if rP.empty or rB.empty: return None
    rc = rP.rolling(win).corr(rB).dropna()
    fig = go.Figure(go.Scatter(x=rc.index, y=rc.values, mode="lines", name=f"Corr {win}D"))
    fig.update_layout(**_layout_base(f"Rolling {win}D Correlation (Port vs Bench)", h=h))
    fig.update_yaxes(title="Correlation", range=[-1,1])
    return fig

def p_ret_histogram(rP, VaR=None, CVaR=None, bins=60, h=360):
    if rP is None or rP.empty: return None
    fig = go.Figure()
    fig.add_histogram(x=rP.values, nbinsx=bins, name="Daily Returns", histnorm="")
    if VaR is not None and not np.isnan(VaR):
        fig.add_vline(x=-VaR, line_dash="dash", annotation_text=f"-VaR {VaR:.2%}", annotation_position="top right")
    if CVaR is not None and not np.isnan(CVaR):
        fig.add_vline(x=-CVaR, line_dash="dot", annotation_text=f"-CVaR {CVaR:.2%}", annotation_position="top right")
    fig.update_layout(**_layout_base("Distribution of Daily Returns (with VaR/CVaR)", h=h))
    fig.update_xaxes(title="Daily Return"); fig.update_yaxes(title="Frequency")
    return fig

def p_monthly_heatmap(rP, h=420):
    if rP is None or rP.empty or not isinstance(rP.index, pd.DatetimeIndex): return None
    m = rP.resample("M").apply(lambda s: (1+s).prod()-1).dropna()
    if m.empty: return None
    df = pd.DataFrame({"Year": m.index.year, "Month": m.index.month, "Ret": m.values})
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = df.pivot(index="Year", columns="Month", values="Ret")
    pivot = pivot.reindex(columns=range(1,13))
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=[months[i-1] for i in pivot.columns], y=pivot.index,
        zmin=-0.20, zmax=0.20, colorscale="RdBu", reversescale=True,
        hovertemplate=" %{y} %{x}<br>%{z:.2%}<extra></extra>"
    ))
    fig.update_layout(**_layout_base("Monthly Return Heatmap", h=h))
    return fig

# -------------------- Export helpers --------------------
def figures_to_zip_html(figs: dict):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figs.items():
            if fig is None: continue
            html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
            zf.writestr(f"{name}.html", html.encode("utf-8"))
    bio.seek(0)
    return bio

def to_excel_bytes(sheets: dict):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if df is None: continue
            if isinstance(df, (pd.Series, pd.Index)): df = df.to_frame()
            df.to_excel(xw, sheet_name=name[:31], index=True)
    bio.seek(0)
    return bio

# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Settings")
uploaded = st.sidebar.file_uploader("Upload Portfolio (CSV/XLSX)", type=["csv","xlsx"], key="uploader")
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
var_limit_pct = st.sidebar.number_input("VaR daily limit (%)", min_value=0.1, value=2.0, step=0.1, help="เกณฑ์ผ่าน/ไม่ผ่าน: VaR และ CVaR (x1.5) ต้องไม่เกิน limit นี้")

st.sidebar.markdown("---")
fred_key  = st.sidebar.text_input("FRED API Key (optional)", value=os.environ.get("FRED_API_KEY",""))

st.sidebar.markdown("---")
chart_height = st.sidebar.slider("Chart height (px)", min_value=320, max_value=900, value=460, step=20)
show_labels  = st.sidebar.toggle("Show labels on bubble chart", value=True)
rangeslider  = st.sidebar.toggle("Show time-range slider (cum chart)", value=True)

run_btn = st.sidebar.button("▶️ Run Analysis", use_container_width=True)

# -------------------- Title --------------------
st.title("Downside-Protection / IVP Portfolio Analyzer — Interactive")
st.caption("Upload → Analyze → Visualize (Plotly) → Export (HTML)")

# -------------------- How-to Tab --------------------
def howto_tab():
    st.subheader("ไฟล์ที่อัปโหลดต้องมีอะไรบ้าง?")
    st.markdown("""
- **ขั้นต่ำ**: คอลัมน์ `Symbol` (ชื่อสินทรัพย์/ติ๊กเกอร์)  
- ระบุ **น้ำหนัก** ได้ 3 ทาง (เลือกอย่างใดอย่างหนึ่ง):
  1) คอลัมน์ **Weight** (0–1 หรือ 0–100%)  
  2) คอลัมน์ **Qty** + **Mkt Price(USD)**  
  3) คอลัมน์ **Mkt Val** (มูลค่าต่อแถว)
- ชื่อคอลัมน์ที่ใช้จริงสามารถปรับได้ใน **Settings** ด้านซ้าย (เช่น `Symbol column`, `Weight column`, `Qty column`, `Price column`)
- ถ้าเป็น Excel ให้ใส่แผ่น **Sheet name** ให้ตรงกับในไฟล์ (`Portfolio`)
- แถวรวมเช่น `TOTAL`/`SUBTOTAL` ไม่ต้องลบ โปรแกรมจะคัดทิ้งให้
""")
    st.subheader("Template ตัวอย่าง")
    sample = pd.DataFrame({
        "Symbol": ["AAPL","MSFT","GLD","BTC-USD"],
        "Qty":    [10,   5,     3,     0.05],
        "Mkt Price(USD)":[200, 420, 185, 60000]
    })
    csv = sample.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV Template", data=csv, file_name="portfolio_template.csv", mime="text/csv")
    try:
        import openpyxl  # already dependency
        import io as _io
        bio = _io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as xw:
            sample.to_excel(xw, sheet_name=SHEET_NAME, index=False)
        bio.seek(0)
        st.download_button("⬇️ Download XLSX Template", data=bio.getvalue(),
                           file_name="portfolio_template.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        pass
    st.subheader("เคล็ดลับ")
    st.markdown("""
- ถ้ารวมตลาดหุ้นกับคริปโตในพอร์ต ข้อมูลรายวันอาจไม่ตรงกันทุกวัน → เราทำ **forward-fill** และ **intersect** อัตโนมัติ
- ไม่มีข้อมูลราคาบางตัว? ระบบจะเตือนและตัดออก พร้อมชั่งน้ำหนักใหม่ให้รวมเป็น 100%
- เปอร์เซ็นต์ที่กรอกเป็น 10/15 จะตีความว่า 10–15% โดยอัตโนมัติ
""")

# -------------------- Main Flow --------------------
should_run = run_btn or st.session_state.auto_run
if run_btn:
    st.session_state.auto_run = True  # keep running after reruns

tab0, tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ How to Use","📊 Metrics & Plan","📈 Charts (Interactive)","🧭 Market & Risk","📦 Export"])
with tab0:
    howto_tab()

if should_run and uploaded:
    try:
        with st.spinner("Loading portfolio..."):
            tickers, w_file = read_portfolio(uploaded, sym_col, (weight_col or None), qty_col, price_col, sheet_name)
        with tab1:
            st.success(f"Loaded {len(tickers)} symbols")
            st.dataframe(pd.DataFrame({"Weight":w_file}).rename_axis("Symbol"), use_container_width=True)

        benches = [s.strip() for s in more_benches.split(",") if s.strip()]
        all_syms = list(dict.fromkeys(tickers + ([primary_bench] if primary_bench not in benches else []) + benches))

        with st.spinner("Downloading prices..."):
            px_all = yf_download(all_syms, period=period)
        if primary_bench not in px_all.columns:
            with tab1:
                st.error(f"ไม่พบ {primary_bench} ในข้อมูลราคา"); st.stop()

        miss = [t for t in tickers if t not in px_all.columns]
        if miss:
            with tab1:
                st.warning(f"ตัดออก (ไม่มีราคา): {', '.join(miss)}")
            tickers = [t for t in tickers if t in px_all.columns]
            w_file  = w_file.reindex(tickers).fillna(0.0)
            w_file  = w_file/(w_file.sum() if w_file.sum()!=0 else 1.0)

        px_assets = px_all[tickers]
        primary_price = px_all[primary_bench].rename(primary_bench)

        metrics, r_assets, r_bench, idx_bench, cov_an, ivp_w, w_file = compute_metrics(px_assets, primary_price, w_file, RISK_FREE_PCTD)
        idx_file = (r_assets.mul(w_file, axis=1).sum(axis=1)).add(1).cumprod()
        rP = idx_file.pct_change().dropna()
        rB = r_bench.reindex(rP.index).dropna()
        rP_beta = rP.reindex(rB.index).dropna()
        betaP = beta_vs(rP_beta, rB) if len(rP_beta) else np.nan

        VaR, CVaR = var_cvar(rP, alpha=var_alpha, method=var_method)

        tp = build_trade_plan(metrics, rc_cap=rc_cap, forbid_bad_buy=True, bad_mdd=bad_mdd, bad_sharpe=bad_shp)
        last_px = latest_prices(tp.index.tolist())
        tp_exec, exec_summary = apply_trade_constraints(
            tp, last_px, pv,
            normalize_pct(cash_now, 0.10),
            normalize_pct(cash_tgt, 0.15),
            min_ticket,
            lot_func=None
        )

        # ---------- Market state ----------
        def fred_fetch(series, start="2005-01-01", key=None, retries=2, timeout=10):
            key = (key or os.environ.get("FRED_API_KEY","")).strip().strip('"').strip("'")
            if not key:
                return pd.Series(dtype=float, name=series), {"ok":False,"reason":"no_api_key"}
            url = "https://api.stlouisfed.org/fred/series/observations"
            p = {"series_id":series,"observation_start":start,"file_type":"json","api_key":key}
            last_err=None
            for _ in range(retries+1):
                try:
                    resp = requests.get(url, params=p, timeout=timeout)
                    if resp.status_code!=200:
                        last_err=f"http_{resp.status_code}:{resp.text[:120]}"; time.sleep(0.4); continue
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

            # breadth
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

        state, state_detail, fred_sum = market_state(primary_price, fred_key if fred_key else None)

        # -------------------- Tabs content --------------------
        bench_prices = {b:px_all[b] for b in benches if b in px_all.columns}
        nav_series = (pd.Series(idx_file)/pd.Series(idx_file).dropna().iloc[0])*REBASE_NAV
        nav_df = nav_series.rename("NAV").to_frame(); nav_df.index.name = "Date"

        with tab1:
            st.subheader("Asset Metrics")
            st.dataframe(metrics.round(4), use_container_width=True)

            st.subheader("Trade Plan (Executed Constraints)")
            st.dataframe(tp_exec.round(4), use_container_width=True)
            st.caption(f"Exec Summary: { {k:(round(v,4) if isinstance(v,(int,float)) and not pd.isna(v) else v) for k,v in exec_summary.items()} }")

        with tab2:
            st.subheader("Portfolio vs Benchmarks")
            p1 = p_cum_vs_bench(idx_file, bench_prices, h=chart_height, rangeslider=rangeslider)
            render_plotly(p1, "ข้อมูลยังไม่พอสำหรับพล็อต")

            st.subheader("Drawdown")
            p2 = p_drawdown(idx_file, h=max(chart_height-60, 320))
            render_plotly(p2, "ข้อมูลยังไม่พอสำหรับพล็อต Drawdown")

            st.subheader("Risk Contribution (current weights)")
            p3 = p_rc_bar(metrics, "RC_File", "Risk Contribution — File Weights", h=max(chart_height-80, 320))
            render_plotly(p3, "ไม่มีค่าที่จะพล็อต (RC_File ว่าง)")

            st.subheader("Risk Contribution — Current vs IVP (side-by-side)")
            p3b = p_rc_compare(metrics, h=max(chart_height-80, 320))
            render_plotly(p3b, "ไม่มีค่าที่จะพล็อต (RC_File/RC_IVP ว่าง)")

            st.subheader("Risk–Return Bubble")
            p4 = p_risk_return_bubble(metrics, h=chart_height, show_labels=show_labels)
            render_plotly(p4, "ยังไม่พอสำหรับพล็อต Risk–Return")

            st.subheader("Correlation Heatmap")
            p5 = p_corr_heatmap(r_assets, h=chart_height)
            render_plotly(p5, "ยังไม่พอสำหรับ Heatmap")

            # ---- Extra charts
            st.subheader("Rolling 60D Annualized Volatility")
            p6 = p_rolling_vol(rP, rB, h=360)
            render_plotly(p6, "ยังไม่พอสำหรับพล็อต Rolling Vol")

            st.subheader("Rolling 60D Correlation (Port vs Bench)")
            p7 = p_rolling_corr(rP, rB, h=360)
            render_plotly(p7, "ยังไม่พอสำหรับพล็อต Rolling Corr")

            st.subheader("Distribution of Daily Returns (with VaR/CVaR)")
            p8 = p_ret_histogram(rP, VaR=VaR, CVaR=CVaR, h=360)
            render_plotly(p8, "ยังไม่พอสำหรับ Histogram")

            st.subheader("Monthly Return Heatmap")
            p9 = p_monthly_heatmap(rP, h=420)
            render_plotly(p9, "ยังไม่พอสำหรับ Monthly Heatmap")

        with tab3:
            st.subheader("Market State")
            st.json({"State":state, **state_detail})

            # Indicator explanations
            st.subheader("How to read the indicators")
            st.markdown("""
- **BAML High Yield OAS (BAMLH0A0HYM2)** — สเปรดเครดิต: ~**<4.5%** = สงบ / **4.5–6%** = เฝ้าระวัง / **>6%** = ตึงเครียด (เสี่ยง Risk-off)
- **Yield Curve 10y–2y (T10Y2Y)** — **<0%** = อินเวิร์ส (สัญญาณชะลอ) / **>0%** = ปกติ
- **2Y Treasury (DGS2)** — ยิ่งสูง→เงื่อนไขการเงินตึงตัว (ประมาณ >**4–5%** ถือว่าตึง)
- **Fed Funds** — นโยบายการเงิน: สูงนาน = ตึง / ลดลง = ผ่อน
- **ISM PMI (NAPM)** — **>50** ขยายตัว / **<50** หดตัว / **<45** เสี่ยงถดถอย
- **Core PCE YoY** — เป้าหมายใกล้ **2%** ดีกับ Risk-on; **>3%** = เสี่ยงเงินเฟ้อเหนียว
- **Sahm Rule (SAHMREALTIME)** — **≥0.5** จุด = สัญญาณถดถอยเริ่มต้น
- **VIX percentile (3y)** — **<20%** สงบ / **20–80%** ปกติ / **>80%** ตึงเครียด
            """)

            st.subheader("VaR / CVaR")
            pass_var = (not np.isnan(VaR)) and (VaR <= var_limit_pct/100.0)
            pass_cvar= (not np.isnan(CVaR)) and (CVaR <= (var_limit_pct/100.0)*1.5)
            verdict = pass_var and pass_cvar
            if verdict:
                st.success(f"PASS ✅  VaR={VaR:.2%}, CVaR={CVaR:.2%}  | Limit={var_limit_pct:.2f}% (CVaR limit {var_limit_pct*1.5:.2f}%)")
            else:
                st.error(f"FAIL ❌  VaR={VaR:.2%}, CVaR={CVaR:.2%}  | Limit={var_limit_pct:.2f}% (CVaR limit {var_limit_pct*1.5:.2f}%)")

        with tab4:
            st.subheader("Download files")

            sheets = {
                "Inputs": pd.DataFrame({"Param":["PrimaryBench","RF(%)","TargetVol","TargetBeta","RC_cap","Period","VaR_method","VaR_alpha","VaR_limit(%)"],
                                        "Value":[primary_bench,RISK_FREE_PCTD,target_vol,target_beta,rc_cap,period,var_method,var_alpha,var_limit_pct]}).set_index("Param"),
                "Asset_Metrics": metrics,
                "TradePlan_Targets": tp,
                "TradePlan_Exec": tp_exec,
                "Exec_Summary": pd.DataFrame([exec_summary]).T.rename(columns={0:"Value"}),
                "MarketState": pd.DataFrame([{"State":state, **state_detail}]).T.rename(columns={0:"Value"}),
                "Cumulative_Index": pd.Series(idx_file, name="Port_File").to_frame(),
                "NAV_Tracker": (pd.Series(idx_file)/pd.Series(idx_file).dropna().iloc[0]*REBASE_NAV).rename("NAV").to_frame(),
                "VaR_CVaR": pd.DataFrame([{"alpha":var_alpha,"method":var_method,"VaR":VaR,"CVaR":CVaR,"Limit(%)":var_limit_pct}]).T.rename(columns={0:"Value"})
            }
            excel_bytes = to_excel_bytes(sheets)
            st.download_button("⬇️ Download Excel Report", data=excel_bytes.getvalue(),
                               file_name=f"DRP_Full_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            figs_zip = figures_to_zip_html({
                "cum_port_vs_bench": p1,
                "drawdown_port": p2,
                "rc_file": p3,
                "rc_compare": p3b,
                "risk_return_bubble": p4,
                "corr_heatmap": p5,
                "rolling_vol": p6,
                "rolling_corr": p7,
                "ret_hist": p8,
                "monthly_heatmap": p9,
            })
            st.download_button("⬇️ Download Charts (ZIP, interactive HTML)", data=figs_zip.getvalue(),
                               file_name=f"figs_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                               mime="application/zip")

        st.success("Finished ✅")

    except Exception as e:
        with tab1:
            st.exception(e)
else:
    with tab1:
        st.info("อัปโหลดไฟล์ แล้วกด ▶️ Run Analysis ทางซ้าย เพื่อเริ่มวิเคราะห์")

