# streamlit_app.py
# Downside-Protection / IVP Analyzer — clean layout + gauges + auto-advice
# - Upload CSV/XLSX (templates included)
# - yfinance prices
# - Metrics: AnnVol, Sharpe, MaxDD, CAGR, Beta, IVP weights, RC (current & IVP)
# - Extra charts (rolling vol/corr, histogram VaR/CVaR, monthly heatmap)
# - Market & Risk: interactive gauges + single PASS/FAIL verdict (VaR/CVaR)
# - Auto-advice when CVaR/VAR ratio fails + high correlation concentration
# - Export Excel + ZIP of interactive HTML charts (no Kaleido/Chrome needed)
# - Prevent "blank page after download" using session_state.auto_run

import os, io, re, math, time, warnings, zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="IVP / DRP Analyzer", layout="wide")

# ---------- Style ----------
PADDING = 24
st.markdown(f"""
<style>
  .block-container {{
    padding: {PADDING}px {PADDING}px;
    max-width: 1600px;
  }}
  .stPlotlyChart {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Defaults ----------
SHEET_NAME     = "Portfolio"
SYMBOL_COL_D   = "Symbol"
WEIGHT_COL_D   = None
QTY_COL_D      = "Qty"
PRICE_COL_D    = "Mkt Price(USD)"
PRIMARY_BENCH  = "^GSPC"
MORE_BENCHES_D = "^GSPC,GLD,BTC-USD,^NDX"
PERIOD_D       = "5y"
RISK_FREE_PCTD = 4.0
TARGET_BETA_D  = 1.00
RC_CAP_D       = 0.15
BAD_MAXDD_D    = -0.40
BAD_SHARPE_D   = 0.0
VAR_ALPHA_D    = 0.95
VAR_METHOD_D   = "hist"      # 'hist'|'normal'|'cornish'
VAR_LIMIT_PCT  = 2.0         # daily limit for VaR; CVaR limit = 1.5x
CVaR_VaR_MAX   = 1.6         # typical tail ratio ceiling (warn if above)
INDEX_MULT     = 200.0
ETF_FALLBACK   = {"^GSPC":"SPY","SPY":"SPY","^NDX":"QQQ","QQQ":"QQQ","^IXIC":"QQQ"}
METHOD_ALIGN   = "intersect"
REBASE_NAV     = 100.0

# ---------- Session ----------
if "auto_run" not in st.session_state: st.session_state.auto_run = False
if "run_seq"  not in st.session_state: st.session_state.run_seq  = 0

# ---------- Utils ----------
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
    base = s.iloc[-1]/s.iloc[0]
    return np.nan if years<=0 or base<=0 else float(base**(1.0/years)-1.0)

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

# ---------- Data IO ----------
@st.cache_data(show_spinner=False)
def read_portfolio(uploaded_file, sym_col, w_col, qty_col, px_col, sheet_name=SHEET_NAME):
    if uploaded_file is None: raise ValueError("กรุณาอัปโหลดพอร์ต (CSV/XLSX)")
    if uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    else:
        df = pd.read_csv(uploaded_file)
    if sym_col not in df.columns: raise ValueError(f"ไม่พบคอลัมน์ '{sym_col}'")
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
    g = g[g>0]; 
    if g.empty: raise ValueError("น้ำหนักรวมเป็นศูนย์")
    return g.index.tolist(), (g/g.sum())

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

# ---------- Core Analytics ----------
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
        s   = pd.Series(x).skew(); k = pd.Series(x).kurt()
        from scipy.stats import norm
        z   = norm.ppf(alpha)
        zcf = (z + (s/6)*(z**2-1) + (k/24)*(z**3-3*z) - (s**2/36)*(2*z**3-5*z))
        cvar = -(mu - sd * (np.exp(-0.5*zcf*zcf)/np.sqrt(2*np.pi))/(1-alpha))
        var  = -(mu + sd*zcf)
    else:
        raise ValueError("method must be 'hist'|'normal'|'cornish'")
    return float(var), float(cvar)

def build_trade_plan(metrics, rc_cap=0.15, forbid_bad_buy=True, bad_mdd=-0.40, bad_sharpe=0.0):
    df = metrics.copy()
    tgt = df["File_Weight"].copy()

    # cap RC
    over = df["RC_File"] > rc_cap
    scale = (rc_cap/df["RC_File"]).clip(upper=1.0)
    tgt.loc[over] = (tgt.loc[over] * scale.loc[over]).values

    # penalize weak risk/return
    bad = (df["Sharpe"]<bad_sharpe) | (df["MaxDD"]<bad_mdd)
    tgt.loc[bad] = tgt.loc[bad]*0.5

    # reward diversifiers
    good = (df["Beta"]<0.6) & (df["CAGR"]>0)
    tgt.loc[good] = tgt.loc[good]*1.2

    # nudge towards IVP
    tgt = tgt + 0.25*(df["IVP_Weight"] - tgt)

    if forbid_bad_buy:  # don't increase allocation to flagged names
        tgt.loc[bad] = np.minimum(tgt.loc[bad], df.loc[bad,"File_Weight"])

    tgt = tgt.clip(lower=0).fillna(0.0); tgt = tgt/(tgt.sum() if tgt.sum()!=0 else 1.0)

    delta  = tgt - df["File_Weight"]
    action = np.where(delta>0.002, "BUY", np.where(delta<-0.002, "SELL", "KEEP"))
    reason=[]
    for t in df.index:
        rs=[]
        if t in over.index and bool(over.loc[t]): rs.append(f"RC>{int(rc_cap*100)}%")
        if t in bad.index  and bool(bad.loc[t]):  rs.append("Sharpe<0/DeepDD")
        if t in good.index and bool(good.loc[t]): rs.append("Diversifier")
        if not rs: rs.append("Rebalance→IVP")
        reason.append(", ".join(rs))

    return pd.DataFrame({"TargetWeight":tgt, "Delta":delta, "Action":action, "Reason":reason}, index=df.index)

def apply_trade_constraints(tp, last_px, pv, cash_now_pct, cash_tgt_pct, min_ticket, lot_func):
    tp = tp.copy()
    tp["Price"] = pd.Series(last_px).reindex(tp.index).astype(float)
    tp["TradeValue_raw"] = tp["Delta"] * pv

    cash_now  = max(0.0, min(1.0, float(cash_now_pct))) * pv
    cash_tgt  = max(0.0, min(1.0, float(cash_tgt_pct))) * pv

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

    summary = {"PV": pv, "Cash_after_pct": cash_after_pct, "Scaled_buys_factor": scale}
    return tp, summary

# ---------- Plotly ----------
import plotly.graph_objects as go
import plotly.io as pio

def _layout(title, h=420, rangeslider=False):
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

def render_plotly(fig, fallback="ยังไม่พอสำหรับพล็อต"):
    if fig is None: st.info(fallback); return
    try: st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except Exception as e:
        st.warning("แสดงกราฟไม่สำเร็จ"); 
        with st.expander("รายละเอียด error"): st.exception(e)

# --- core charts
def p_cum_vs_bench(idx_port, bench_prices, h=420, rangeslider=True):
    s = pd.Series(idx_port).dropna()
    if s.size < 2: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=(s/s.iloc[0]).values, mode="lines", name="Portfolio",
                             hovertemplate="%{x|%Y-%m-%d}<br>Idx=%{y:.3f}<extra></extra>"))
    for name, px in bench_prices.items():
        p = pd.Series(px).dropna()
        if p.size >= 2:
            fig.add_trace(go.Scatter(x=p.index, y=(p/p.iloc[0]).values, mode="lines", name=name))
    fig.update_layout(**_layout("Cumulative Index: Portfolio vs Benchmarks", h=h, rangeslider=rangeslider))
    fig.update_yaxes(title="Index (rebased=1.0)")
    return fig

def p_drawdown(idx_port, h=380):
    s = pd.Series(idx_port).dropna()
    if s.size < 2: return None
    dd = s/s.cummax() - 1.0
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
    fig.update_layout(**_layout("Portfolio Drawdown", h=h))
    fig.update_yaxes(title="Drawdown")
    return fig

def p_rc_bar(metrics, col="RC_File", title=None, h=360):
    if metrics is None or col not in metrics.columns: return None
    ser = metrics[col].copy().astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    if ser.empty: return None
    ser = ser.sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=ser.index, y=ser.values))
    fig.update_layout(**_layout(title or f"Risk Contribution — {col}", h=h))
    fig.update_yaxes(title="Contribution")
    return fig

def p_rc_compare(metrics, h=360, cols=("RC_File","RC_IVP")):
    if metrics is None or not set(cols).issubset(metrics.columns): return None
    df = metrics[list(cols)].copy().astype(float).dropna(how="all")
    if df.empty: return None
    order = df["RC_File"].fillna(-999).sort_values(ascending=False).index
    df = df.reindex(order)
    fig = go.Figure()
    for c in cols:
        ser = df[c].dropna()
        fig.add_trace(go.Bar(x=ser.index, y=ser.values, name=("Current RC" if c=="RC_File" else "IVP RC")))
    fig.update_layout(**_layout("Risk Contribution — Current vs IVP (side-by-side)", h=h), barmode="group")
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
        marker=dict(size=size)
    ))
    fig.update_layout(**_layout("Risk–Return Map (bubble = current weight)", h=h))
    fig.update_xaxes(title="AnnVol"); fig.update_yaxes(title="CAGR")
    return fig

def p_corr_heatmap(r_assets, h=480):
    if r_assets is None or r_assets.empty: return None
    if r_assets.shape[1] > 80: r_assets = r_assets.iloc[:, :80]
    corr = r_assets.corr().replace([np.inf,-np.inf], np.nan)
    corr = corr.dropna(how="all").dropna(how="all", axis=1)
    if corr.empty: return None
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1))
    fig.update_layout(**_layout("Correlation Heatmap", h=h))
    return fig

# --- extra charts
def p_rolling_vol(rP, rB, h=360, win=60):
    if rP is None or rP.empty: return None
    k = 252.0
    rvP = (rP.rolling(win).std(ddof=0) * np.sqrt(k)).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rvP.index, y=rvP.values, mode="lines", name="Port 60D Vol"))
    if rB is not None and not rB.empty:
        rB = rB.reindex(rvP.index).dropna()
        rvB = (rB.rolling(win).std(ddof=0) * np.sqrt(k)).dropna()
        if not rvB.empty: fig.add_trace(go.Scatter(x=rvB.index, y=rvB.values, mode="lines", name="Bench 60D Vol"))
    fig.update_layout(**_layout("Rolling 60D Annualized Volatility", h=h))
    fig.update_yaxes(title="Ann. Vol")
    return fig

def p_rolling_corr(rP, rB, h=360, win=60):
    if rP is None or rB is None or rP.empty or rB.empty: return None
    rB = rB.reindex(rP.index).dropna(); rP = rP.reindex(rB.index).dropna()
    if rP.empty or rB.empty: return None
    rc = rP.rolling(win).corr(rB).dropna()
    fig = go.Figure(go.Scatter(x=rc.index, y=rc.values, mode="lines", name=f"Corr {win}D"))
    fig.update_layout(**_layout(f"Rolling {win}D Correlation (Port vs Bench)", h=h))
    fig.update_yaxes(title="Correlation", range=[-1,1])
    return fig

def p_ret_histogram(rP, VaR=None, CVaR=None, bins=60, h=360):
    if rP is None or rP.empty: return None
    fig = go.Figure()
    fig.add_histogram(x=rP.values, nbinsx=bins, name="Daily Returns")
    if VaR is not None and not np.isnan(VaR):  fig.add_vline(x=-VaR,  line_dash="dash", annotation_text=f"-VaR {VaR:.2%}")
    if CVaR is not None and not np.isnan(CVaR): fig.add_vline(x=-CVaR, line_dash="dot",  annotation_text=f"-CVaR {CVaR:.2%}")
    fig.update_layout(**_layout("Distribution of Daily Returns (with VaR/CVaR)", h=h))
    fig.update_xaxes(title="Daily Return"); fig.update_yaxes(title="Frequency")
    return fig

def p_monthly_heatmap(rP, h=420):
    if rP is None or rP.empty or not isinstance(rP.index, pd.DatetimeIndex): return None
    m = rP.resample("M").apply(lambda s: (1+s).prod()-1).dropna()
    if m.empty: return None
    df = pd.DataFrame({"Year": m.index.year, "Month": m.index.month, "Ret": m.values})
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = df.pivot(index="Year", columns="Month", values="Ret").reindex(columns=range(1,13))
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=[months[i-1] for i in pivot.columns], y=pivot.index,
                                    zmin=-0.20, zmax=0.20, colorscale="RdBu", reversescale=True))
    fig.update_layout(**_layout("Monthly Return Heatmap", h=h))
    return fig

# ---------- Gauges ----------
def make_gauge(title, value, vmin, vmax, zones, fmt="auto", h=260):
    if value is None or (isinstance(value, float) and np.isnan(value)): return None
    if fmt == "pct" or fmt == "pct1": disp = f"{value:.1%}"
    elif fmt == "bp": disp = f"{value:.2f}%"
    else: disp = f"{value:.3f}"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix":"", "font":{"size":18}},
        title={"text": title, "font":{"size":14}},
        gauge={
            "axis":{"range":[vmin, vmax]},
            "bar":{"color":"#2563eb"},
            "steps":[
                {"range":[vmin, zones[0][0]], "color": zones[0][1]},
                *[
                    {"range":[zones[i-1][0], zones[i][0]], "color": zones[i][1]}
                    for i in range(1, len(zones))
                ]
            ],
        }
    ))
    fig.update_layout(margin=dict(l=18,r=18,t=38,b=10), height=h)
    return fig

def judge_text(name, v):
    def lvl(low, mid, good_is_high=True):
        if np.isnan(v): return "ไม่มีข้อมูล"
        if good_is_high:
            return "อ่อน/เสี่ยง" if v<low else ("กลาง" if v<mid else "แข็ง/เอื้อ")
        return "ดี/เสี่ยงต่ำ" if v<low else ("กลาง" if v<mid else "สูง/ระวัง")
    if name == "trend_score":              return lvl(0.55, 0.70, True)  + " (แนวโน้ม)"
    if name == "risk_score":               return lvl(0.30, 0.60, False) + " (ความเสี่ยงรวม)"
    if name == "breadth_5ETF_%>200D":      return "แข็ง" if v>=0.7 else ("กลาง" if v>=0.4 else "อ่อน")
    if name == "vix_pctile_3y":            return "สงบ" if v<0.2 else ("ปกติ" if v<=0.8 else "ตึงเครียด")
    if name == "hy_spread_risk":           return lvl(0.30, 0.60, False) + " (เครดิต)"
    if name == "curve_10y2y":              return "ปกติ" if v>0 else "โค้งกลับหัว (เสี่ยง)"
    if name == "pmi_score":                return "ขยายตัว" if v>0.55 else ("ทรงตัว" if v>=0.45 else "หดตัว")
    if name == "infl_risk_corePCE":        return lvl(0.40, 0.70, False) + " (เงินเฟ้อ)"
    return ""

# ---------- Export helpers ----------
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
    bio.seek(0); return bio

# ---------- Sidebar ----------
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
target_beta=st.sidebar.number_input("Target β", min_value=0.0, value=TARGET_BETA_D, step=0.05)
rc_cap    = st.sidebar.number_input("RC cap (share)", min_value=0.05, value=RC_CAP_D, step=0.01)
bad_mdd   = st.sidebar.number_input("Bad MaxDD (≤)", value=BAD_MAXDD_D, step=0.05, format="%.2f")
bad_shp   = st.sidebar.number_input("Bad Sharpe (≤)", value=BAD_SHARPE_D, step=0.1, format="%.2f")

st.sidebar.markdown("---")
var_alpha = st.sidebar.slider("VaR/CVaR α", 0.80, 0.995, VAR_ALPHA_D, 0.005)
var_method= st.sidebar.selectbox("VaR method", ["hist","normal","cornish"], index=["hist","normal","cornish"].index(VAR_METHOD_D))
var_limit_pct = st.sidebar.number_input("VaR daily limit (%)", min_value=0.1, value=VAR_LIMIT_PCT, step=0.1,
                                        help="เกณฑ์ผ่าน/ไม่ผ่าน: VaR ≤ limit นี้ และ CVaR ≤ 1.5×limit")

st.sidebar.markdown("---")
fred_key  = st.sidebar.text_input("FRED API Key (optional)", value=os.environ.get("FRED_API_KEY",""))
show_market_json     = st.sidebar.toggle("Show Market JSON (raw)", value=False)
show_indicator_guide = st.sidebar.toggle("Show Indicator Guide", value=True)
toast_on_var_fail    = st.sidebar.toggle("Pop-up on VaR fail", value=True)

st.sidebar.markdown("---")
chart_height = st.sidebar.slider("Chart height (px)", min_value=320, max_value=900, value=460, step=20)
show_labels  = st.sidebar.toggle("Show labels on bubble chart", value=True)
rangeslider  = st.sidebar.toggle("Show time-range slider (cum chart)", value=True)

run_btn = st.sidebar.button("▶️ Run Analysis", use_container_width=True)
if run_btn:
    st.session_state.auto_run = True
    st.session_state.run_seq += 1

# ---------- Title & How-to ----------
st.title("IVP Analyzer")
st.caption("Inverse Volatility Portfolio Analyzer")

def howto_tab():
    st.subheader("ไฟล์ที่อัปโหลดต้องมีอะไรบ้าง?")
    st.markdown("""
**ขั้นต่ำ**: คอลัมน์ `Symbol`  
**เลือกอย่างใดอย่างหนึ่งเพื่อคำนวณน้ำหนัก**  
1) `Weight` (0–1 หรือ 0–100%) — กำหนดใน Settings ว่าชื่อคอลัมน์คืออะไร  
2) `Qty` + `Mkt Price(USD)`  
3) `Mkt Val`
> ชื่อคอลัมน์ทั้งหมดปรับได้ในแถบ Settings
""")
    sample = pd.DataFrame({"Symbol":["AAPL","MSFT","GLD","BTC-USD"],"Qty":[10,5,3,0.05],"Mkt Price(USD)":[200,420,185,60000]})
    st.download_button("⬇️ CSV Template", sample.to_csv(index=False).encode(), "portfolio_template.csv", "text/csv")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw: sample.to_excel(xw, sheet_name=SHEET_NAME, index=False)
    bio.seek(0)
    st.download_button("⬇️ XLSX Template", bio.getvalue(), "portfolio_template.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- Tabs ----------
tab0, tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ How to Use","📊 Metrics & Plan","📈 Charts","🧭 Market & Risk","📦 Export"])
with tab0: howto_tab()

# ---------- Run ----------
should_run = run_btn or st.session_state.auto_run
if should_run and uploaded:
    try:
        with st.spinner("Loading portfolio..."):
            tickers, w_file = read_portfolio(uploaded, sym_col, (weight_col or None), qty_col, price_col, sheet_name)
        benches = [s.strip() for s in more_benches.split(",") if s.strip()]
        all_syms = list(dict.fromkeys(tickers + ([primary_bench] if primary_bench not in benches else []) + benches))
        with st.spinner("Downloading prices..."):
            px_all = yf_download(all_syms, period=period)
        if primary_bench not in px_all.columns:
            with tab1: st.error(f"ไม่พบ {primary_bench} ในข้อมูลราคา"); st.stop()

        miss = [t for t in tickers if t not in px_all.columns]
        if miss:
            tickers = [t for t in tickers if t in px_all.columns]
            w_file  = w_file.reindex(tickers).fillna(0.0)
            w_file  = w_file/(w_file.sum() if w_file.sum()!=0 else 1.0)
            with tab1: st.warning(f"ตัดออก (ไม่มีราคา): {', '.join(miss)}")

        px_assets = px_all[tickers]
        primary_price = px_all[primary_bench].rename(primary_bench)

        metrics, r_assets, r_bench, idx_bench, cov_an, ivp_w, w_file = compute_metrics(px_assets, primary_price, w_file, RISK_FREE_PCTD)
        idx_file = (r_assets.mul(w_file, axis=1).sum(axis=1)).add(1).cumprod()
        rP = idx_file.pct_change().dropna()
        rB = r_bench.reindex(rP.index).dropna()
        betaP = beta_vs(rP.reindex(rB.index), rB) if len(rB) else np.nan

        VaR, CVaR = var_cvar(rP, alpha=var_alpha, method=var_method)
        cvar_var_ratio = (CVaR / VaR) if (VaR and not np.isnan(VaR) and not np.isnan(CVaR) and VaR>0) else np.nan

        tp = build_trade_plan(metrics, rc_cap=rc_cap, forbid_bad_buy=True, bad_mdd=bad_mdd, bad_sharpe=bad_shp)
        last_px = latest_prices(tp.index.tolist())
        tp_exec, exec_summary = apply_trade_constraints(tp, last_px, pv,
                            normalize_pct(cash_now, 0.10), normalize_pct(cash_tgt, 0.15),
                            min_ticket, lot_func=None)

        # ---- Correlation concentration quick stats
        corr = r_assets.corr().replace([np.inf,-np.inf], np.nan)
        avg_offdiag = np.nan
        max_pair = np.nan
        if not corr.empty:
            mask = ~np.eye(len(corr), dtype=bool)
            vals = corr.values[mask]
            if vals.size: avg_offdiag = float(np.nanmean(vals)); max_pair = float(np.nanmax(vals))
        hhi = float(np.sum(np.square(w_file.values))) if len(w_file) else np.nan  # Herfindahl of weights

        # ---- Tabs content
        with tab1:
            st.subheader("Asset Metrics")
            st.dataframe(metrics.round(4), use_container_width=True)

            st.subheader("Trade Plan (Executed Constraints)")
            st.dataframe(tp_exec.round(4), use_container_width=True)
            st.caption(f"Summary: { {k:(round(v,4) if isinstance(v,(int,float)) and not pd.isna(v) else v) for k,v in exec_summary.items()} }")

        bench_prices = {b:px_all[b] for b in benches if b in px_all.columns}
        with tab2:
            st.subheader("Portfolio vs Benchmarks")
            p1 = p_cum_vs_bench(idx_file, bench_prices, h=chart_height, rangeslider=rangeslider); render_plotly(p1)

            st.subheader("Drawdown"); render_plotly(p_drawdown(idx_file, h=max(chart_height-60, 320)))
            st.subheader("Risk Contribution (current weights)"); p3 = p_rc_bar(metrics,"RC_File","Risk Contribution — File Weights", h=max(chart_height-80, 320)); render_plotly(p3)
            st.subheader("Risk Contribution — Current vs IVP (side-by-side)"); p3b = p_rc_compare(metrics, h=max(chart_height-80, 320)); render_plotly(p3b)
            st.subheader("Risk–Return Bubble"); render_plotly(p_risk_return_bubble(metrics, h=chart_height, show_labels=show_labels))
            st.subheader("Correlation Heatmap"); render_plotly(p_corr_heatmap(r_assets, h=chart_height))
            st.subheader("Rolling 60D Annualized Volatility"); render_plotly(p_rolling_vol(rP, rB, h=360))
            st.subheader("Rolling 60D Correlation (Port vs Bench)"); render_plotly(p_rolling_corr(rP, rB, h=360))
            st.subheader("Distribution of Daily Returns (with VaR/CVaR)"); render_plotly(p_ret_histogram(rP, VaR=VaR, CVaR=CVaR, h=360))
            st.subheader("Monthly Return Heatmap"); render_plotly(p_monthly_heatmap(rP, h=420))

        # ---------- Market & Risk ----------
        # FRED summary (lightweight) — only last values with guide; fetch lazily when API key provided
        def fred_fetch(series, start="2005-01-01", key=None, retries=2, timeout=10):
            key = (key or os.environ.get("FRED_API_KEY","")).strip().strip('"').strip("'")
            if not key: return pd.Series(dtype=float, name=series), {"ok":False,"reason":"no_api_key"}
            url = "https://api.stlouisfed.org/fred/series/observations"
            p = {"series_id":series,"observation_start":start,"file_type":"json","api_key":key}
            last_err=None
            for _ in range(retries+1):
                try:
                    resp = requests.get(url, params=p, timeout=timeout)
                    if resp.status_code!=200:
                        last_err=f"http_{resp.status_code}"; time.sleep(0.4); continue
                    js = resp.json().get("observations",[])
                    data = pd.Series({pd.to_datetime(i["date"]):(float(i["value"]) if i["value"]!="." else np.nan) for i in js}).sort_index()
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
            return df, core_yoy

        def market_state(primary_bench_price, fred_key=None):
            bench_price = pd.Series(primary_bench_price).dropna()
            ma200 = bench_price.rolling(200).mean()
            above200 = 1.0 if (len(ma200.dropna()) and bench_price.iloc[-1]>ma200.iloc[-1]) else 0.0
            slope200 = 0.0
            if len(ma200.dropna())>22 and ma200.iloc[-1]>0:
                slope200 = float((ma200.iloc[-1]-ma200.shift(20).iloc[-1])/ma200.iloc[-1])
            try:
                etfs = ["SPY","QQQ","IWM","EFA","EEM"]; px_etf = yf_download(etfs, period="3y"); breadth = (px_etf.apply(lambda s: s.iloc[-1] > s.rolling(200).mean().iloc[-1])).mean()
            except Exception: breadth = np.nan
            trend_score = np.nanmean([above200, np.tanh(10*slope200), breadth])

            try: vix = yf_download(["^VIX"], period="3y")["^VIX"].dropna(); vix_pct = float(vix.rank(pct=True).iloc[-1])
            except Exception: vix_pct = np.nan

            fred_raw, core_yoy = fred_summary(key=fred_key)
            credit_risk = np.nan
            if "BAMLH0A0HYM2" in fred_raw:
                ser = fred_raw["BAMLH0A0HYM2"].dropna(); hy = ser.tail(min(len(ser), 3*252))
                if hy.size > 10 and hy.std() > 0:
                    z_hy = float((hy.iloc[-1] - hy.mean()) / hy.std()); credit_risk = 1.0 / (1.0 + np.exp(-z_hy))
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
            return state, detail

        state, state_detail = market_state(primary_price, fred_key if fred_key else None)

        with tab3:
            st.subheader("Market State")
            if show_market_json: st.json({"State":state, **state_detail})

            st.markdown("### Gauges")
            vals = {
                "trend_score": state_detail.get("trend_score", np.nan),
                "risk_score": state_detail.get("risk_score", np.nan),
                "breadth_5ETF_%>200D": state_detail.get("breadth_5ETF_%>200D", np.nan),
                "vix_pctile_3y": state_detail.get("vix_pctile_3y", np.nan),
                "hy_spread_risk": state_detail.get("hy_spread_risk", np.nan),
                "curve_10y2y": state_detail.get("curve_10y2y", np.nan),
                "pmi_score": state_detail.get("pmi_score", np.nan),
                "infl_risk_corePCE": state_detail.get("infl_risk_corePCE", np.nan),
            }

            rows = [
                [("Trend",   make_gauge("Trend score", vals["trend_score"], 0, 1, [(0.55,"#ff4b4b"),(0.70,"#f7c948"),(1.0,"#10b981")], fmt="pct1"), ("trend_score", vals["trend_score"])),
                 ("Risk",    make_gauge("Risk score",  vals["risk_score"],  0, 1, [(0.30,"#10b981"),(0.60,"#f7c948"),(1.0,"#ff4b4b")], fmt="pct1"), ("risk_score", vals["risk_score"])),
                 ("Breadth", make_gauge("%>200D (5 ETFs)", vals["breadth_5ETF_%>200D"], 0, 1, [(0.40,"#ff4b4b"),(0.70,"#f7c948"),(1.0,"#10b981")], fmt="pct1"), ("breadth_5ETF_%>200D", vals["breadth_5ETF_%>200D"]))],
                [("VIX", make_gauge("VIX percentile (3y)", vals["vix_pctile_3y"], 0, 1, [(0.20,"#10b981"),(0.80,"#f7c948"),(1.0,"#ff4b4b")], fmt="pct1"), ("vix_pctile_3y", vals["vix_pctile_3y"])),
                 ("HY",  make_gauge("HY spread risk", vals["hy_spread_risk"], 0, 1, [(0.30,"#10b981"),(0.60,"#f7c948"),(1.0,"#ff4b4b")], fmt="pct1"), ("hy_spread_risk", vals["hy_spread_risk"])),
                 ("Curve", make_gauge("10y-2y", vals["curve_10y2y"], -1.0, 2.0, [(0.0,"#ff4b4b"),(0.5,"#f7c948"),(2.0,"#10b981")], fmt="bp"), ("curve_10y2y", vals["curve_10y2y"]))],
                [("PMI",  make_gauge("PMI score", vals["pmi_score"], 0, 1, [(0.45,"#ff4b4b"),(0.55,"#f7c948"),(1.0,"#10b981")], fmt="pct1"), ("pmi_score", vals["pmi_score"])),
                 ("Infl.",make_gauge("Inflation risk (Core PCE)", vals["infl_risk_corePCE"], 0, 1, [(0.40,"#10b981"),(0.70,"#f7c948"),(1.0,"#ff4b4b")], fmt="pct1"), ("infl_risk_corePCE", vals["infl_risk_corePCE"])),
                 (None, None, (None, None))]
            ]
            for row in rows:
                cols = st.columns(3, vertical_alignment="top")
                for col, item in zip(cols, row):
                    label, fig, keypair = item
                    name, val = keypair
                    if fig is None: col.empty()
                    else:
                        with col:
                            render_plotly(fig)
                            if name: st.caption("**"+judge_text(name, val)+"**")

            with st.expander("How to read the indicators", expanded=show_indicator_guide):
                st.markdown("""- **BAML High Yield OAS (BAMLH0A0HYM2)** — ~**<4.5%** สงบ / **4.5–6%** เฝ้าระวัง / **>6%** ตึงเครียด
- **Yield Curve 10y–2y (T10Y2Y)** — **<0%** อินเวิร์ส / **>0%** ปกติ
- **2Y Treasury (DGS2)** — สูง = เงื่อนไขการเงินตึง (**>4–5%** ตึง)
- **Fed Funds** — สูงนาน = ตึง / ลดลง = ผ่อน
- **ISM PMI (NAPM)** — **>50** ขยายตัว / **<50** หดตัว / **<45** เสี่ยงถดถอย
- **Core PCE YoY** — ใกล้ **2%** เอื้อ Risk-on; **>3%** เสี่ยงเงินเฟ้อเหนียว
- **Sahm Rule** — **≥0.5** จุด = สัญญาณถดถอยเริ่ม
- **VIX percentile (3y)** — **<20%** สงบ / **20–80%** ปกติ / **>80%** ตึงเครียด""")

            # ---- VaR/CVaR verdict (single place)
            st.subheader("VaR / CVaR")
            pass_var  = (not np.isnan(VaR))  and (VaR  <= var_limit_pct/100.0)
            pass_cvar = (not np.isnan(CVaR)) and (CVaR <= (var_limit_pct/100.0)*1.5)
            verdict = pass_var and pass_cvar
            msg = f"VaR={VaR:.2%}, CVaR={CVaR:.2%} | Limit={var_limit_pct:.2f}% (CVaR limit {(var_limit_pct*1.5):.2f}%)"
            (st.success if verdict else st.error)(("PASS ✅  " if verdict else "FAIL ❌  ")+msg)
            if not verdict and toast_on_var_fail:
                k = f"var_toast_{st.session_state.run_seq}"
                if not st.session_state.get(k):
                    st.toast("VaR/CVaR ไม่ผ่านเกณฑ์", icon="⚠️"); st.session_state[k]=True

            # ---- Recommendations (tail risk & concentration)
            st.subheader("Recommendations")
            bullet = []
            if not verdict:
                if not pass_var:  bullet.append("ลด **VaR**: ลดเลเวอเรจ/ลดน้ำหนักสินทรัพย์ที่ผันผวนสูง, เพิ่มสัดส่วนพันธบัตรระยะสั้น/เงินสด 5–15%")
                if not pass_cvar: bullet.append("ลด **CVaR (หางหนัก)**: กระจายไปสินทรัพย์ป้องกันความเสี่ยง (เช่น **Gold 5–10%**, **IG Bonds 10–20%**), พิจารณา hedge บางส่วน (เช่น SPY/QQQ หรือ put spread)")
                if not np.isnan(cvar_var_ratio) and cvar_var_ratio>CVaR_VaR_MAX:
                    bullet.append(f"อัตราส่วน **CVaR/VaR ≈ {cvar_var_ratio:.2f}** สูง → หางหนา: หลีกเลี่ยงสินทรัพย์ tail-heavy (เช่น **Crypto/Small-Cap** มากเกินไป), ใช้ position limit ต่อสินทรัพย์ที่ 5–10% และ RC cap ที่ {int(rc_cap*100)}%")
            # correlation concentration
            if not np.isnan(avg_offdiag):
                if avg_offdiag >= 0.6 or (not np.isnan(max_pair) and max_pair>=0.85):
                    bullet.append("พอร์ต **กระจุกตัวเชิงสหสัมพันธ์**: เพิ่มสินทรัพย์ low-corr เช่น **Treasuries/IG Bonds**, **Gold**, **Commodity ex-energy**, หรือเพิ่ม **DM ex-US/Emerging**; ลดน้ำหนักกลุ่มที่เคลื่อนพร้อมกัน")
                elif avg_offdiag >= 0.4:
                    bullet.append("ความสัมพันธ์เฉลี่ยค่อนข้างสูง: ทยอยปรับให้สัดส่วนระหว่าง **Equity : Bond : Gold : Cash** ใกล้ 60:25:10:5 (ตัวอย่างแนวทาง)")
            if not np.isnan(hhi) and hhi>0.20:
                bullet.append("น้ำหนักกระจุก (HHI>0.20): กำหนด **position limit** ต่อชื่อ ~5–8% และ **RC cap** ~10–15%")

            if bullet:
                st.markdown("\n".join([f"- {b}" for b in bullet]))
            else:
                st.markdown("พอร์ตสมดุลดีอยู่แล้ว: คงวินัย rebalancing และเฝ้าระวัง infl./policy risk")

        # ---------- Export ----------
        with tab4:
            st.subheader("Download files")
            nav_df = (pd.Series(idx_file)/pd.Series(idx_file).dropna().iloc[0]*REBASE_NAV).rename("NAV").to_frame()
            sheets = {
                "Inputs": pd.DataFrame({"Param":["PrimaryBench","RF(%)","TargetBeta","RC_cap","Period","VaR_method","VaR_alpha","VaR_limit(%)"],
                                        "Value":[primary_bench,RISK_FREE_PCTD,target_beta,rc_cap,period,var_method,var_alpha,var_limit_pct]}).set_index("Param"),
                "Asset_Metrics": metrics,
                "TradePlan_Targets": tp,
                "TradePlan_Exec": tp_exec,
                "Summary": pd.DataFrame([{"VaR":VaR,"CVaR":CVaR,"CVaR/VaR":cvar_var_ratio,"AvgCorr":avg_offdiag,"MaxPairCorr":max_pair,"HHI":hhi}]).T.rename(columns={0:"Value"}),
                "NAV_Tracker": nav_df.reset_index()
            }
            excel_bytes = to_excel_bytes(sheets)
            st.download_button("⬇️ Excel Report", data=excel_bytes.getvalue(),
                               file_name=f"DRP_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            figs_zip = figures_to_zip_html({
                "cum_port_vs_bench": p1,
                "drawdown_port": p_drawdown(idx_file),
                "rc_file": p3,
                "rc_compare": p_rc_compare(metrics),
                "risk_return_bubble": p_risk_return_bubble(metrics),
                "corr_heatmap": p_corr_heatmap(r_assets),
                "rolling_vol": p_rolling_vol(rP, rB),
                "rolling_corr": p_rolling_corr(rP, rB),
                "ret_hist": p_ret_histogram(rP, VaR=VaR, CVaR=CVaR),
                "monthly_heatmap": p_monthly_heatmap(rP),
            })
            st.download_button("⬇️ Charts (ZIP, interactive HTML)", data=figs_zip.getvalue(),
                               file_name=f"figs_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                               mime="application/zip")

        st.success("Finished ✅")

    except Exception as e:
        with tab1: st.exception(e)
else:
    with tab1: st.info("อัปโหลดไฟล์ แล้วกด ▶️ Run Analysis ทางซ้าย")
