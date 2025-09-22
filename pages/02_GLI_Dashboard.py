# pages/02_GLI_Dashboard.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import gli_lib as gl

# =========================
# Page & Helpers
# =========================
st.set_page_config(page_title="GLI Dashboard", layout="wide")

def _fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x*100:.2f}%"

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" | ".join([str(s) for s in tup]).strip() for tup in out.columns]
    out.columns = [str(c) for c in out.columns]
    return out

def _to_display_df(x: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.Series):
        df = x.to_frame(name=str(x.name) if x.name is not None else "value")
    else:
        df = x.copy()
    df = _flatten_cols(df)
    # make index a string column (avoid pyarrow issues)
    if not isinstance(df.index, pd.RangeIndex):
        try:
            if isinstance(df.index, pd.MultiIndex):
                df.insert(0, "Index", df.index.map(lambda tup: " | ".join([str(v) for v in tup])))
            else:
                df.insert(0, "Index", df.index.astype(str))
            df = df.reset_index(drop=True)
        except Exception:
            df = df.reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    return df

def _col_of(df: pd.DataFrame | None, logical_name: str) -> str | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cands = {
        "GLI_INDEX": ["GLI_INDEX", "GLI"],
        "NASDAQ":    ["NASDAQ", "^IXIC", "NDX", "NASDAQCOM"],
        "SP500":     ["SP500", "^GSPC", "S&P500", "SPX"],
        "GOLD":      ["GOLD", "GC=F", "XAUUSD", "GLD"],
        "BTC":       ["BTC", "BTC-USD"],
        "ETH":       ["ETH", "ETH-USD"],
    }.get(logical_name, [logical_name])
    cols_low = {str(c).lower(): str(c) for c in df.columns}
    for key in cands:
        k = str(key).lower()
        if k in cols_low:
            return cols_low[k]
    for c in df.columns:
        s = str(c).lower()
        if any(k.lower() in s for k in cands):
            return str(c)
    return None

def _pick_monthly_series(monthly: pd.DataFrame, logical_name: str) -> pd.Series:
    if not isinstance(monthly, pd.DataFrame) or monthly.empty:
        return pd.Series(dtype=float, name=logical_name)
    col = _col_of(monthly, logical_name)
    return monthly[col] if (col and col in monthly.columns) else pd.Series(dtype=float, name=logical_name)

def _pick_annual_series(annual: pd.DataFrame, monthly: pd.DataFrame, logical_name: str) -> pd.Series:
    col = _col_of(annual, logical_name) if isinstance(annual, pd.DataFrame) else None
    if col and col in getattr(annual, "columns", []):
        return annual[col]
    m = _pick_monthly_series(monthly, logical_name)
    if m.empty:
        return pd.Series(dtype=float, name=logical_name)
    return m.resample("A-DEC").last()

def _years_span(idx: pd.DatetimeIndex) -> float:
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return np.nan
    return (idx[-1] - idx[0]).days / 365.25

def _cagr_from_series_any(s: pd.Series) -> float:
    try:
        s = pd.Series(s).dropna().astype(float)
        if s.size < 2:
            return np.nan
        idx = pd.to_datetime(s.index)
        yrs = _years_span(idx)
        if not np.isfinite(yrs) or yrs <= 0:
            return np.nan
        a, b = float(s.iloc[0]), float(s.iloc[-1])
        if a <= 0 or b <= 0:
            return np.nan
        return (b / a) ** (1.0 / yrs) - 1.0
    except Exception:
        return np.nan

def _cagr_from_any(annual: pd.DataFrame, monthly: pd.DataFrame, rebased_m: pd.DataFrame, logical_name: str) -> float:
    # Annual first (or build from monthly)
    s_ann = _pick_annual_series(annual, monthly, logical_name)
    c = _cagr_from_series_any(s_ann)
    if pd.notna(c):
        return c
    # Monthly direct
    s_m = _pick_monthly_series(monthly, logical_name)
    c = _cagr_from_series_any(s_m)
    if pd.notna(c):
        return c
    # Rebased fallback
    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        map_rm = {"GLI_INDEX": "GLI", "NASDAQ": "NASDAQ", "GOLD": "GOLD"}
        rm_col = map_rm.get(logical_name, logical_name)
        if rm_col in rebased_m.columns:
            c = _cagr_from_series_any(rebased_m[rm_col])
            if pd.notna(c):
                return c
    return np.nan

def _build_rebased(monthly: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(monthly, pd.DataFrame) or monthly.empty:
        return pd.DataFrame()
    def _rb(s):
        s = pd.Series(s).dropna()
        return (s / s.iloc[0] * 100.0) if len(s) else s
    out = pd.DataFrame()
    gli = _pick_monthly_series(monthly, "GLI_INDEX")
    if not gli.empty: out["GLI"] = _rb(gli)
    for nm in ["NASDAQ", "SP500", "GOLD", "BTC", "ETH"]:
        ser = _pick_monthly_series(monthly, nm)
        if not ser.empty:
            out[nm] = _rb(ser)
    return out

# =========================
# Sidebar (no manual cache clear)
# =========================
st.sidebar.caption("GLI: Fed+ECB+BoJ−TGA−ONRRP (+PBoC optional)")
start     = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
years_n   = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10, step=1)
rf_annual = st.sidebar.number_input("Risk-free (annual)", 0.00, 0.10, 0.02, step=0.0025, format="%.4f")
win_m     = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=1)

fred_key = (st.secrets.get("FRED_API_KEY", "") or os.environ.get("FRED_API_KEY","")).strip()

# =========================
# Load Data (cached in gli_lib)
# =========================
with st.spinner("Loading GLI & assets..."):
    data = gl.load_all(
        fred_api_key=fred_key,
        start=start,
        end=None,
        years_for_cagr=int(years_n),
        risk_free_annual=float(rf_annual),
        include_pboc=False,
        pboc_series_id=None
    )

wk              = data.get("wk")
monthly         = data.get("monthly")
monthly_rets    = data.get("monthly_rets")
annual          = data.get("annual")
metrics_table   = data.get("metrics_table")
corr_matrix     = data.get("corr_matrix")
betas_df        = data.get("betas_df")
rebased_m       = data.get("rebased_m")
annual_yoy_fig  = data.get("annual_yoy_fig")

# normalize column names
if isinstance(monthly, pd.DataFrame) and "GLI" in monthly.columns and "GLI_INDEX" not in monthly.columns:
    monthly = monthly.rename(columns={"GLI": "GLI_INDEX"})
if isinstance(monthly_rets, pd.DataFrame) and "GLI" in monthly_rets.columns and "GLI_INDEX" not in monthly_rets.columns:
    monthly_rets = monthly_rets.rename(columns={"GLI": "GLI_INDEX"})

if not isinstance(rebased_m, pd.DataFrame) or rebased_m.empty:
    rebased_m = _build_rebased(monthly)

# =========================
# Title & Tabs
# =========================
st.title("GLI Dashboard")
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Rebased + Annual YoY", "📉 Rolling", "🧭 Regime & Events", "📋 Tables"]
)

# =========================
# KPI Row
# =========================
colA, colB, colC, colD, colE = st.columns(5)

gli_full = _cagr_from_any(annual, monthly, rebased_m, "GLI_INDEX")
# last N years CAGR (from annual if possible, else from monthly->annual)
gli_ann_for_n = _pick_annual_series(annual, monthly, "GLI_INDEX")
gli_n = gl.cagr_last_n_years(gli_ann_for_n, int(years_n)) if isinstance(gli_ann_for_n, pd.Series) else np.nan
if pd.isna(gli_n):
    gli_n = gl.cagr_last_n_years(_pick_monthly_series(monthly, "GLI_INDEX").resample("A-DEC").last(), int(years_n))

nas_full  = _cagr_from_any(annual, monthly, rebased_m, "NASDAQ")
gold_full = _cagr_from_any(annual, monthly, rebased_m, "GOLD")

colA.metric("GLI (CAGR, full)", _fmt_pct(gli_full))
colB.metric(f"GLI (CAGR, {int(years_n)}y)", _fmt_pct(gli_n))

nas_liq  = (nas_full - gli_full)  if pd.notna(nas_full)  and pd.notna(gli_full)  else np.nan
gold_liq = (gold_full - gli_full) if pd.notna(gold_full) and pd.notna(gli_full) else np.nan
colC.metric("NASDAQ − GLI (CAGR)", _fmt_pct(nas_liq))
colD.metric("GOLD − GLI (CAGR)",   _fmt_pct(gold_liq))

shp_gli = gl.sharpe(_pick_monthly_series(monthly_rets, "GLI_INDEX"), float(rf_annual), 12) \
          if (isinstance(monthly_rets, pd.DataFrame) and "GLI_INDEX" in monthly_rets.columns) else np.nan
colE.metric("Sharpe (GLI)", f"{shp_gli:.2f}" if pd.notna(shp_gli) else "—")

# =========================
# Tab 1: Rebased + Annual YoY
# =========================
with tab_main:
    st.subheader("(Monthly) Rebased = 100")
    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        options = list(rebased_m.columns)
        selected = set(st.multiselect(
            "เลือกเส้นที่ต้องการแสดง",
            options=options,
            default=options,
            key="rebased_sel",
            help="ซ่อน/แสดงซีรีส์ที่ต้องการเปรียบเทียบ"
        ))
        fig_rebase = go.Figure()
        for col in options:
            fig_rebase.add_trace(
                go.Scatter(
                    x=rebased_m.index, y=rebased_m[col], mode="lines", name=col,
                    visible=True if col in selected else "legendonly"
                )
            )
        fig_rebase.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            xaxis=dict(rangeslider=dict(visible=True)),
            margin=dict(t=30, l=40, r=20, b=40),
            height=520
        )
        st.plotly_chart(fig_rebase, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("No rebased data")

    st.markdown("#### Annual YoY: GLI (line) vs Assets (bars)")
    if annual_yoy_fig is None:
        if isinstance(monthly, pd.DataFrame) and not monthly.empty:
            ann = monthly.resample("A-DEC").last().pct_change().dropna() * 100.0
            ann = ann.rename(columns={"GLI_INDEX": "GLI_%YoY"})
            fig = go.Figure()
            if "GLI_%YoY" in ann.columns:
                fig.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"],
                                         mode="lines+markers", name="GLI_%YoY"))
            for c in [c for c in ann.columns if c != "GLI_%YoY"]:
                fig.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{c}_%YoY"))
            fig.update_layout(
                barmode="group", hovermode="x unified",
                legend=dict(orientation="h", y=1.05),
                xaxis=dict(rangeslider=dict(visible=True)),
                height=440
            )
            annual_yoy_fig = fig
    if annual_yoy_fig is not None:
        st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})

# =========================
# Rolling (precompute)
# =========================
roll = gl.rolling_corr_beta_alpha(monthly_rets, window=int(win_m))
roll_corr_m_df, roll_beta_m_df, roll_alpha_m_df = roll["corr"], roll["beta"], roll["alpha"]

# =========================
# Tab 2: Rolling
# =========================
with tab_roll:
    st.subheader(f"Rolling {int(win_m)}-Month vs GLI")
    c1, c2 = st.columns(2)

    with c1:
        fig_rc = go.Figure()
        for col in [c for c in roll_corr_m_df.columns if c != "GLI_INDEX"]:
            fig_rc.add_trace(go.Scatter(x=roll_corr_m_df.index, y=roll_corr_m_df[col], mode="lines", name=col))
        fig_rc.update_layout(title=f"Correlation (Rolling {int(win_m)}M)",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)),
                             yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})

    with c2:
        fig_rb = go.Figure()
        for col in [c for c in roll_beta_m_df.columns if c != "GLI_INDEX"]:
            fig_rb.add_trace(go.Scatter(x=roll_beta_m_df.index, y=roll_beta_m_df[col], mode="lines", name=col))
        fig_rb.update_layout(title=f"Beta (Rolling {int(win_m)}M)",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    fig_ra = go.Figure()
    for col in [c for c in roll_alpha_m_df.columns if c != "GLI_INDEX"]:
        fig_ra.add_trace(go.Scatter(x=roll_alpha_m_df.index, y=roll_alpha_m_df[col], mode="lines", name=col))
    fig_ra.update_layout(title=f"Alpha %/mo (Rolling {int(win_m)}M)",
                         hovermode="x unified",
                         legend=dict(orientation="h", y=1.02),
                         xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_ra, use_container_width=True, config={"displaylogo": False})

# =========================
# Regime (precompute)
# =========================
reg = gl.regime_and_events(monthly, monthly_rets)
regime_df      = reg["regime_df"]
exp_periods    = reg["expansion_periods"]
evt_up         = reg["evt_up"]
evt_down       = reg["evt_down"]

# =========================
# Tab 3: Regime & Events
# =========================
with tab_regime:
    st.subheader("GLI Regime (YoY>0) & Event Study")

    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        fig_reg = go.Figure()
        for col in rebased_m.columns:
            fig_reg.add_trace(go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines", name=col))
        for s, e in exp_periods:
            fig_reg.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
        fig_reg.update_layout(hovermode="x unified",
                              legend=dict(orientation="h", y=1.02),
                              xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_reg, use_container_width=True, config={"displaylogo": False})

    fig_gold_yoy = gl.gli_yoy_vs_gold(monthly, monthly_rets, regime_df, exp_periods)
    st.plotly_chart(fig_gold_yoy, use_container_width=True, config={"displaylogo": False})

    st.markdown("##### Event Study — ผลตอบแทนสะสมเฉลี่ยหลังจุดเปลี่ยนระบอบ (3/6/12 เดือนถัดไป)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**หลัง Upturn**")
        st.dataframe(_to_display_df(evt_up.round(2)), use_container_width=True)
    with c2:
        st.markdown("**หลัง Downturn**")
        st.dataframe(_to_display_df(evt_down.round(2)), use_container_width=True)

    st.markdown("#### 📌 Auto Summary")
    st.info(gl.auto_summary(metrics_table, betas_df, evt_up, evt_down, gl.perf_regime_table(monthly_rets, regime_df)))

# =========================
# Tab 4: Tables
# =========================
with tab_tables:
    st.subheader("Tables (compact)")
    with st.expander("📊 Liquidity-Adjusted & Risk Metrics", expanded=True):
        st.dataframe(_to_display_df(metrics_table), use_container_width=True, height=340)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔗 Correlation Matrix (monthly %)", expanded=False):
            st.dataframe(_to_display_df(corr_matrix.round(2)), use_container_width=True, height=350)
    with col2:
        with st.expander("β vs GLI (Monthly OLS)", expanded=False):
            st.dataframe(_to_display_df(betas_df.round(3)), use_container_width=True, height=350)
    with st.expander("📈 Monthly closes (preview)", expanded=False):
        st.dataframe(_to_display_df(monthly.tail(12)), use_container_width=True, height=320)
