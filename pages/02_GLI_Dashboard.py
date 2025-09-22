# pages/02_GLI_Dashboard.py
import os, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import gli_lib as gl

# ---------------- Page & Style ----------------
st.set_page_config(page_title="GLI Dashboard", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1500px;}
  .stPlotlyChart {background: transparent;}
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers (safe display / names) ----------------
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" - ".join([str(x) for x in tup if str(x)!=""]) for tup in out.columns.values]
    else:
        out.columns = [str(c) for c in out.columns]
    return out

def _to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """ทำ DF ให้ปลอดภัยต่อ st.dataframe/Arrow (รวม MultiIndex, Period/Datetime index, object dtype)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # --- normalize index ---
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
        lvl_cols = [c for c in out.columns if str(c).startswith("level_")]
        if lvl_cols:
            out.insert(0, "Index", out[lvl_cols].astype(str).agg(" - ".join, axis=1))
            out.drop(columns=lvl_cols, inplace=True)
    elif not isinstance(out.index, pd.RangeIndex):
        # Datetime/Period/อื่น ๆ → string
        try:
            idx_str = out.index.astype(str)
        except Exception:
            idx_str = out.index.map(lambda x: str(getattr(x, "to_timestamp", lambda: x)()))
        out.insert(0, "Index", idx_str)
        out = out.reset_index(drop=True)

    # --- column names to str ---
    out.columns = [str(c) for c in out.columns]

    # --- object columns to primitive/string ---
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].apply(lambda v: v.item() if isinstance(v, np.generic) else v)
            out[c] = out[c].apply(lambda v: v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
    return out

def _fmt_pct(x):
    if x is None: return "—"
    try:
        if isinstance(x, (float, int)) and not np.isnan(x) and not np.isinf(x):
            return f"{x*100:.2f}%"
    except Exception:
        pass
    return "—"

def _safe_get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if isinstance(df, pd.DataFrame) and col in df.columns:
        return df[col]
    return pd.Series(dtype=float)

def _pick_col(df: pd.DataFrame, candidates) -> str | None:
    """เลือกคอลัมน์แบบยืดหยุ่น (รับทั้งชื่อเดี่ยว หรือชื่อที่ถูกแปลงจาก MultiIndex เช่น 'NASDAQ - 0')."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = [str(c) for c in df.columns]

    # ตรงตัวก่อน
    for t in candidates:
        if t in cols:
            return t

    # แบบ contains / ขอบ-กลาง (รองรับพวก "('NASDAQ', 0)" -> ถูก flatten เป็น "NASDAQ - 0")
    low_cols = [c.lower() for c in cols]
    for t in candidates:
        t_low = str(t).lower()
        for i, c_low in enumerate(low_cols):
            if (t_low in c_low) or c_low.startswith(t_low + " -") or c_low.endswith("- " + t_low):
                return cols[i]
    return None

# ---------------- Sidebar ----------------
st.sidebar.caption("GLI: Fed + ECB + BoJ − TGA − ONRRP (+PBoC optional)")
start     = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
years_n   = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10, step=1)
rf_annual = st.sidebar.number_input("Risk-free (annual)", 0.00, 0.10, 0.02, step=0.0025, format="%.4f")
win_m     = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=1)
refresh   = st.sidebar.button("🔄 Refresh cache")
if refresh:
    st.cache_data.clear()

fred_key = (st.secrets.get("FRED_API_KEY", "") or os.environ.get("FRED_API_KEY","")).strip()

# ---------------- Cached loaders (ค้าง cache ข้าม page) ----------------
@st.cache_data(show_spinner=True)
def _load_all_cached(fred_api_key: str, start: str, years_for_cagr: int, risk_free_annual: float):
    return gl.load_all(
        fred_api_key=fred_api_key,
        start=start,
        end=None,
        years_for_cagr=years_for_cagr,
        risk_free_annual=risk_free_annual,
        include_pboc=False,
        pboc_series_id=None
    )

@st.cache_data(show_spinner=False)
def _rolling_cached(monthly_rets: pd.DataFrame, window: int):
    return gl.rolling_corr_beta_alpha(monthly_rets, window=window)

@st.cache_data(show_spinner=False)
def _regime_cached(monthly: pd.DataFrame, monthly_rets: pd.DataFrame):
    return gl.regime_and_events(monthly, monthly_rets)

# ---------------- Load all once ----------------
with st.spinner("Loading GLI & assets..."):
    data = _load_all_cached(fred_key, start, int(years_n), float(rf_annual))

# unpack
wk              = data.get("wk")
monthly         = _flatten_cols(data.get("monthly"))
monthly_rets    = _flatten_cols(data.get("monthly_rets"))
annual          = _flatten_cols(data.get("annual"))
metrics_table   = _flatten_cols(data.get("metrics_table"))
corr_matrix     = _flatten_cols(data.get("corr_matrix"))
betas_df        = _flatten_cols(data.get("betas_df"))
rebased_m       = _flatten_cols(data.get("rebased_m"))
annual_yoy_fig  = data.get("annual_yoy_fig")

# fallback annual_yoy_fig (ในกรณี lib ยังไม่ได้คืนมา)
if annual_yoy_fig is None and isinstance(monthly, pd.DataFrame) and not monthly.empty:
    ann = monthly.resample("A-DEC").last().pct_change().dropna() * 100.0
    ann = ann.rename(columns={"GLI_INDEX": "GLI_%YoY"})
    fig = go.Figure()
    if "GLI_%YoY" in ann.columns:
        fig.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"], mode="lines+markers", name="GLI YoY"))
    for c in [c for c in ann.columns if c != "GLI_%YoY"]:
        fig.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{c} YoY"))
    fig.update_layout(
        title="Annual YoY: GLI (line) vs Assets (bars)",
        barmode="group", hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        xaxis=dict(rangeslider=dict(visible=True)),
        margin=dict(t=50, l=40, r=20, b=40), height=420
    )
    annual_yoy_fig = fig

# rolling & regime (cached)
roll = _rolling_cached(monthly_rets, int(win_m))
roll_corr_m_df = _flatten_cols(roll.get("corr"))
roll_beta_m_df = _flatten_cols(roll.get("beta"))
roll_alpha_m_df= _flatten_cols(roll.get("alpha"))

reg = _regime_cached(monthly, monthly_rets)
regime_df      = reg.get("regime_df")
exp_periods    = reg.get("expansion_periods") or []
evt_up         = reg.get("evt_up")
evt_down       = reg.get("evt_down")

# ---------------- Title ----------------
st.title("GLI Dashboard")

# ---------------- KPI row ----------------
colA, colB, colC, colD, colE = st.columns(5)

gli_full = gl.cagr_from_series(_safe_get_series(annual, "GLI_INDEX"))
gli_n    = gl.cagr_last_n_years(_safe_get_series(annual, "GLI_INDEX"), int(years_n))

# --- เลือกคอลัมน์ NASDAQ / GOLD แบบยืดหยุ่น ---
nas_col  = _pick_col(annual, ["NASDAQ", "NDX", "IXIC"])
gold_col = _pick_col(annual, ["GOLD", "GC=F", "XAUUSD"])

nas_full  = gl.cagr_from_series(_safe_get_series(annual, nas_col))  if nas_col  else np.nan
gold_full = gl.cagr_from_series(_safe_get_series(annual, gold_col)) if gold_col else np.nan

colA.metric("GLI CAGR (full)", _fmt_pct(gli_full))
colB.metric(f"GLI CAGR ({int(years_n)}y)", _fmt_pct(gli_n))

# Liquidity-adjusted (asset − GLI)
nas_liq = (nas_full - gli_full) if (pd.notna(nas_full) and pd.notna(gli_full)) else np.nan
gold_liq= (gold_full - gli_full) if (pd.notna(gold_full) and pd.notna(gli_full)) else np.nan
colC.metric("NASDAQ − GLI (CAGR)", _fmt_pct(nas_liq))
colD.metric("GOLD − GLI (CAGR)",   _fmt_pct(gold_liq))

shp_gli = gl.sharpe(_safe_get_series(monthly_rets, "GLI_INDEX"), float(rf_annual), 12) if "GLI_INDEX" in monthly_rets.columns else np.nan
colE.metric("Sharpe (GLI)", f"{shp_gli:.2f}" if pd.notna(shp_gli) else "—")

# ---------------- Tabs ----------------
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Overview", "📉 Rolling", "🧭 Regime", "📋 Tables"]
)

# ---------- Tab: Overview ----------
with tab_main:
    st.subheader("Rebased = 100 (Monthly)")
    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        # Selector
        show_cols = st.multiselect(
            "Show series",
            options=list(rebased_m.columns),
            default=list(rebased_m.columns),
            key="rebased_sel",
            help="ซ่อน/แสดงซีรีส์ที่ต้องการ"
        )
        fig_rebase = go.Figure()
        for col in rebased_m.columns:
            fig_rebase.add_trace(
                go.Scatter(
                    x=rebased_m.index, y=rebased_m[col], mode="lines", name=col,
                    visible=True if col in show_cols else "legendonly"
                )
            )
        fig_rebase.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            xaxis=dict(rangeslider=dict(visible=True)),
            margin=dict(t=30, l=40, r=20, b=40),
            height=480
        )
        st.plotly_chart(fig_rebase, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("No rebased data.")

    st.markdown("#### Annual YoY: GLI (line) vs Assets (bars)")
    if annual_yoy_fig is not None:
        st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("No annual YoY figure.")

# ---------- Tab: Rolling ----------
with tab_roll:
    st.subheader(f"Rolling {int(win_m)}-Month vs GLI (Monthly Returns)")
    c1, c2 = st.columns(2)
    with c1:
        fig_rc = go.Figure()
        if isinstance(roll_corr_m_df, pd.DataFrame) and not roll_corr_m_df.empty:
            for col in [c for c in roll_corr_m_df.columns if c != "GLI_INDEX"]:
                fig_rc.add_trace(go.Scatter(x=roll_corr_m_df.index, y=roll_corr_m_df[col], mode="lines", name=col))
        fig_rc.update_layout(title="Correlation", hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)),
                             yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})
    with c2:
        fig_rb = go.Figure()
        if isinstance(roll_beta_m_df, pd.DataFrame) and not roll_beta_m_df.empty:
            for col in [c for c in roll_beta_m_df.columns if c != "GLI_INDEX"]:
                fig_rb.add_trace(go.Scatter(x=roll_beta_m_df.index, y=roll_beta_m_df[col], mode="lines", name=col))
        fig_rb.update_layout(title="Beta", hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    fig_ra = go.Figure()
    if isinstance(roll_alpha_m_df, pd.DataFrame) and not roll_alpha_m_df.empty:
        for col in [c for c in roll_alpha_m_df.columns if c != "GLI_INDEX"]:
            fig_ra.add_trace(go.Scatter(x=roll_alpha_m_df.index, y=roll_alpha_m_df[col], mode="lines", name=col))
    fig_ra.update_layout(title="Alpha (approx, %/mo)", hovermode="x unified",
                         legend=dict(orientation="h", y=1.02),
                         xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_ra, use_container_width=True, config={"displaylogo": False})

# ---------- Tab: Regime ----------
with tab_regime:
    st.subheader("GLI Regime (YoY>0 = Expansion) & Event Study")
    # Rebased + shading
    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        fig_reg = go.Figure()
        for col in rebased_m.columns:
            fig_reg.add_trace(go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines", name=col))
        for s, e in exp_periods:
            fig_reg.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
        fig_reg.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.02),
                              xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_reg, use_container_width=True, config={"displaylogo": False})

    # GLI YoY vs GOLD %/mo
    try:
        fig_gold_yoy = gl.gli_yoy_vs_gold(monthly, monthly_rets, regime_df, exp_periods)
        st.plotly_chart(fig_gold_yoy, use_container_width=True, config={"displaylogo": False})
    except Exception:
        pass

    st.markdown("##### Event Study — Avg cumulative returns after regime switch")
    st.caption("Upturn = GLI หด→ขยาย, Downturn = GLI ขยาย→หด; วัดผลสะสมถัดไป 3/6/12 เดือน")

    evt_up_disp   = _to_display_df(evt_up)   if isinstance(evt_up, pd.DataFrame) else pd.DataFrame()
    evt_down_disp = _to_display_df(evt_down) if isinstance(evt_down, pd.DataFrame) else pd.DataFrame()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**หลัง Upturn**")
        st.dataframe(evt_up_disp, use_container_width=True, height=320)
    with c2:
        st.markdown("**หลัง Downturn**")
        st.dataframe(evt_down_disp, use_container_width=True, height=320)

    # Auto summary (Thai) — ทำให้ข้อความสะอาด ไม่ซ้ำชื่อ tuple
    try:
        perf_tbl = gl.perf_regime_table(monthly_rets, regime_df)
        summary_txt = gl.auto_summary(metrics_table, betas_df, evt_up, evt_down, perf_tbl)
        summary_txt = summary_txt.replace("('", "").replace("')", "").replace(" - ", " ").replace("_%/mo", "")
        st.markdown("#### 📌 Auto Summary")
        st.info(summary_txt)
    except Exception:
        pass

# ---------- Tab: Tables ----------
with tab_tables:
    st.subheader("Tables (compact)")

    with st.expander("📊 Liquidity-Adjusted & Risk Metrics", expanded=True):
        st.dataframe(_to_display_df(metrics_table), use_container_width=True, height=360)

    # สี Heatmap ของ Correlation (plotly) + ตารางดิบ
    st.markdown("#### Correlation (monthly %)")
    c1, c2 = st.columns([3,2], vertical_alignment="top")
    with c1:
        if isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.empty:
            cm = corr_matrix.copy()
            if cm.shape[0] > 30:
                cm = cm.iloc[:30, :30]
            fig_hm = px.imshow(cm, text_auto=False, zmin=-1, zmax=1,
                               color_continuous_scale="RdBu", origin="lower",
                               aspect="auto", title="Correlation heatmap")
            fig_hm.update_layout(margin=dict(t=50, l=40, r=20, b=40), height=500)
            st.plotly_chart(fig_hm, use_container_width=True, config={"displaylogo": False})
        else:
            st.info("No correlation matrix.")
    with c2:
        st.dataframe(_to_display_df(corr_matrix.round(2)), use_container_width=True, height=500)

    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        with st.expander("β vs GLI (Monthly OLS)", expanded=False):
            st.dataframe(_to_display_df(betas_df.round(3)), use_container_width=True, height=350)
    with col2:
        with st.expander("📈 Monthly closes (preview)", expanded=False):
            st.dataframe(_to_display_df(monthly.tail(12)), use_container_width=True, height=350)
