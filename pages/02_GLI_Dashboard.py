# pages/02_GLI_Dashboard.py
import os, math, re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import gli_lib as gl  # ต้องมีไฟล์ gli_lib.py ตามที่ตั้งไว้

st.set_page_config(page_title="GLI Dashboard", layout="wide")

# =========================
# Helpers
# =========================
def _fmt_pct(x):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "—"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """แก้เคสคอลัมน์เป็น tuple เช่น ('GOLD','GC=F') -> 'GOLD' """
    if df is None or df.empty:
        return df
    newcols = []
    for c in df.columns:
        if isinstance(c, tuple) and len(c):
            newcols.append(str(c[0]))
        else:
            s = str(c)
            m = re.match(r"\('([^']+)'", s)
            newcols.append(m.group(1) if m else s)
    out = df.copy()
    out.columns = newcols
    return out

# ชื่อที่อยากโชว์บนกราฟ/ตาราง
PRETTY = {
    "GLI_INDEX": "GLI",
    "GLI": "GLI",
    "NASDAQ": "NASDAQ",
    "SP500": "S&P 500",
    "GOLD": "Gold",
    "BTC": "Bitcoin",
    "ETH": "Ether",
}

def _pick_symbol(raw):
    if isinstance(raw, tuple) and raw:
        return str(raw[0])
    s = str(raw)
    m = re.match(r"\('([^']+)'", s)
    return m.group(1) if m else s

def pretty_name(raw):
    return PRETTY.get(_pick_symbol(raw), _pick_symbol(raw))

def rename_pretty_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.rename(columns={c: pretty_name(c) for c in df.columns})

def _sanitize_for_st(df: pd.DataFrame) -> pd.DataFrame:
    """ป้องกัน pyarrow error จาก dtype แปลก ๆ"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            # ถ้าภายในเป็นลิสต์/ทูเพิล ให้แปลงเป็นสตริง
            out[c] = out[c].map(lambda v: str(v))
    return out

# =========================
# Sidebar
# =========================
st.sidebar.caption("GLI: Fed + ECB + BoJ − TGA − ONRRP (+PBoC optional)")
start     = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
years_n   = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10, step=1)
rf_annual = st.sidebar.number_input("Risk-free (annual)", 0.00, 0.10, 0.02, step=0.0025, format="%.4f")
win_m     = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=1)
st.sidebar.button("🔄 Refresh cache", on_click=lambda: st.cache_data.clear())

fred_key = (st.secrets.get("FRED_API_KEY", "") or os.environ.get("FRED_API_KEY","")).strip()

# =========================
# Load data
# =========================
with st.spinner("Loading GLI & assets..."):
    data = gl.load_all(
        fred_api_key=fred_key,
        start=start,
        end=None,
        years_for_cagr=years_n,
        risk_free_annual=rf_annual,
        include_pboc=False,
        pboc_series_id=None
    )

# core dfs
wk              = data["wk"]
monthly         = data["monthly"]
monthly_rets    = data["monthly_rets"]
annual          = data["annual"]
metrics_table   = data["metrics_table"]
corr_matrix     = data["corr_matrix"]
betas_df        = data["betas_df"]
rebased_m       = data["rebased_m"]
annual_yoy_fig  = data["annual_yoy_fig"]

# --- normalize columns then prettify names everywhere
monthly       = rename_pretty_cols(_flatten_cols(monthly))
monthly_rets  = rename_pretty_cols(_flatten_cols(monthly_rets))
annual        = rename_pretty_cols(_flatten_cols(annual))
rebased_m     = rename_pretty_cols(_flatten_cols(rebased_m))
if isinstance(corr_matrix, pd.DataFrame): corr_matrix = rename_pretty_cols(_flatten_cols(corr_matrix))
if isinstance(betas_df,   pd.DataFrame): betas_df   = rename_pretty_cols(_flatten_cols(betas_df))

if isinstance(metrics_table, pd.DataFrame):
    mt = metrics_table.copy()
    if "Asset" in mt.columns:
        mt["Asset"] = mt["Asset"].apply(pretty_name)
    metrics_table = mt

# Rolling stats
roll = gl.rolling_corr_beta_alpha(monthly_rets, window=win_m)
roll_corr_m_df = rename_pretty_cols(_flatten_cols(roll["corr"]))
roll_beta_m_df = rename_pretty_cols(_flatten_cols(roll["beta"]))
roll_alpha_m_df= rename_pretty_cols(_flatten_cols(roll["alpha"]))

# Regime & events
reg            = gl.regime_and_events(monthly, monthly_rets)
regime_df      = reg["regime_df"]
exp_periods    = reg["expansion_periods"]
evt_up, evt_down = reg["evt_up"], reg["evt_down"]

# =========================
# Title
# =========================
st.title("GLI Dashboard")

# =========================
# KPI row
# =========================
colA, colB, colC, colD, colE = st.columns(5)

# GLI CAGR
try:
    gli_full = gl.cagr_from_series(annual["GLI"])
    gli_n    = gl.cagr_last_n_years(annual["GLI"], years_n)
except Exception:
    gli_full = gli_n = np.nan

def _liquidity_adj(asset_name: str):
    try:
        if isinstance(metrics_table, pd.DataFrame) and not metrics_table.empty:
            row = metrics_table.loc[metrics_table["Asset"] == asset_name]
            if not row.empty and "LiquidityAdj_CAGR_full_%" in row.columns:
                return float(row["LiquidityAdj_CAGR_full_%"].iloc[0]) / 100.0
    except Exception:
        pass
    return np.nan

colA.metric("GLI CAGR (full)", _fmt_pct(gli_full))
colB.metric(f"GLI CAGR ({years_n}y)", _fmt_pct(gli_n))
colC.metric("NASDAQ Liquidity-Adj CAGR (full)", _fmt_pct(_liquidity_adj("NASDAQ")))
colD.metric("Gold Liquidity-Adj CAGR (full)",    _fmt_pct(_liquidity_adj("Gold")))
colE.metric("Sharpe (GLI regime mix)",
            f"{gl.sharpe(monthly_rets['GLI'], rf_annual, 12):.2f}" if "GLI" in monthly_rets.columns else "—")

# =========================
# Tabs
# =========================
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Rebased + Annual YoY", "📉 Rolling", "🧭 Regime & Events", "📋 Tables"]
)

# ---------- Tab 1: Rebased + Annual YoY ----------
with tab_main:
    st.subheader("(Monthly) GLI vs Assets — Rebased = 100")

    labels = list(rebased_m.columns)
    sel = set(st.multiselect(
        "เลือกเส้นที่ต้องการแสดง", options=labels, default=labels,
        key="rebased_sel", help="ซ่อน/แสดงซีรีส์ที่ต้องการเปรียบเทียบ"
    ))

    fig_rebase = go.Figure()
    for label in labels:
        s = rebased_m[label].dropna()
        fig_rebase.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=label,
            visible=True if label in sel else "legendonly"
        ))
    fig_rebase.update_layout(
        title="(Monthly) Rebased = 100",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(rangeslider=dict(visible=True)),
        margin=dict(t=60, l=40, r=20, b=40),
        height=520
    )
    st.plotly_chart(fig_rebase, use_container_width=True, config={"displaylogo": False})

    st.markdown("#### Annual YoY: GLI (line) vs Assets (bars)")
    if annual_yoy_fig is None:
        ann = monthly.resample("A-DEC").last().pct_change().dropna() * 100.0
        ann = ann.rename(columns={"GLI": "GLI_%YoY"})
        fig = go.Figure()
        if "GLI_%YoY" in ann.columns:
            fig.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"],
                                     mode="lines+markers", name="GLI %YoY"))
        for c in [c for c in ann.columns if c != "GLI_%YoY"]:
            fig.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{c} %YoY"))
        fig.update_layout(title="Annual YoY: GLI (line) vs Assets (bars)",
                          barmode="group", hovermode="x unified",
                          legend=dict(orientation="h", y=1.05),
                          xaxis=dict(rangeslider=dict(visible=True)))
        annual_yoy_fig = fig
    st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 2: Rolling ----------
with tab_roll:
    st.subheader(f"Rolling {win_m}-Month Statistics vs GLI (Monthly Returns)")
    c1, c2 = st.columns(2)

    # Rolling Corr
    with c1:
        fig_rc = go.Figure()
        for col in [c for c in roll_corr_m_df.columns if c != "GLI"]:
            fig_rc.add_trace(go.Scatter(x=roll_corr_m_df.index, y=roll_corr_m_df[col], mode="lines", name=col))
        fig_rc.update_layout(title=f"Rolling {win_m}M Correlation vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)),
                             yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})

    # Rolling Beta
    with c2:
        fig_rb = go.Figure()
        for col in [c for c in roll_beta_m_df.columns if c != "GLI"]:
            fig_rb.add_trace(go.Scatter(x=roll_beta_m_df.index, y=roll_beta_m_df[col], mode="lines", name=col))
        fig_rb.update_layout(title=f"Rolling {win_m}M Beta vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    # Rolling Alpha
    fig_ra = go.Figure()
    for col in [c for c in roll_alpha_m_df.columns if c != "GLI"]:
        fig_ra.add_trace(go.Scatter(x=roll_alpha_m_df.index, y=roll_alpha_m_df[col], mode="lines", name=col))
    fig_ra.update_layout(title=f"Rolling {win_m}M Alpha vs GLI (approx, %/mo)",
                         hovermode="x unified",
                         legend=dict(orientation="h", y=1.02),
                         xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_ra, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 3: Regime & Events ----------
with tab_regime:
    st.subheader("GLI Regime (YoY>0 = Expansion) & Event Study")

    # Rebased + shaded expansion
    fig_reg = go.Figure()
    for col in rebased_m.columns:
        fig_reg.add_trace(go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines", name=col))
    for s, e in exp_periods:
        fig_reg.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
    fig_reg.update_layout(title="Rebased (Monthly) + Expansion Shading",
                          hovermode="x unified",
                          legend=dict(orientation="h", y=1.02),
                          xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_reg, use_container_width=True, config={"displaylogo": False})

    # GLI YoY vs GOLD %/mo (dual axis)
    fig_gold_yoy = gl.gli_yoy_vs_gold(monthly, monthly_rets, regime_df, exp_periods)
    st.plotly_chart(fig_gold_yoy, use_container_width=True, config={"displaylogo": False})

    st.markdown("##### Event Study — ผลตอบแทนสะสมโดยเฉลี่ยหลังจุดเปลี่ยนระบอบ")
    st.caption("**Upturn** = GLI จากหดตัว → ขยายตัว, **Downturn** = GLI จากขยายตัว → หดตัว; วัดผลสะสมถัดไป 3/6/12 เดือน")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**หลัง Upturn**")
        st.dataframe(evt_up.round(2), use_container_width=True)
    with c2:
        st.markdown("**หลัง Downturn**")
        st.dataframe(evt_down.round(2), use_container_width=True)

    # Auto summary (Thai)
    st.markdown("#### 📌 Auto Summary")
    st.info(gl.auto_summary(metrics_table, betas_df, evt_up, evt_down,
                            gl.perf_regime_table(monthly_rets, regime_df)))

# ---------- Tab 4: Tables ----------
with tab_tables:
    st.subheader("Tables (compact)")
    with st.expander("📊 Liquidity-Adjusted & Risk Metrics", expanded=True):
        show = _sanitize_for_st(metrics_table if isinstance(metrics_table, pd.DataFrame) else pd.DataFrame())
        st.dataframe(show, use_container_width=True, height=340)

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔗 Correlation Matrix (monthly %)", expanded=False):
            df_corr = corr_matrix if isinstance(corr_matrix, pd.DataFrame) else pd.DataFrame()
            if not df_corr.empty:
                figc = px.imshow(df_corr, text_auto=".2f",
                                 color_continuous_scale="RdBu", zmin=-1, zmax=1)
                figc.update_layout(margin=dict(t=30, l=10, r=10, b=10))
                st.plotly_chart(figc, use_container_width=True, config={"displaylogo": False})
            st.dataframe(_sanitize_for_st(df_corr.round(2)), use_container_width=True, height=350)

    with col2:
        with st.expander("β vs GLI (Monthly OLS)", expanded=False):
            st.dataframe(_sanitize_for_st(
                betas_df.round(3) if isinstance(betas_df, pd.DataFrame) else pd.DataFrame()
            ), use_container_width=True, height=350)

    with st.expander("📈 Monthly closes (preview)", expanded=False):
        st.dataframe(_sanitize_for_st(
            monthly.tail(12) if isinstance(monthly, pd.DataFrame) else pd.DataFrame()
        ), use_container_width=True, height=320)
