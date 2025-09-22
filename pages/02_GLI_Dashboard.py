import os, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import gli_lib as gl

st.set_page_config(page_title="GLI Dashboard", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.caption("GLI: Fed+ECB+BoJ−TGA−ONRRP (+PBoC optional)")
start     = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
years_n   = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10, step=1)
rf_annual = st.sidebar.number_input("Risk-free (annual)", 0.00, 0.10, 0.02, step=0.0025, format="%.4f")
win_m     = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=1)
st.sidebar.button("🔄 Refresh cache", on_click=lambda: st.cache_data.clear())

# FRED key: อ่านจาก secrets > env > input
fred_key = (st.secrets.get("FRED_API_KEY", "") or os.environ.get("FRED_API_KEY","")).strip()

# ---------------- Load data ----------------
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

wk              = data["wk"]                 # weekly GLI proxy
monthly         = data["monthly"]            # GLI_INDEX + assets (M close)
monthly_rets    = data["monthly_rets"]       # %/mo
annual          = data["annual"]             # A-DEC close
metrics_table   = data["metrics_table"]      # summary table
corr_matrix     = data["corr_matrix"]
betas_df        = data["betas_df"]
rebased_m       = data["rebased_m"]          # for plotting
annual_yoy_fig  = data["annual_yoy_fig"]     # GLI (line) vs assets (bars)

roll = gl.rolling_corr_beta_alpha(monthly_rets, window=win_m)
roll_corr_m_df, roll_beta_m_df, roll_alpha_m_df = roll["corr"], roll["beta"], roll["alpha"]

reg = gl.regime_and_events(monthly, monthly_rets)
regime_df      = reg["regime_df"]
exp_periods    = reg["expansion_periods"]
evt_up, evt_down = reg["evt_up"], reg["evt_down"]

# ---------------- Title ----------------
st.title("GLI Dashboard")

# ---------------- KPI row (compact) ----------------
colA, colB, colC, colD, colE = st.columns(5)
gli_full   = gl.cagr_from_series(annual["GLI_INDEX"])
gli_n      = gl.cagr_last_n_years(annual["GLI_INDEX"], years_n)
gold_full  = gl.cagr_from_series(annual.get("GOLD", pd.Series(dtype=float)))
nas_full   = gl.cagr_from_series(annual.get("NASDAQ", pd.Series(dtype=float)))

def fmtpct(x): 
    return "—" if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))) else f"{x*100:.2f}%"

colA.metric("GLI CAGR (full)", fmtpct(gli_full))
colB.metric(f"GLI CAGR ({years_n}y)", fmtpct(gli_n))
colC.metric("NASDAQ Liquidity-Adj CAGR (full)", 
            fmtpct(gl.cagr_from_series(annual.get("NASDAQ", pd.Series(dtype=float))) - gli_full if pd.notna(gli_full) else np.nan))
colD.metric("Gold Liquidity-Adj CAGR (full)", 
            fmtpct(gl.cagr_from_series(annual.get("GOLD", pd.Series(dtype=float))) - gli_full if pd.notna(gli_full) else np.nan))
colE.metric("Sharpe (GLI regime mix)", 
            f"{gl.sharpe(monthly_rets['GLI_INDEX'], rf_annual, 12):.2f}" if "GLI_INDEX" in monthly_rets else "—")

# ---------------- Navigation (in-page tabs) ----------------
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Rebased + Annual YoY", "📉 Rolling", "🧭 Regime & Events", "📋 Tables"]
)

# ---------- Tab 1: Rebased + Annual YoY ----------
with tab_main:
    st.subheader("(Monthly) GLI vs NASDAQ / S&P500 / GOLD / BTC / ETH — Rebased = 100")

    # Toggle buttons (ไม่บังกราฟ)
    btns = set(st.multiselect(
    "เลือกเส้นที่ต้องการแสดง",
    options=list(rebased_m.columns),
    default=list(rebased_m.columns),
    key="rebased_sel",
    help="ซ่อน/แสดงซีรีส์ที่ต้องการเปรียบเทียบ"
))

    fig_rebase = go.Figure()
    for col in rebased_m.columns:
        fig_rebase.add_trace(
            go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines", name=col,
                       visible=True if col in btns else "legendonly")
        )
    fig_rebase.update_layout(
        title="(Monthly) Rebased=100",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(rangeslider=dict(visible=True)),
        margin=dict(t=60, l=40, r=20, b=40),
        height=520
    )
    st.plotly_chart(fig_rebase, use_container_width=True, config={"displaylogo": False})

    st.markdown("#### Annual YoY: GLI (line) vs Assets (bars)")
        # >>> วางบล็อก fallback ตรงนี้ <<<
    if annual_yoy_fig is None:
        ann = monthly.resample("A-DEC").last().pct_change().dropna() * 100.0
        ann = ann.rename(columns={"GLI_INDEX": "GLI_%YoY"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"],
                                 mode="lines+markers", name="GLI_%YoY"))
        for c in [c for c in ann.columns if c != "GLI_%YoY"]:
            fig.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{c}_%YoY"))
        fig.update_layout(title="Annual YoY: GLI (line) vs Assets (bars)",
                          barmode="group", hovermode="x unified",
                          legend=dict(orientation="h", y=1.05),
                          xaxis=dict(rangeslider=dict(visible=True)))
        annual_yoy_fig = fig

    # แล้วค่อยแสดงกราฟ (จะใช้ของจริงจาก gli_lib ถ้ามี หรือใช้ fallback ถ้า None)
    st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 2: Rolling ----------
with tab_roll:
    st.subheader(f"Rolling {win_m}-Month Statistics vs GLI (Monthly Returns)")
    c1, c2 = st.columns(2)

    # Rolling Corr
    with c1:
        fig_rc = go.Figure()
        for col in [c for c in roll_corr_m_df.columns if c != "GLI_INDEX"]:
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
        for col in [c for c in roll_beta_m_df.columns if c != "GLI_INDEX"]:
            fig_rb.add_trace(go.Scatter(x=roll_beta_m_df.index, y=roll_beta_m_df[col], mode="lines", name=col))
        fig_rb.update_layout(title=f"Rolling {win_m}M Beta vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    # Rolling Alpha
    fig_ra = go.Figure()
    for col in [c for c in roll_alpha_m_df.columns if c != "GLI_INDEX"]:
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
    st.info(gl.auto_summary(metrics_table, betas_df, evt_up, evt_down, gl.perf_regime_table(monthly_rets, regime_df)))

# ---------- Tab 4: Tables (in compact expanders) ----------
with tab_tables:
    st.subheader("Tables (compact)")
    with st.expander("📊 Liquidity-Adjusted & Risk Metrics", expanded=True):
        st.dataframe(metrics_table, use_container_width=True, height=340)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔗 Correlation Matrix (monthly %)", expanded=False):
            st.dataframe(corr_matrix.round(2), use_container_width=True, height=350)
    with col2:
        with st.expander("β vs GLI (Monthly OLS)", expanded=False):
            st.dataframe(betas_df.round(3), use_container_width=True, height=350)
    with st.expander("📈 Monthly closes (preview)", expanded=False):
        st.dataframe(monthly.tail(12), use_container_width=True, height=320)
