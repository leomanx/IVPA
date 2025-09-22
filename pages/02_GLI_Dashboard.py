# pages/02_GLI_Dashboard.py
import os, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import gli_lib as gl

st.set_page_config(page_title="GLI Dashboard", layout="wide")

# ---------- Helpers ----------
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        return df.to_frame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(df)
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            cols.append(" / ".join([str(x) for x in c if str(x) != ""]))
        else:
            cols.append(str(c))
    out = df.copy()
    out.columns = cols
    return out

PRETTY = {
    "GLI_INDEX":"GLI",
    "NASDAQ":"Nasdaq",
    "SP500":"S&P 500",
    "GOLD":"Gold",
    "BTC":"BTC",
    "ETH":"ETH",
}
def rename_pretty_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [PRETTY.get(c, c) for c in out.columns]
    return out

def safe_df_for_st(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ถ้ามีคอลัมน์ object เป็น tuple/list -> แปลงเป็น string
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].apply(lambda v: ", ".join(map(str, v)) if isinstance(v, (tuple, list)) else v)
    return out

def fmtpct(x):
    if x is None: return "—"
    try:
        if isinstance(x,(int,float)) and (np.isnan(x) or np.isinf(x)):
            return "—"
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

# ---------- Sidebar ----------
st.sidebar.caption("GLI = Fed + ECB + BoJ − TGA − ONRRP (+PBoC optional)")
start     = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
years_n   = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10, step=1)
rf_annual = st.sidebar.number_input("Risk-free (annual)", 0.00, 0.10, 0.02, step=0.0025, format="%.4f")
win_m     = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=1)
st.sidebar.button("🔄 Refresh cache", on_click=lambda: st.cache_data.clear())

fred_key = (st.secrets.get("FRED_API_KEY","") or os.environ.get("FRED_API_KEY","")).strip()

# ---------- Load ----------
with st.spinner("Loading GLI & assets..."):
    data = gl.load_all(
        fred_api_key=fred_key, start=start, end=None,
        years_for_cagr=years_n, risk_free_annual=rf_annual,
        include_pboc=False, pboc_series_id=None
    )

# raw (เก็บชื่อคอลัมน์เดิม GLI_INDEX)
raw_monthly       = data["monthly"]
raw_monthly_rets  = data["monthly_rets"]
annual            = data["annual"]
metrics_table_raw = data["metrics_table"]
corr_matrix_raw   = data["corr_matrix"]
betas_df_raw      = data["betas_df"]
annual_yoy_fig    = data.get("annual_yoy_fig", None)

# UI copies (รีเนมสั้น/แบนคอลัมน์)
monthly      = rename_pretty_cols(_flatten_cols(raw_monthly))
monthly_rets = rename_pretty_cols(_flatten_cols(raw_monthly_rets))
corr_matrix  = rename_pretty_cols(_flatten_cols(corr_matrix_raw))
betas_df     = safe_df_for_st(rename_pretty_cols(_flatten_cols(betas_df_raw)))
metrics_table= safe_df_for_st(rename_pretty_cols(_flatten_cols(metrics_table_raw)))

# Rebased (monthly)
def rebase_to_100(s: pd.Series) -> pd.Series:
    s = pd.Series(s).dropna()
    return 100.0 * s / s.iloc[0] if len(s) else s

rebased_m = pd.DataFrame()
for c in monthly.columns:
    rebased_m[c] = rebase_to_100(monthly[c])

# Rolling/Regime ใช้ raw + GLI_INDEX
mr_for_roll = raw_monthly_rets.copy()
if "GLI" in mr_for_roll.columns and "GLI_INDEX" not in mr_for_roll.columns:
    mr_for_roll = mr_for_roll.rename(columns={"GLI":"GLI_INDEX"})

m_for_reg = raw_monthly.copy()
if "GLI" in m_for_reg.columns and "GLI_INDEX" not in m_for_reg.columns:
    m_for_reg = m_for_reg.rename(columns={"GLI":"GLI_INDEX"})

roll = gl.rolling_corr_beta_alpha(mr_for_roll, window=win_m)
roll_corr_m_df = rename_pretty_cols(_flatten_cols(roll["corr"]))
roll_beta_m_df = rename_pretty_cols(_flatten_cols(roll["beta"]))
roll_alpha_m_df= rename_pretty_cols(_flatten_cols(roll["alpha"]))

reg = gl.regime_and_events(m_for_reg, mr_for_roll)
regime_df      = reg["regime_df"]
exp_periods    = reg["expansion_periods"]
evt_up, evt_down = reg["evt_up"], reg["evt_down"]

# ---------- Title ----------
st.title("GLI Dashboard")

# ---------- KPI row ----------
colA, colB, colC, colD, colE = st.columns(5)
gli_full = gl.cagr_from_series(annual["GLI_INDEX"])
gli_n    = gl.cagr_last_n_years(annual["GLI_INDEX"], years_n)

# หา Liquidity-Adj จาก metrics_table_raw (ปลอดภัยสุด)
liqa_nas = liqa_gold = np.nan
try:
    mt = metrics_table_raw.copy()
    mt.columns = [str(c) for c in mt.columns]
    row_nas = mt.loc[mt["Asset"].astype(str).str.contains("NASDAQ", case=False)]
    row_gld = mt.loc[mt["Asset"].astype(str).str.contains("GOLD",   case=False)]
    # คอลัมน์ที่เป็น LiquidityAdj_* (เลือก full period)
    liqcol = [c for c in mt.columns if "LiquidityAdj_CAGR_full" in c]
    if liqcol:
        if not row_nas.empty:  liqa_nas  = float(row_nas.iloc[0][liqcol[0]])/100.0
        if not row_gld.empty:  liqa_gold = float(row_gld.iloc[0][liqcol[0]])/100.0
except Exception:
    pass

colA.metric("GLI CAGR (full)", fmtpct(gli_full))
colB.metric(f"GLI CAGR ({int(years_n)}y)", fmtpct(gli_n))
colC.metric("Nasdaq Liquidity-Adj CAGR (full)", fmtpct(liqa_nas))
colD.metric("Gold Liquidity-Adj CAGR (full)",   fmtpct(liqa_gold))
colE.metric(
    "Sharpe (GLI regime mix)",
    f"{gl.sharpe(mr_for_roll['GLI_INDEX'], rf_annual, 12):.2f}"
    if 'GLI_INDEX' in mr_for_roll.columns else "—"
)

# ---------- Tabs ----------
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Rebased + Annual YoY", "📉 Rolling", "🧭 Regime & Events", "📋 Tables"]
)

# ===== Tab 1: Rebased + Annual YoY =====
with tab_main:
    st.subheader("(Monthly) Rebased = 100 — GLI vs Assets")

    selected = st.multiselect(
        "เลือกซีรีส์ที่ต้องการแสดง", options=list(rebased_m.columns),
        default=list(rebased_m.columns), key="rb_sel"
    )
    selected = set(selected)

    fig_rebase = go.Figure()
    for col in rebased_m.columns:
        fig_rebase.add_trace(go.Scatter(
            x=rebased_m.index, y=rebased_m[col],
            mode="lines", name=col,
            visible=True if col in selected else "legendonly"
        ))
    fig_rebase.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        xaxis=dict(rangeslider=dict(visible=True)),
        margin=dict(t=36, l=40, r=20, b=40),
        height=520
    )
    st.plotly_chart(fig_rebase, use_container_width=True, config={"displaylogo": False})

    st.markdown("#### Annual YoY: GLI (line) vs Assets (bars)")
    if annual_yoy_fig is None:
        ann = raw_monthly.resample("A-DEC").last().pct_change().dropna() * 100.0
        ann = ann.rename(columns={"GLI_INDEX": "GLI_%YoY"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"],
                                 mode="lines+markers", name="GLI_%YoY"))
        for c in [c for c in ann.columns if c != "GLI_%YoY"]:
            nice = PRETTY.get(c, c)
            fig.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{nice} %YoY"))
        fig.update_layout(
            barmode="group", hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
            xaxis=dict(rangeslider=dict(visible=True)), height=420
        )
        annual_yoy_fig = fig
    st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})

# ===== Tab 2: Rolling =====
with tab_roll:
    st.subheader(f"Rolling {win_m}-Month vs GLI (Monthly Returns)")
    c1, c2 = st.columns(2)

    with c1:
        fig_rc = go.Figure()
        for col in [c for c in roll_corr_m_df.columns if c != "GLI"]:
            fig_rc.add_trace(go.Scatter(x=roll_corr_m_df.index, y=roll_corr_m_df[col],
                                        mode="lines", name=col))
        fig_rc.update_layout(title=f"Correlation (window={win_m}M)",
                             hovermode="x unified", legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)),
                             yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})

    with c2:
        fig_rb = go.Figure()
        for col in [c for c in roll_beta_m_df.columns if c != "GLI"]:
            fig_rb.add_trace(go.Scatter(x=roll_beta_m_df.index, y=roll_beta_m_df[col],
                                        mode="lines", name=col))
        fig_rb.update_layout(title=f"Beta (window={win_m}M)",
                             hovermode="x unified", legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    fig_ra = go.Figure()
    for col in [c for c in roll_alpha_m_df.columns if c != "GLI"]:
        fig_ra.add_trace(go.Scatter(x=roll_alpha_m_df.index, y=roll_alpha_m_df[col],
                                    mode="lines", name=col))
    fig_ra.update_layout(title=f"Alpha (approx, %/mo, window={win_m}M)",
                         hovermode="x unified", legend=dict(orientation="h", y=1.02),
                         xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_ra, use_container_width=True, config={"displaylogo": False})

# ===== Tab 3: Regime & Events =====
with tab_regime:
    st.subheader("GLI Regime (YoY>0 = Expansion) + Events")
    # Rebased + shading (ใช้ UI labels)
    fig_reg = go.Figure()
    for col in rebased_m.columns:
        fig_reg.add_trace(go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines", name=col))
    for s, e in exp_periods:
        fig_reg.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
    fig_reg.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.02),
                          xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_reg, use_container_width=True, config={"displaylogo": False})

    fig_gold_yoy = gl.gli_yoy_vs_gold(raw_monthly, raw_monthly_rets, regime_df, exp_periods)
    st.plotly_chart(fig_gold_yoy, use_container_width=True, config={"displaylogo": False})

    st.markdown("##### Event Study — ผลตอบแทนสะสมเฉลี่ยหลังจุดเปลี่ยนระบอบ")
    st.caption("**Upturn** = GLI จากหดตัว→ขยายตัว, **Downturn** = GLI จากขยายตัว→หดตัว (คำนวณผลสะสมถัดไป 3/6/12 เดือน)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**หลัง Upturn**")
        st.dataframe(evt_up.round(2), use_container_width=True)
    with c2:
        st.markdown("**หลัง Downturn**")
        st.dataframe(evt_down.round(2), use_container_width=True)

    st.markdown("#### 📌 Auto Summary")
    st.info(gl.auto_summary(metrics_table_raw, betas_df_raw, evt_up, evt_down, gl.perf_regime_table(raw_monthly_rets, regime_df)))

# ===== Tab 4: Tables =====
with tab_tables:
    st.subheader("Tables (compact)")

    with st.expander("📊 Liquidity-Adjusted & Risk Metrics", expanded=True):
        st.dataframe(metrics_table if isinstance(metrics_table, pd.DataFrame) else pd.DataFrame(),
                     use_container_width=True, height=340)

    c1, c2 = st.columns(2, vertical_alignment="top")
    with c1:
        with st.expander("🔗 Correlation Matrix (monthly %)", expanded=False):
            try:
                heat = px.imshow(corr_matrix.astype(float), text_auto=".2f",
                                 color_continuous_scale="RdBu", zmin=-1, zmax=1)
                heat.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=420)
                st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False})
            except Exception:
                st.dataframe(corr_matrix.round(2), use_container_width=True, height=350)
    with c2:
        with st.expander("β vs GLI (Monthly OLS)", expanded=False):
            st.dataframe(betas_df.round(3), use_container_width=True, height=350)

    with st.expander("📈 Monthly closes (preview)", expanded=False):
        st.dataframe(monthly.tail(12), use_container_width=True, height=320)
