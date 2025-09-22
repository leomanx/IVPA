# pages/02_GLI_Dashboard.py
import os, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import gli_lib as gl

st.set_page_config(page_title="GLI Dashboard", layout="wide")

# ---------------- Helpers (safe/flatten) ----------------
def _flatten_cols(obj):
    """Return object with string/flat columns. Safe for None/Series/DataFrame."""
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        name = str(obj.name) if obj.name is not None else "value"
        return obj.to_frame(name)
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in tup]).strip() for tup in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]
        return df
    return obj  # other types untouched

def _sanitize_for_st(df: pd.DataFrame) -> pd.DataFrame:
    """ทำความสะอาด df เพื่อให้ st.dataframe แปลงเป็น Arrow ได้ไม่พัง"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # เปลี่ยน inf เป็น NaN ก่อน
    d = d.replace([np.inf, -np.inf], np.nan)

    # บังคับชื่อคอลัมน์และ index เป็นสตริง
    d.columns = [str(c) for c in d.columns]
    try:
        d.index = d.index.map(str)
    except Exception:
        pass

    # แปลงคอลัมน์ object → numeric ถ้าได้; ไม่ได้ให้เป็น string
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_object_dtype(s) or getattr(s, "dtype", None) == "Sparse[int]":
            try:
                s_num = pd.to_numeric(s, errors="coerce")
                # ถ้าแปลงเป็นตัวเลขได้ ≥ 50% ให้ใช้ numeric; ไม่งั้นใช้ string
                if s_num.notna().mean() >= 0.5:
                    d[c] = s_num
                else:
                    d[c] = s.astype(str)
            except Exception:
                d[c] = s.astype(str)

    return d

def _fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x*100:.2f}%"

# ---------------- Sidebar ----------------
st.sidebar.caption("GLI: Fed+ECB+BoJ − TGA − ONRRP (+PBoC optional)")
start     = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
years_n   = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10, step=1)
rf_annual = st.sidebar.number_input("Risk-free (annual)", 0.00, 0.10, 0.02, step=0.0025, format="%.4f")
win_m     = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=1)
st.sidebar.button("🔄 Refresh cache", on_click=st.cache_data.clear)

# FRED key: secrets > env
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

# unpack
wk              = data.get("wk")
monthly         = data.get("monthly")
monthly_rets    = data.get("monthly_rets")
annual          = data.get("annual")
metrics_table   = data.get("metrics_table")
corr_matrix     = data.get("corr_matrix")
betas_df        = data.get("betas_df")
rebased_m       = data.get("rebased_m")
annual_yoy_fig  = data.get("annual_yoy_fig")

# normalize/flatten
monthly       = _flatten_cols(monthly)
monthly_rets  = _flatten_cols(monthly_rets)
annual        = _flatten_cols(annual)
rebased_m     = _flatten_cols(rebased_m)
if isinstance(corr_matrix, pd.DataFrame):
    corr_matrix = _flatten_cols(corr_matrix)
if isinstance(betas_df, pd.DataFrame):
    try:
        betas_df.index = betas_df.index.map(str)
    except Exception:
        pass

# defensive guards
if monthly is None or monthly.empty:
    st.error("ไม่พบข้อมูลรายเดือน (monthly) — ตรวจ API/เน็ต แล้วลอง Refresh cache")
    st.stop()
if rebased_m is None or rebased_m.empty:
    # สร้างจาก monthly หากไม่มี
    try:
        cols = [c for c in monthly.columns if c]  # all
        rebased_m = pd.DataFrame({ c: (monthly[c]/monthly[c].dropna().iloc[0])*100.0 for c in cols if monthly[c].dropna().size })
    except Exception:
        st.error("สร้าง rebased panel ไม่สำเร็จ")
        st.stop()

# rolling stats
try:
    roll = gl.rolling_corr_beta_alpha(monthly_rets, window=win_m)
    roll_corr_m_df, roll_beta_m_df, roll_alpha_m_df = roll["corr"], roll["beta"], roll["alpha"]
    roll_corr_m_df  = _flatten_cols(roll_corr_m_df)
    roll_beta_m_df  = _flatten_cols(roll_beta_m_df)
    roll_alpha_m_df = _flatten_cols(roll_alpha_m_df)
except Exception as e:
    roll_corr_m_df = roll_beta_m_df = roll_alpha_m_df = pd.DataFrame()
    st.warning(f"คำนวณ rolling ไม่สำเร็จ: {e}")

# regime/events
try:
    reg = gl.regime_and_events(monthly, monthly_rets)
    regime_df      = reg["regime_df"]
    exp_periods    = reg["expansion_periods"]
    evt_up, evt_down = reg["evt_up"], reg["evt_down"]
except Exception as e:
    regime_df = pd.DataFrame()
    exp_periods, evt_up, evt_down = [], pd.DataFrame(), pd.DataFrame()
    st.warning(f"คำนวณ Regime/Events ไม่สำเร็จ: {e}")

# ---------------- Title ----------------
st.title("GLI Dashboard")

# ---------------- KPI row (compact) ----------------
colA, colB, colC, colD, colE = st.columns(5)

try:
    gli_full = gl.cagr_from_series(annual["GLI_INDEX"])
    gli_n    = gl.cagr_last_n_years(annual["GLI_INDEX"], years_n)
except Exception:
    gli_full = gli_n = np.nan

nas_full  = gl.cagr_from_series(annual.get("NASDAQ", pd.Series(dtype=float))) if isinstance(annual, pd.DataFrame) else np.nan
gold_full = gl.cagr_from_series(annual.get("GOLD",   pd.Series(dtype=float))) if isinstance(annual, pd.DataFrame) else np.nan

colA.metric("GLI CAGR (full)", _fmt_pct(gli_full))
colB.metric(f"GLI CAGR ({years_n}y)", _fmt_pct(gli_n))
colC.metric("NASDAQ Liquidity-Adj CAGR (full)",
            _fmt_pct((gl.cagr_from_series(annual.get("NASDAQ", pd.Series(dtype=float))) - gli_full) if pd.notna(gli_full) else np.nan))
colD.metric("Gold Liquidity-Adj CAGR (full)",
            _fmt_pct((gl.cagr_from_series(annual.get("GOLD", pd.Series(dtype=float))) - gli_full) if pd.notna(gli_full) else np.nan))
colE.metric("Sharpe (GLI regime mix)",
            f"{gl.sharpe(monthly_rets['GLI_INDEX'], rf_annual, 12):.2f}" if ("GLI_INDEX" in monthly_rets.columns) else "—")

# ---------------- Tabs ----------------
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Rebased + Annual YoY", "📉 Rolling", "🧭 Regime & Events", "📋 Tables"]
)

# ---------- Tab 1: Rebased + Annual YoY ----------
with tab_main:
    st.subheader("(Monthly) GLI vs NASDAQ / S&P500 / GOLD / BTC / ETH — Rebased = 100")

    # multiselect ใช้ label เป็น string 100%
    label_map = {str(c): c for c in rebased_m.columns}
    labels    = list(label_map.keys())
    sel = set(st.multiselect(
        "เลือกเส้นที่ต้องการแสดง",
        options=labels,
        default=labels,
        key="rebased_sel",
        help="ซ่อน/แสดงซีรีส์ที่ต้องการเปรียบเทียบ"
    ))

    fig_rebase = go.Figure()
    for label, col in label_map.items():
        s = rebased_m[col].dropna()
        fig_rebase.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            name=label, visible=True if label in sel else "legendonly"
        ))
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

    # fallback ถ้า gli_lib ไม่ได้ส่งรูปมา
    if annual_yoy_fig is None:
        try:
            ann = monthly.resample("A-DEC").last().pct_change().dropna() * 100.0
            ann = ann.rename(columns={"GLI_INDEX": "GLI_%YoY"})
            fig = go.Figure()
            if "GLI_%YoY" in ann.columns:
                fig.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"],
                                         mode="lines+markers", name="GLI_%YoY"))
            for c in [c for c in ann.columns if c != "GLI_%YoY"]:
                fig.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{c}_%YoY"))
            fig.update_layout(title="Annual YoY: GLI (line) vs Assets (bars)",
                              barmode="group", hovermode="x unified",
                              legend=dict(orientation="h", y=1.05),
                              xaxis=dict(rangeslider=dict(visible=True)))
            annual_yoy_fig = fig
        except Exception as e:
            st.warning(f"สร้างกราฟ Annual YoY ไม่สำเร็จ: {e}")

    if annual_yoy_fig is not None:
        st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 2: Rolling ----------
with tab_roll:
    st.subheader(f"Rolling {win_m}-Month Statistics vs GLI (Monthly Returns)")
    c1, c2 = st.columns(2)

    # Rolling Corr
    with c1:
        fig_rc = go.Figure()
        if not roll_corr_m_df.empty:
            for col in [c for c in roll_corr_m_df.columns if c != "GLI_INDEX"]:
                s = roll_corr_m_df[col].dropna()
                fig_rc.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col)))
        fig_rc.update_layout(title=f"Rolling {win_m}M Correlation vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)),
                             yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})

    # Rolling Beta
    with c2:
        fig_rb = go.Figure()
        if not roll_beta_m_df.empty:
            for col in [c for c in roll_beta_m_df.columns if c != "GLI_INDEX"]:
                s = roll_beta_m_df[col].dropna()
                fig_rb.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col)))
        fig_rb.update_layout(title=f"Rolling {win_m}M Beta vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    # Rolling Alpha
    fig_ra = go.Figure()
    if not roll_alpha_m_df.empty:
        for col in [c for c in roll_alpha_m_df.columns if c != "GLI_INDEX"]:
            s = roll_alpha_m_df[col].dropna()
            fig_ra.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col)))
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
        s = rebased_m[col].dropna()
        fig_reg.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col)))
    for s, e in (exp_periods or []):
        fig_reg.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
    fig_reg.update_layout(title="Rebased (Monthly) + Expansion Shading",
                          hovermode="x unified",
                          legend=dict(orientation="h", y=1.02),
                          xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_reg, use_container_width=True, config={"displaylogo": False})

    # GLI YoY vs GOLD %/mo (dual axis)
    try:
        fig_gold_yoy = gl.gli_yoy_vs_gold(monthly, monthly_rets, regime_df, exp_periods)
        st.plotly_chart(fig_gold_yoy, use_container_width=True, config={"displaylogo": False})
    except Exception as e:
        st.warning(f"สร้างกราฟ GLI YoY vs GOLD ไม่สำเร็จ: {e}")

    st.markdown("##### Event Study — ผลตอบแทนสะสมโดยเฉลี่ยหลังจุดเปลี่ยนระบอบ")
    st.caption("**Upturn** = GLI จากหดตัว → ขยายตัว, **Downturn** = GLI จากขยายตัว → หดตัว; วัดผลสะสมถัดไป 3/6/12 เดือน")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**หลัง Upturn**")
        st.dataframe(evt_up.round(2) if not evt_up.empty else pd.DataFrame(), use_container_width=True)
    with c2:
        st.markdown("**หลัง Downturn**")
        st.dataframe(evt_down.round(2) if not evt_down.empty else pd.DataFrame(), use_container_width=True)

    # Auto summary (Thai)
    try:
        perf_tbl = gl.perf_regime_table(monthly_rets, regime_df)
        st.markdown("#### 📌 Auto Summary")
        st.info(gl.auto_summary(metrics_table, betas_df, evt_up, evt_down, perf_tbl))
    except Exception as e:
        st.warning(f"สรุป Auto Summary ไม่สำเร็จ: {e}")

# ---------- Tab 4: Tables (compact) ----------
with tab_tables:
    st.subheader("Tables (compact)")
    with st.expander("📊 Liquidity-Adjusted & Risk Metrics", expanded=True):
    st.dataframe(_sanitize_for_st(metrics_table), use_container_width=True, height=340)

with col1:
    with st.expander("🔗 Correlation Matrix (monthly %)", expanded=False):
        st.dataframe(_sanitize_for_st(corr_matrix.round(2) if isinstance(corr_matrix, pd.DataFrame) else pd.DataFrame()),
                     use_container_width=True, height=350)

with col2:
    with st.expander("β vs GLI (Monthly OLS)", expanded=False):
        st.dataframe(_sanitize_for_st(betas_df.round(3) if isinstance(betas_df, pd.DataFrame) else pd.DataFrame()),
                     use_container_width=True, height=350)

with st.expander("📈 Monthly closes (preview)", expanded=False):
    st.dataframe(_sanitize_for_st(monthly.tail(12) if isinstance(monthly, pd.DataFrame) else pd.DataFrame()),
                 use_container_width=True, height=320)
