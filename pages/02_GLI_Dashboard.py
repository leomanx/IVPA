# pages/02_GLI_Dashboard.py
import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import gli_lib as gl

# =========================
# Page config
# =========================
st.set_page_config(page_title="GLI Dashboard", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.caption("GLI = Fed + ECB + BoJ − TGA − ONRRP (+ PBoC optional)")
start     = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
years_n   = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10, step=1)
rf_annual = st.sidebar.number_input("Risk-free (annual)", 0.00, 0.10, 0.02, step=0.0025, format="%.4f")
win_m     = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=1)

# ปุ่ม refresh cache
def _clear_cache():
    st.cache_data.clear()
st.sidebar.button("🔄 Refresh cache", on_click=_clear_cache)

# FRED key: secrets > env
fred_key = (st.secrets.get("FRED_API_KEY", "") or os.environ.get("FRED_API_KEY","")).strip()

# =========================
# Helpers
# =========================
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ทำให้ชื่อ column เป็น string เดียว (กัน Arrow/Plotly งอแง)"""
    if isinstance(df, pd.DataFrame):
        new = df.copy()
        new.columns = [(" / ".join([str(x) for x in c]) if isinstance(c, (list, tuple)) else str(c)) for c in new.columns]
        return new
    return df

# ---------- Helper: safe for st.dataframe ----------
def _to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """ทำให้ DataFrame ปลอดภัยต่อ st.dataframe/Arrow."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()

    # เอา index แปลก ๆ ออกมาก่อน แล้ว reset ให้เรียบ
    if not isinstance(out.index, pd.RangeIndex):
        out.insert(0, "Index", out.index.astype(str))
        out = out.reset_index(drop=True)

    # ชื่อคอลัมน์เป็น string ล้วน
    out.columns = [str(c) for c in out.columns]

    # คอลัมน์ object -> primitive/string
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].apply(lambda v: v.item() if isinstance(v, np.generic) else v)
            out[c] = out[c].apply(
                lambda v: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
            )
    return out

def _pretty_name(s: str) -> str:
    s = str(s)
    s = s.replace("('", "").replace("')", "").replace("', '", " - ").replace("(", "").replace(")", "")
    s = s.replace("SP500", "S&P 500").replace("NASDAQ", "NASDAQ").replace("GC=F", "Gold").replace("BTC-USD", "Bitcoin").replace("ETH-USD", "Ethereum")
    s = s.replace("BTC - 2", "Bitcoin").replace("ETH - ETH-USD", "Ethereum")
    return s

def _pick_liq_adj(metrics_table: pd.DataFrame, keyword: str):
    """ดึงคอลัมน์ LiquidityAdj_CAGR_full_% ของสินทรัพย์ที่ระบุด้วย keyword"""
    try:
        m = metrics_table.copy()
        m["Asset_str"] = m["Asset"].astype(str)
        row = m[m["Asset_str"].str.contains(keyword, case=False)]
        if row.empty:
            return None
        val = row["LiquidityAdj_CAGR_full_%"].iloc[0]
        return None if pd.isna(val) else float(val)/100.0  # เป็นอัตรา (ไม่ใช่ %)
    except Exception:
        return None

def _fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x*100:.2f}%"

def _make_summary(metrics_table, betas_df, evt_up, evt_down, perf_regime_table):
    lines = []
    # Past
    try:
        liq_cols = [c for c in metrics_table.columns if "LiquidityAdj_CAGR" in c]
        liq_mean = metrics_table.set_index("Asset")[liq_cols].mean(axis=1).sort_values(ascending=False)
        past_top = ", ".join([_pretty_name(a) for a in liq_mean.head(2).index.tolist()])
        past_bot = ", ".join([_pretty_name(a) for a in liq_mean.tail(1).index.tolist()])
        lines.append(f"- อดีต: เด่นสุด {past_top}; แผ่วสุด {past_bot}")
    except Exception:
        lines.append("- อดีต: ข้อมูลยังไม่พอสำหรับ Liquidity-Adj CAGR")

    # Present
    try:
        latest_beta = betas_df["Beta_vs_GLI"].sort_values(ascending=False)
        lines.append(f"- ปัจจุบัน: Beta สูงสุด = {_pretty_name(latest_beta.index[0])} | ต่ำสุด = {_pretty_name(latest_beta.index[-1])}")
    except Exception:
        lines.append("- ปัจจุบัน: ข้อมูลเบต้าไม่ครบ")

    # Forward (ตาม regime)
    try:
        avg = perf_regime_table["Avg_%/mo"]
        exp_winners = ", ".join([_pretty_name(x) for x in avg.loc[True].sort_values(ascending=False).head(2).index.tolist()])
        con_winners = ", ".join([_pretty_name(x) for x in avg.loc[False].sort_values(ascending=False).head(2).index.tolist()])
        lines.append(f"- อนาคต (สมมติ): ถ้า GLI ขยาย → เน้น {exp_winners}; ถ้า GLI หด → เน้น {con_winners}")
    except Exception:
        lines.append("- อนาคต: ข้อมูลระบอบไม่พอ")

    return "\n".join(lines)

# =========================
# Caching: ไม่ดึงใหม่ทุกครั้งตอนสลับหน้า
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def _load_all_cached(fred_api_key, start, years_for_cagr, risk_free_annual, include_pboc, pboc_series_id):
    # end=None เสมอสำหรับ app นี้
    return gl.load_all(
        fred_api_key=fred_api_key,
        start=start,
        end=None,
        years_for_cagr=years_for_cagr,
        risk_free_annual=risk_free_annual,
        include_pboc=include_pboc,
        pboc_series_id=pboc_series_id
    )

# ---------------- Load data ----------------
with st.spinner("Loading GLI & assets… (cached)"):
    data = _load_all_cached(
        fred_api_key=fred_key,
        start=start,
        years_for_cagr=years_n,
        risk_free_annual=rf_annual,
        include_pboc=False,
        pboc_series_id=None
    )

wk              = data["wk"]                 # weekly GLI proxy
monthly         = data["monthly"]            # GLI_INDEX + assets (M close)
monthly_rets    = data["monthly_rets"]       # %/mo
annual          = data["annual"]             # A-DEC close
metrics_table   = _flatten_cols(data["metrics_table"])      # summary table
corr_matrix     = _flatten_cols(data["corr_matrix"])
betas_df        = _flatten_cols(data["betas_df"])
rebased_m       = _flatten_cols(data["rebased_m"])          # for plotting
annual_yoy_fig  = data.get("annual_yoy_fig")                # GLI (line) vs assets (bars); อาจไม่มี

# Rolling
roll = gl.rolling_corr_beta_alpha(monthly_rets, window=win_m)
roll_corr_m_df = _flatten_cols(roll["corr"])
roll_beta_m_df = _flatten_cols(roll["beta"])
roll_alpha_m_df = _flatten_cols(roll["alpha"])

# Regime / Events
reg = gl.regime_and_events(monthly, monthly_rets)
regime_df      = reg["regime_df"]
exp_periods    = reg["expansion_periods"]
evt_up, evt_down = _to_display_df(reg["evt_up"]), _to_display_df(reg["evt_down"])
perf_regime_tbl = gl.perf_regime_table(monthly_rets, regime_df)

# ---------------- Title ----------------
st.title("GLI Dashboard")

# ---------------- KPI row ----------------
colA, colB, colC, colD, colE = st.columns(5)

# GLI CAGR
gli_full = gl.cagr_from_series(annual["GLI_INDEX"])
gli_n    = gl.cagr_last_n_years(annual["GLI_INDEX"], years_n)

# NASDAQ-GLI / GOLD-GLI (Liquidity-Adj)
nas_liq = _pick_liq_adj(metrics_table, "NASDAQ")
gold_liq = _pick_liq_adj(metrics_table, "GOLD")

colA.metric("GLI CAGR (full)", _fmt_pct(gli_full))
colB.metric(f"GLI CAGR ({years_n}y)", _fmt_pct(gli_n))
colC.metric("NASDAQ − GLI (CAGR)", _fmt_pct(nas_liq))
colD.metric("GOLD − GLI (CAGR)", _fmt_pct(gold_liq))
colE.metric("Sharpe (GLI regime mix)",
            f"{gl.sharpe(monthly_rets['GLI_INDEX'], rf_annual, 12):.2f}" if "GLI_INDEX" in monthly_rets else "—")

# ---------------- Tabs ----------------
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Rebased + Annual YoY", "📉 Rolling", "🧭 Regime & Events", "📋 Tables"]
)

# ---------- Tab 1 ----------
with tab_main:
    st.subheader("(Monthly) GLI vs NASDAQ / S&P 500 / Gold / BTC / ETH — Rebased = 100")

    options = list(rebased_m.columns)
    sel = set(st.multiselect("เลือกเส้นที่ต้องการแสดง", options=options, default=options,
                             key="rebased_sel", help="ซ่อน/แสดงซีรีส์ที่ต้องการเปรียบเทียบ"))

    fig_rebase = go.Figure()
    for col in rebased_m.columns:
        fig_rebase.add_trace(
            go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines",
                       name=_pretty_name(col),
                       visible=True if (col in sel) else "legendonly")
        )
    fig_rebase.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(rangeslider=dict(visible=True)),
        margin=dict(t=40, l=40, r=20, b=40),
        height=520
    )
    st.plotly_chart(fig_rebase, use_container_width=True, config={"displaylogo": False})

    # Annual YoY (fallback ถ้า lib ไม่ได้สร้างมาให้)
    if annual_yoy_fig is None:
        ann = monthly.resample("A-DEC").last().pct_change().dropna() * 100.0
        ann = ann.rename(columns={"GLI_INDEX": "GLI_%YoY"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"], mode="lines+markers", name="GLI %YoY"))
        for c in [c for c in ann.columns if c != "GLI_%YoY"]:
            fig.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{_pretty_name(c)} %YoY"))
        fig.update_layout(title="Annual YoY: GLI (line) vs Assets (bars)",
                          barmode="group", hovermode="x unified",
                          legend=dict(orientation="h", y=1.05),
                          xaxis=dict(rangeslider=dict(visible=True)))
        annual_yoy_fig = fig

    st.markdown("#### Annual YoY: GLI (line) vs Assets (bars)")
    st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 2 ----------
with tab_roll:
    st.subheader(f"Rolling {win_m}-Month Statistics vs GLI (Monthly Returns)")
    c1, c2 = st.columns(2)

    with c1:
        fig_rc = go.Figure()
        for col in [c for c in roll_corr_m_df.columns if c != "GLI_INDEX"]:
            fig_rc.add_trace(go.Scatter(x=roll_corr_m_df.index, y=roll_corr_m_df[col], mode="lines", name=_pretty_name(col)))
        fig_rc.update_layout(title=f"Rolling {win_m}M Correlation vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)),
                             yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})

    with c2:
        fig_rb = go.Figure()
        for col in [c for c in roll_beta_m_df.columns if c != "GLI_INDEX"]:
            fig_rb.add_trace(go.Scatter(x=roll_beta_m_df.index, y=roll_beta_m_df[col], mode="lines", name=_pretty_name(col)))
        fig_rb.update_layout(title=f"Rolling {win_m}M Beta vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    fig_ra = go.Figure()
    for col in [c for c in roll_alpha_m_df.columns if c != "GLI_INDEX"]:
        fig_ra.add_trace(go.Scatter(x=roll_alpha_m_df.index, y=roll_alpha_m_df[col], mode="lines", name=_pretty_name(col)))
    fig_ra.update_layout(title=f"Rolling {win_m}M Alpha vs GLI (approx, %/mo)",
                         hovermode="x unified",
                         legend=dict(orientation="h", y=1.02),
                         xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_ra, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 3 ----------
with tab_regime:
    st.subheader("GLI Regime (YoY>0 = Expansion) & Event Study")

    # Rebased + shaded expansion
    fig_reg = go.Figure()
    for col in rebased_m.columns:
        fig_reg.add_trace(go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines", name=_pretty_name(col)))
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

    # Event Study tables
    st.markdown("##### Event Study — ผลตอบแทนสะสมโดยเฉลี่ยหลังจุดเปลี่ยนระบอบ")
    st.caption("Upturn = GLI หด → ขยาย, Downturn = GLI ขยาย → หด; วัดสะสมถัดไป 3/6/12 เดือน")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**หลัง Upturn**")
        st.dataframe(evt_up, use_container_width=True, height=260)
    with c2:
        st.markdown("**หลัง Downturn**")
        st.dataframe(evt_down, use_container_width=True, height=260)

    # Summary (อันเดียวจบ)
    st.markdown("#### 📌 Summary")
    st.info(_make_summary(metrics_table, betas_df, evt_up, evt_down, perf_regime_tbl))

# ---------- Tab 4 ----------
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

