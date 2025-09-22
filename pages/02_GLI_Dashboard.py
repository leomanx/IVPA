# pages/02_GLI_Dashboard.py — GLI Dashboard (robust)
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import gli_lib as gl

st.set_page_config(page_title="GLI Dashboard", layout="wide")
st.title("GLI Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
fred_key = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", "")).strip()
start    = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
window   = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=3)
years_n  = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10)
rf_ann   = st.sidebar.number_input("Risk-free (annual)", 0.0, 0.10, 0.02, step=0.005)
refresh  = st.sidebar.button("♻️ Refresh cache")

if refresh:
    try:
        st.cache_data.clear()
        st.toast("Cache cleared", icon="♻️")
    except Exception:
        pass

# ---------------- Data loader (cached) ----------------
@st.cache_data(show_spinner=True, ttl=30*60)
def load_all(_fred_key, _start, _win, _years, _rf):
    fred = None
    if _fred_key:
        try:
            fred = gl.get_fred(_fred_key)
        except Exception as e:
            # Continue without fred → will fail later if GLI needed
            fred = None

    # Require FRED for GLI; if missing key, raise a friendly error
    if fred is None:
        raise RuntimeError("FRED API key not found/invalid — กรุณาใส่คีย์ใน .streamlit/secrets.toml หรือ ENV FRED_API_KEY")

    wk = gl.build_gli_proxy(fred, start=_start)
    assets = gl.fetch_assets(fred, start=_start)
    monthly, mrets = gl.monthly_panels(wk, assets)
    annual = gl.annual_panel(wk, assets)
    metrics, corr, betas = gl.metrics_tables(monthly, mrets, annual, rf_annual=_rf, years_n=_years)
    rc, rb, ra = gl.roll_metrics(monthly, mrets, window=_win)
    regime = gl.build_regime(monthly["GLI_INDEX"])
    up, down = gl.event_study(mrets, regime)
    return wk, assets, monthly, mrets, annual, metrics, corr, betas, rc, rb, ra, regime, up, down

# Try load
data = None
with st.spinner("Loading GLI & assets..."):
    try:
        data = load_all(fred_key, start, window, years_n, rf_ann)
    except Exception as e:
        st.error("โหลดข้อมูลไม่สำเร็จ — ตรวจ FRED_API_KEY / อินเทอร์เน็ต")
        with st.expander("รายละเอียดข้อผิดพลาด"):
            st.exception(e)

if not data:
    st.stop()

(wk, assets, monthly, mrets, annual, metrics, corr, betas,
 roll_corr, roll_beta, roll_alpha, regime, evt_up, evt_down) = data

# ---------------- Tabs ----------------
tabs = st.tabs(["Overview","Rebased","GLI vs GOLD","Rolling","Regime & Events","Tables"])

def plot_rebased(df: pd.DataFrame, title="Rebased to 100"):
    fig = go.Figure()
    for col in df.columns:
        s = df[col].dropna()
        if s.size < 2: continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=col))
    fig.update_layout(
        title=title,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        legend=dict(orientation="h", y=1.05)
    )
    return fig

with tabs[0]:
    st.subheader("Overview")
    st.caption("เทียบ **GLI** กับ NASDAQ / S&P500 / GOLD / BTC / ETH (รายเดือน) + Rolling/Regime/Event-study")
    st.markdown(f"- ข้อมูลเริ่มจาก: **{pd.to_datetime(start).date()}**")
    st.dataframe(assets.tail().astype(float).round(4))

with tabs[1]:
    st.subheader("Rebased to 100 (Monthly)")
    rb = pd.DataFrame({"GLI": gl.rebase(monthly["GLI_INDEX"])})
    for c in [x for x in monthly.columns if x != "GLI_INDEX"]:
        rb[c] = gl.rebase(monthly[c])
    st.plotly_chart(plot_rebased(rb, "GLI vs Assets — Rebased=100"), use_container_width=True)

with tabs[2]:
    st.subheader("GLI vs GOLD (Rebased)")
    if "GOLD" not in monthly.columns:
        st.warning("ไม่พบ GOLD ในช่วงวันที่เลือก")
    else:
        rb = pd.DataFrame({"GLI": gl.rebase(monthly["GLI_INDEX"]),
                           "GOLD": gl.rebase(monthly["GOLD"])})
        st.plotly_chart(plot_rebased(rb, "GLI vs GOLD — Rebased=100"), use_container_width=True)

with tabs[3]:
    st.subheader(f"Rolling ({window}M) — Correlation / Beta / Alpha vs GLI")
    if roll_corr.empty and roll_beta.empty and roll_alpha.empty:
        st.info("หน้าต่าง rolling ยังสั้นเกินไปสำหรับการคำนวณ")
    else:
        # Corr
        fc = go.Figure()
        for col in roll_corr.columns:
            fc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr[col], mode="lines", name=f"{col} Corr"))
        fc.update_layout(title="Rolling Correlation vs GLI", hovermode="x unified",
                         xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fc, use_container_width=True)

        # Beta
        fb = go.Figure()
        for col in roll_beta.columns:
            fb.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta[col], mode="lines", name=f"{col} β"))
        fb.update_layout(title="Rolling Beta vs GLI", hovermode="x unified",
                         xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fb, use_container_width=True)

        # Alpha
        fa = go.Figure()
        for col in roll_alpha.columns:
            fa.add_trace(go.Scatter(x=roll_alpha.index, y=roll_alpha[col], mode="lines", name=f"{col} α (%/mo)"))
        fa.update_layout(title="Rolling Alpha vs GLI (approx)", hovermode="x unified",
                         xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fa, use_container_width=True)

with tabs[4]:
    st.subheader("Rebased + GLI Expansion (YoY>0) shading")
    rb = pd.DataFrame({"GLI": gl.rebase(monthly["GLI_INDEX"])})
    for c in [x for x in monthly.columns if x != "GLI_INDEX"]:
        rb[c] = gl.rebase(monthly[c])

    # Build expansion blocks
    exp_blocks = []
    in_blk, s0 = False, None
    for t, ok in regime["GLI_Expansion"].items():
        if ok and not in_blk:
            in_blk, s0 = True, t
        if (not ok or t == regime.index[-1]) and in_blk:
            exp_blocks.append((s0, t))
            in_blk = False

    fig = plot_rebased(rb, "Rebased + GLI Expansion")
    for s, e in exp_blocks:
        fig.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Avg/Std Monthly Returns when GLI expands vs contracts**")
    try:
        align = mrets.join(regime["GLI_Expansion"], how="inner")
        avg = align.groupby("GLI_Expansion").mean().round(2)
        std = align.groupby("GLI_Expansion").std().round(2)
        st.dataframe(pd.concat({"Avg_%/mo": avg, "Std_%/mo": std}, axis=1))
    except Exception as e:
        st.warning("สร้างตารางไม่สำเร็จ"); st.exception(e)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Event Study — Upturns (False→True)**")
        st.dataframe(evt_up if not evt_up.empty else pd.DataFrame({"note": ["insufficient data"]}))
    with c2:
        st.markdown("**Event Study — Downturns (True→False)**")
        st.dataframe(evt_down if not evt_down.empty else pd.DataFrame({"note": ["insufficient data"]}))

with tabs[5]:
    st.subheader("Metrics (Liquidity-adjusted, Risk)")
    st.dataframe(metrics)
    st.subheader("Correlation (Monthly Returns)")
    st.dataframe(corr)
    st.subheader("Beta/Alpha vs GLI (Monthly OLS)")
    st.dataframe(betas)
