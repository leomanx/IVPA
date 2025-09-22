# pages/02_GLI_Dashboard.py
import os, streamlit as st, plotly.graph_objects as go, pandas as pd
import gli_lib as gl

st.set_page_config(page_title="GLI Dashboard", layout="wide")
st.title("GLI Dashboard")

# Sidebar
st.sidebar.header("Settings")
fred_key = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY",""))
if not fred_key: st.sidebar.warning("ใส่ FRED_API_KEY ใน secrets.toml หรือ ENV ครับ")
start   = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
window  = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=3)
years_n = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10)
rf_ann  = st.sidebar.number_input("Risk-free (annual)", 0.0, 0.1, 0.02, step=0.005)

@st.cache_data(show_spinner=True)
def load(_key,_start,_win,_years,_rf):
    fred = gl.get_fred(_key)
    wk = gl.build_gli_proxy(fred, start=_start)
    assets = gl.fetch_assets(fred, start=_start)
    monthly, mrets = gl.monthly_panels(wk, assets)
    annual = gl.annual_panel(wk, assets)
    metrics, corr, betas = gl.metrics_tables(monthly, mrets, annual, rf_annual=_rf, years_n=_years)
    rc, rb, ra = gl.roll_metrics(monthly, mrets, window=_win)
    regime = gl.build_regime(monthly["GLI_INDEX"])
    up, down = gl.event_study(mrets, regime)
    return wk, assets, monthly, mrets, annual, metrics, corr, betas, rc, rb, ra, regime, up, down

if fred_key:
    (wk, assets, monthly, mrets, annual, metrics, corr, betas,
     roll_corr, roll_beta, roll_alpha, regime, evt_up, evt_down) = load(fred_key, start, window, years_n, rf_ann)

tabs = st.tabs(["Overview","Rebased","GLI vs GOLD","Rolling","Regime & Events","Tables"])

with tabs[0]:
    st.subheader("Overview")
    st.write("เทียบ **GLI** กับ NASDAQ / S&P500 / GOLD / BTC / ETH รายเดือน + Rolling/Regime")
    if fred_key:
        st.dataframe(assets.tail())

with tabs[1]:
    st.subheader("Rebased to 100 (Monthly)")
    rb = pd.DataFrame({"GLI": gl.rebase(monthly["GLI_INDEX"])})
    for c in [x for x in monthly.columns if x!="GLI_INDEX"]:
        rb[c] = gl.rebase(monthly[c])
    fig = go.Figure()
    for col in rb.columns:
        fig.add_trace(go.Scatter(x=rb.index, y=rb[col], mode="lines", name=col))
    fig.update_layout(hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("GLI vs GOLD (rebased)")
    rb = pd.DataFrame({"GLI": gl.rebase(monthly["GLI_INDEX"]), "GOLD": gl.rebase(monthly["GOLD"])})
    fig = go.Figure()
    for col in rb.columns:
        fig.add_trace(go.Scatter(x=rb.index, y=rb[col], mode="lines", name=col))
    fig.update_layout(hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader(f"Rolling ({window}M) — Corr / Beta / Alpha")
    f1 = go.Figure()
    for c in roll_corr.columns: f1.add_trace(go.Scatter(x=roll_corr.index,y=roll_corr[c],mode="lines",name=f"{c} Corr"))
    f1.update_layout(hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True))); st.plotly_chart(f1, use_container_width=True)

    f2 = go.Figure()
    for c in roll_beta.columns: f2.add_trace(go.Scatter(x=roll_beta.index,y=roll_beta[c],mode="lines",name=f"{c} β"))
    f2.update_layout(hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True))); st.plotly_chart(f2, use_container_width=True)

    f3 = go.Figure()
    for c in roll_alpha.columns: f3.add_trace(go.Scatter(x=roll_alpha.index,y=roll_alpha[c],mode="lines",name=f"{c} α(%/mo)"))
    f3.update_layout(hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True))); st.plotly_chart(f3, use_container_width=True)

with tabs[4]:
    st.subheader("Rebased + GLI Expansion shading")
    rb = pd.DataFrame({"GLI": gl.rebase(monthly["GLI_INDEX"])})
    for c in [x for x in monthly.columns if x!="GLI_INDEX"]: rb[c]=gl.rebase(monthly[c])
    # สร้างบล็อกช่วง GLI_Expansion
    exp=[]; inblk=False; s0=None
    for t, ok in regime["GLI_Expansion"].items():
        if ok and not inblk: inblk=True; s0=t
        if (not ok or t==regime.index[-1]) and inblk:
            exp.append((s0,t)); inblk=False
    fig = go.Figure()
    for col in rb.columns: fig.add_trace(go.Scatter(x=rb.index,y=rb[col],mode="lines",name=col))
    for s,e in exp: fig.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
    fig.update_layout(hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Avg/Std Monthly Returns when GLI expands vs contracts**")
    align = mrets.join(regime["GLI_Expansion"], how="inner")
    st.dataframe(pd.concat({"Avg_%/mo": align.groupby("GLI_Expansion").mean().round(2),
                            "Std_%/mo": align.groupby("GLI_Expansion").std().round(2)}, axis=1))

    col1,col2 = st.columns(2)
    with col1: st.markdown("**Event Study — Upturns (False→True)**"); st.dataframe(evt_up)
    with col2: st.markdown("**Event Study — Downturns (True→False)**"); st.dataframe(evt_down)

with tabs[5]:
    st.subheader("Metrics"); st.dataframe(metrics)
    st.subheader("Correlation"); st.dataframe(corr)
    st.subheader("Beta vs GLI (OLS)"); st.dataframe(betas)
