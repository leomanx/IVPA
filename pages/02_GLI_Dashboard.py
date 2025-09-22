import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gli_lib as gl

st.set_page_config(page_title="GLI Dashboard", layout="wide")
st.title("GLI Dashboard")

# ---------- Sidebar ----------
fred_key = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY","")).strip()
start    = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
end      = None
years_n  = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10)
rf_annual= st.sidebar.number_input("Risk-free (annual)", 0.0, 0.10, 0.02, step=0.005)
win      = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=3)
refresh  = st.sidebar.button("♻️ Refresh cache")

if refresh:
    st.cache_data.clear()
    st.toast("Cache cleared", icon="♻️")

# ---------- Loader (cached) ----------
@st.cache_data(show_spinner=True, ttl=30*60)
def load_all(_key, _start, _end, _years, _rf, _win):
    fred = gl.get_fred(_key)
    wk   = gl.build_gli_proxy(fred, start=_start, end=_end)
    assets = gl.fetch_assets(fred, start=_start, end=_end)
    monthly, mrets = gl.monthly_panels(wk, assets)
    annual = gl.annual_panel(wk, assets)
    metrics, corr, betas = gl.metrics_tables(monthly, mrets, annual, rf_annual=_rf, years_n=_years)
    rc, rb, ra = gl.roll_metrics(monthly, mrets, window=_win)
    regime = gl.build_regime(monthly["GLI_INDEX"])
    up, down = gl.event_study(mrets, regime)
    return wk, assets, monthly, mrets, annual, metrics, corr, betas, rc, rb, ra, regime, up, down

if not fred_key:
    st.error("ยังไม่มี FRED_API_KEY — ใส่ใน `.streamlit/secrets.toml` หรือ Secrets ของ Cloud")
    st.stop()

try:
    (wk, assets, monthly, mrets, annual, metrics, corr, betas,
     roll_corr, roll_beta, roll_alpha, regime, evt_up, evt_down) = load_all(
        fred_key, start, end, years_n, rf_annual, win
    )
except Exception as e:
    st.error("โหลดข้อมูลไม่สำเร็จ"); st.exception(e); st.stop()

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Rebased","Rolling","Regime & Events","Tables"])

def plot_rebased(df: pd.DataFrame, title="Rebased to 100"):
    fig = go.Figure()
    for col in df.columns:
        s = df[col].dropna()
        if s.size < 2: continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=col))
    fig.update_layout(title=title, hovermode="x unified",
                      xaxis=dict(rangeslider=dict(visible=True)),
                      legend=dict(orientation="h", y=1.05))
    return fig

with tab1:
    st.caption("GLI vs NASDAQ / S&P500 / GOLD / BTC / ETH")
    st.dataframe(assets.tail().round(4), use_container_width=True)

with tab2:
    # Rebased all + ปุ่มเลือกคู่
    rb = pd.DataFrame({"GLI": gl.rebase(monthly["GLI_INDEX"])})
    for c in [x for x in monthly.columns if x != "GLI_INDEX"]:
        rb[c] = gl.rebase(monthly[c])

    fig = go.Figure()
    cols = rb.columns.tolist()
    for col in cols:
        fig.add_trace(go.Scatter(x=rb.index, y=rb[col], mode="lines", name=col, visible=True))
    def vis(show): return [c in show for c in cols]
    fig.update_layout(
        title="(Monthly) GLI vs Assets — Rebased=100",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        legend=dict(orientation="h", y=1.14),
        updatemenus=[dict(
            type="buttons", x=0, y=1.22, direction="right",
            buttons=[
                dict(label="All", method="update", args=[{"visible":[True]*len(cols)}]),
                dict(label="GLI + GOLD", args=[{"visible":vis(['GLI','GOLD'])}], method="update"),
                dict(label="GLI + NASDAQ", args=[{"visible":vis(['GLI','NASDAQ'])}], method="update"),
                dict(label="GLI + SP500", args=[{"visible":vis(['GLI','SP500'])}], method="update"),
                dict(label="GLI + BTC", args=[{"visible":vis(['GLI','BTC'])}], method="update"),
                dict(label="GLI + ETH", args=[{"visible":vis(['GLI','ETH'])}], method="update"),
            ]
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Rolling Corr/Beta/Alpha
    if not roll_corr.empty:
        fc = go.Figure([go.Scatter(x=roll_corr.index, y=roll_corr[c], name=f"{c} Corr") for c in roll_corr.columns])
        fc.update_layout(title=f"Rolling {win}M Correlation vs GLI", xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fc, use_container_width=True)

    if not roll_beta.empty:
        fb = go.Figure([go.Scatter(x=roll_beta.index, y=roll_beta[c], name=f"{c} β") for c in roll_beta.columns])
        fb.update_layout(title=f"Rolling {win}M Beta vs GLI", xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fb, use_container_width=True)

    if not roll_alpha.empty:
        fa = go.Figure([go.Scatter(x=roll_alpha.index, y=roll_alpha[c], name=f"{c} α (%/mo)") for c in roll_alpha.columns])
        fa.update_layout(title=f"Rolling {win}M Alpha vs GLI (approx)", xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fa, use_container_width=True)

with tab4:
    # Rebased + ช่วง GLI expansion (YoY>0) เป็นพื้นเขียว
    rb = pd.DataFrame({"GLI": gl.rebase(monthly["GLI_INDEX"])})
    for c in [x for x in monthly.columns if x != "GLI_INDEX"]:
        rb[c] = gl.rebase(monthly[c])
    # หา block expansion
    blocks = []
    on, s0 = False, None
    for t, ok in regime["GLI_Expansion"].items():
        if ok and not on: on, s0 = True, t
        if (not ok or t == regime.index[-1]) and on:
            blocks.append((s0, t)); on = False
    fig = plot_rebased(rb, "Rebased + GLI Expansion (green)")
    for s, e in blocks: fig.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Avg/Std Monthly Returns: GLI Exp(True) vs Contraction(False)**")
    align = mrets.join(regime["GLI_Expansion"], how="inner")
    avg = align.groupby("GLI_Expansion").mean().round(2)
    std = align.groupby("GLI_Expansion").std().round(2)
    perf_table = pd.concat({"Avg_%/mo": avg, "Std_%/mo": std}, axis=1)
    st.dataframe(perf_table)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Event Study — Upturns (False→True)**")
        st.dataframe(evt_up if not evt_up.empty else pd.DataFrame({"note":["insufficient data"]}))
    with c2:
        st.markdown("**Event Study — Downturns (True→False)**")
        st.dataframe(evt_down if not evt_down.empty else pd.DataFrame({"note":["insufficient data"]}))

with tab5:
    st.subheader("Liquidity-Adjusted & Risk Metrics")
    st.dataframe(metrics, use_container_width=True)
    st.subheader("Correlation (Monthly Returns)")
    st.dataframe(corr, use_container_width=True)
    st.subheader("Beta/Alpha vs GLI (Monthly OLS)")
    st.dataframe(betas, use_container_width=True)

    # Downloads
    st.markdown("### ⬇️ Download CSV")
    def dl(df, name): st.download_button(f"Download {name}.csv", df.to_csv().encode(), f"{name}.csv", "text/csv")
    dl(monthly, "monthly_closes_GLI_assets")
    dl(mrets,   "monthly_returns_GLI_assets_pct_per_month")
    ann = annual.copy(); ann.insert(0, "Year", ann.index.year)
    dl(ann,     "annual_closes_GLI_assets")
    dl(metrics, "liquidity_adjusted_metrics_summary")
    dl(corr,    "quant_matrix_correlation_monthly")
    dl(betas,   "beta_vs_GLI_monthly")
