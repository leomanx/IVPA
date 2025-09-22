# pages/02_GLI_Dashboard.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import gli_lib as gl  # ต้องมีไฟล์ gli_lib.py ตามที่เราจัดโครงไว้

st.set_page_config(page_title="GLI Dashboard", layout="wide")

# ---------------- Helpers ----------------
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """แปลงคอลัมน์ tuple/MultiIndex ให้เป็นสตริงเรียบๆ"""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    new_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            new_cols.append(" - ".join(map(str, c)))
        else:
            new_cols.append(str(c))
    out.columns = new_cols
    return out

def safe_df_for_st(df: pd.DataFrame) -> pd.DataFrame:
    """ทำ DataFrame ให้ปลอดภัยต่อ st.dataframe()/Arrow"""
    if df is None:
        return pd.DataFrame()
    out = df.copy()

    # ถ้า index เป็น tuple/MultiIndex แปลงเป็นคอลัมน์ธรรมดา
    if isinstance(out.index, pd.MultiIndex) or any(isinstance(i, (tuple, list)) for i in out.index):
        out = out.reset_index().rename(columns={"index": "Index"})

    # แปลงค่า object ที่เป็น list/tuple/ndarray เป็นสตริง
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].apply(
                lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray)) else v
            )

    # บังคับ numeric ให้เป็นตัวเลข
    for c in out.select_dtypes(include=["float64","float32","int64","int32","Float64","Int64"]).columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def pct(x):
    return "—" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x*100:.2f}%"

def find_col(df: pd.DataFrame, *cands) -> str | None:
    """หาคอลัมน์แบบยืดหยุ่น: ตรงเป๊ะก่อน, ไม่งั้นหาแบบ partial (case-insensitive)"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = [str(c) for c in df.columns]

    # ตรงเป๊ะ (ลำดับผู้สมัครสำคัญสุดก่อน)
    for k in cands:
        if k in df.columns:
            return k
        if k in cols:
            # กรณีคอลัมน์เป็น tuple ที่ถูกแปลงเป็นสตริง
            return cols[cols.index(k)]

    # partial (case-insensitive)
    low = [c.lower() for c in cols]
    for k in cands:
        klow = k.lower()
        for i, name in enumerate(low):
            if klow in name:
                return cols[i]
    return None

def cagr_from_annual(annual_df: pd.DataFrame, *name_candidates) -> float:
    """ดึงซีรีส์จาก annual ด้วย find_col แล้วคำนวณ CAGR; หาไม่เจอคืน NaN"""
    col = find_col(annual_df, *name_candidates)
    if col is None:
        return np.nan
    try:
        return gl.cagr_from_series(annual_df[col])
    except Exception:
        return np.nan

# ---------------- Sidebar ----------------
st.sidebar.caption("GLI = Fed + ECB + BoJ − TGA − ONRRP (+PBoC optional)")
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

wk               = data.get("wk")                  # weekly GLI proxy
monthly          = _flatten_cols(data.get("monthly"))            # GLI_INDEX + assets (M close)
monthly_rets     = _flatten_cols(data.get("monthly_rets"))       # %/mo
annual           = _flatten_cols(data.get("annual"))             # A-DEC close
metrics_table    = _flatten_cols(data.get("metrics_table"))      # summary table
corr_matrix      = _flatten_cols(data.get("corr_matrix"))
betas_df         = _flatten_cols(data.get("betas_df"))
rebased_m        = _flatten_cols(data.get("rebased_m"))          # for plotting
annual_yoy_fig   = data.get("annual_yoy_fig")     # GLI (line) vs assets (bars) — อาจเป็น None

# ป้องกันกรณีไม่มี GLI_INDEX ใน monthly_rets (ชื่อเพี้ยน)
if monthly_rets is not None and "GLI_INDEX" not in monthly_rets.columns:
    # บางทีชื่อหลุดเป็น "GLI" ให้ map กลับ
    if "GLI" in monthly_rets.columns:
        monthly_rets = monthly_rets.rename(columns={"GLI": "GLI_INDEX"})

# Rolling stats
roll = gl.rolling_corr_beta_alpha(monthly_rets, window=win_m)
roll_corr_m_df, roll_beta_m_df, roll_alpha_m_df = (
    _flatten_cols(roll.get("corr")),
    _flatten_cols(roll.get("beta")),
    _flatten_cols(roll.get("alpha")),
)

# Regime & events
reg = gl.regime_and_events(monthly, monthly_rets)
regime_df      = _flatten_cols(reg.get("regime_df"))
exp_periods    = reg.get("expansion_periods", [])
evt_up         = _flatten_cols(reg.get("evt_up"))
evt_down       = _flatten_cols(reg.get("evt_down"))

# ---------------- Title ----------------
st.title("GLI Dashboard")

# ---------------- KPI row ----------------
colA, colB, colC, colD, colE = st.columns(5)

# GLI
gli_full = gl.cagr_from_series(annual["GLI_INDEX"]) if (isinstance(annual, pd.DataFrame) and "GLI_INDEX" in annual.columns) else np.nan
gli_n    = gl.cagr_last_n_years(annual["GLI_INDEX"], years_n) if (isinstance(annual, pd.DataFrame) and "GLI_INDEX" in annual.columns) else np.nan

# NASDAQ / GOLD — ใช้ชื่อสำรองหลายแบบ เผื่อคอลัมน์เป็น tuple หรือชื่อยาว
nas_full  = cagr_from_annual(annual, "NASDAQ", "^IXIC", "NASDAQCOM")
gold_full = cagr_from_annual(annual, "GOLD", "GC=F", "GLD")

colA.metric("GLI CAGR (full)", pct(gli_full))
colB.metric(f"GLI CAGR ({years_n}y)", pct(gli_n))
colC.metric("NASDAQ − GLI (CAGR)",
            pct(nas_full - gli_full) if np.isfinite(nas_full) and np.isfinite(gli_full) else "—")
colD.metric("GOLD − GLI (CAGR)",
            pct(gold_full - gli_full) if np.isfinite(gold_full) and np.isfinite(gli_full) else "—")
colE.metric("Sharpe(GLI monthly)",
            f"{gl.sharpe(monthly_rets['GLI_INDEX'], rf_annual, 12):.2f}"
            if (isinstance(monthly_rets, pd.DataFrame) and "GLI_INDEX" in monthly_rets.columns) else "—")

# ---------------- Tabs ----------------
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Rebased + Annual YoY", "📉 Rolling", "🧭 Regime & Events", "📋 Tables"]
)

# ---------- Tab 1: Rebased + Annual YoY ----------
with tab_main:
    st.subheader("Rebased (Monthly) — GLI vs NASDAQ / S&P500 / GOLD / BTC / ETH")

    # Multiselect ซ่อน/แสดงเส้น
    options = list(rebased_m.columns) if isinstance(rebased_m, pd.DataFrame) else []
    default = options
    sel = set(st.multiselect("เลือกซีรีส์", options=options, default=default, key="rebased_sel"))

    fig_rebase = go.Figure()
    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        for col in rebased_m.columns:
            fig_rebase.add_trace(
                go.Scatter(
                    x=rebased_m.index, y=rebased_m[col],
                    mode="lines", name=col,
                    visible=True if (col in sel or len(sel)==0) else "legendonly"
                )
            )
    fig_rebase.update_layout(
        title="Rebased = 100 (Monthly)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(rangeslider=dict(visible=True)),
        margin=dict(t=60, l=40, r=20, b=40),
        height=520
    )
    st.plotly_chart(fig_rebase, use_container_width=True, config={"displaylogo": False})

    st.markdown("#### Annual YoY — GLI (line) vs Assets (bars)")
    if annual_yoy_fig is None:
        # สร้าง fallback ถ้า library ไม่ได้ส่งกราฟมา
        ann = monthly.resample("A-DEC").last().pct_change().dropna() * 100.0 if isinstance(monthly, pd.DataFrame) else pd.DataFrame()
        ann = ann.rename(columns={"GLI_INDEX": "GLI_%YoY"}) if not ann.empty else ann
        fig = go.Figure()
        if not ann.empty and "GLI_%YoY" in ann.columns:
            fig.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"], mode="lines+markers", name="GLI %YoY"))
            for c in [c for c in ann.columns if c != "GLI_%YoY"]:
                fig.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{c} %YoY"))
        fig.update_layout(
            title="Annual YoY: GLI (line) vs Assets (bars)",
            barmode="group", hovermode="x unified",
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(rangeslider=dict(visible=True))
        )
        annual_yoy_fig = fig
    st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 2: Rolling ----------
with tab_roll:
    st.subheader(f"Rolling {win_m}-Month — Correlation / Beta / Alpha vs GLI")

    c1, c2 = st.columns(2)
    with c1:
        fig_rc = go.Figure()
        if isinstance(roll_corr_m_df, pd.DataFrame) and not roll_corr_m_df.empty:
            for col in [c for c in roll_corr_m_df.columns if c != "GLI_INDEX"]:
                fig_rc.add_trace(go.Scatter(x=roll_corr_m_df.index, y=roll_corr_m_df[col], mode="lines", name=col))
        fig_rc.update_layout(title=f"Rolling {win_m}M Correlation", hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)),
                             yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})

    with c2:
        fig_rb = go.Figure()
        if isinstance(roll_beta_m_df, pd.DataFrame) and not roll_beta_m_df.empty:
            for col in [c for c in roll_beta_m_df.columns if c != "GLI_INDEX"]:
                fig_rb.add_trace(go.Scatter(x=roll_beta_m_df.index, y=roll_beta_m_df[col], mode="lines", name=col))
        fig_rb.update_layout(title=f"Rolling {win_m}M Beta", hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    fig_ra = go.Figure()
    if isinstance(roll_alpha_m_df, pd.DataFrame) and not roll_alpha_m_df.empty:
        for col in [c for c in roll_alpha_m_df.columns if c != "GLI_INDEX"]:
            fig_ra.add_trace(go.Scatter(x=roll_alpha_m_df.index, y=roll_alpha_m_df[col], mode="lines", name=col))
    fig_ra.update_layout(title=f"Rolling {win_m}M Alpha (%/mo)", hovermode="x unified",
                         legend=dict(orientation="h", y=1.02),
                         xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_ra, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 3: Regime & Events ----------
with tab_regime:
    st.subheader("Regime & Event Study")

    # Rebased + ช่วง Expansion
    fig_reg = go.Figure()
    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        for col in rebased_m.columns:
            fig_reg.add_trace(go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines", name=col))
    for s, e in exp_periods:
        fig_reg.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.18, line_width=0)
    fig_reg.update_layout(title="Rebased (Monthly) + Expansion Shading",
                          hovermode="x unified",
                          legend=dict(orientation="h", y=1.02),
                          xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_reg, use_container_width=True, config={"displaylogo": False})

    # GLI YoY vs GOLD %/mo (dual axis) — จาก lib
    fig_gold_yoy = gl.gli_yoy_vs_gold(monthly, monthly_rets, regime_df, exp_periods)
    st.plotly_chart(fig_gold_yoy, use_container_width=True, config={"displaylogo": False})

    st.markdown("##### Event Study — ผลตอบแทนสะสมเฉลี่ยหลังจุดเปลี่ยนระบอบ")
    st.caption("**Upturn** = GLI จากหดตัว → ขยายตัว, **Downturn** = GLI จากขยายตัว → หดตัว; วัดผลสะสมถัดไป 3/6/12 เดือน")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**หลัง Upturn**")
        st.dataframe(
            safe_df_for_st(evt_up.round(2)).reset_index().rename(columns={'index': 'Asset'}),
            use_container_width=True
        )
    with c2:
        st.markdown("**หลัง Downturn**")
        st.dataframe(
            safe_df_for_st(evt_down.round(2)).reset_index().rename(columns={'index': 'Asset'}),
            use_container_width=True
        )
        
# -------- Summary --------
perf_tbl = gl.perf_regime_table(monthly_rets, regime_df)
summary_text = _make_summary(metrics_table, betas_df, evt_up, evt_down, perf_tbl)
st.markdown("#### 📌 Summary")
st.info(summary_text)

# ---------- Pretty summary (replace the old auto summary block) ----------
def _pretty_name(x):
    # tuple -> ใช้สมาชิกตัวแรก
    if isinstance(x, tuple) and len(x) > 0:
        x = x[0]
    s = str(x)
    # map ให้สั้นและสวย
    m = {
        "GLI": "GLI",
        "NASDAQ": "NASDAQ",
        "SP500": "S&P 500",
        "GOLD": "Gold",
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
    }
    # เผื่อเคสเป็นสตริงรวม เช่น "('GOLD','GC=F')"
    for k, v in m.items():
        if k in s:
            return v
    return s

def _safe_top_names(series, n=2):
    """รับ Series (index=asset, value=metric) -> คืนรายชื่อ top n แบบสวย"""
    if series is None or len(series.dropna()) == 0:
        return []
    return [_pretty_name(idx) for idx in series.dropna().sort_values(ascending=False).head(n).index]

def _make_summary(metrics_table, betas_df, evt_up, evt_down, perf_regime_tbl):
    lines = []

    # 1) Past — Liquidity-Adjusted CAGR
    try:
        mt = metrics_table.copy()
        if "Asset" in mt.columns:
            mt = mt.set_index("Asset")
        liq_cols = [c for c in mt.columns if "LiquidityAdj_CAGR" in c]
        if len(liq_cols) >= 1:
            liq_mean = mt[liq_cols].mean(axis=1)
            past_top = _safe_top_names(liq_mean, 2)
            past_bot = _safe_top_names(-liq_mean, 1)  # แย่สุด = ค่าน้อยสุด
            if past_top:
                lines.append(f"- **อดีต**: เด่นสุด {', '.join(past_top)}; แผ่วสุด {', '.join(past_bot) if past_bot else '—'}")
            else:
                lines.append("- **อดีต**: ข้อมูลยังไม่พอ")
        else:
            lines.append("- **อดีต**: ข้อมูลยังไม่พอ")
    except Exception:
        lines.append("- **อดีต**: ข้อมูลยังไม่พอ")

    # 2) Present — Beta สูง/ต่ำ
    try:
        bd = betas_df.copy()
        if "Beta_vs_GLI" in bd.columns and len(bd) > 0:
            bd2 = bd["Beta_vs_GLI"].dropna()
            if len(bd2) > 0:
                hi = _pretty_name(bd2.idxmax())
                lo = _pretty_name(bd2.idxmin())
                lines.append(f"- **ปัจจุบัน**: เบต้าเทียบ GLI สูงสุด **{hi}** | ต่ำสุด **{lo}**")
            else:
                lines.append("- **ปัจจุบัน**: ข้อมูลยังไม่พอ")
        else:
            lines.append("- **ปัจจุบัน**: ข้อมูลยังไม่พอ")
    except Exception:
        lines.append("- **ปัจจุบัน**: ข้อมูลยังไม่พอ")

    # 3) Forward — Regime tilt
    try:
        pr = perf_regime_tbl["Avg_%/mo"] if ("Avg_%/mo" in perf_regime_tbl) else None
        if pr is not None and (True in pr.index) and (False in pr.index):
            exp_winners = _safe_top_names(pr.loc[True], 2)
            con_winners = _safe_top_names(pr.loc[False], 2)
            lines.append(
                "- **อนาคต (ตามสถานการณ์)**: "
                f"ถ้า GLI **ขยาย** → เน้น {', '.join(exp_winners) if exp_winners else '—'}; "
                f"ถ้า GLI **หด** → เน้น {', '.join(con_winners) if con_winners else '—'}"
            )
        else:
            lines.append("- **อนาคต**: ข้อมูลยังไม่พอ")
    except Exception:
        lines.append("- **อนาคต**: ข้อมูลยังไม่พอ")

    return "สรุปย่อ (รายเดือน)\n" + "\n".join(lines)

# ใช้งาน:
perf_tbl = gl.perf_regime_table(monthly_rets, regime_df)
summary_text = _make_summary(metrics_table, betas_df, evt_up, evt_down, perf_tbl)
st.markdown("#### 📌 Auto Summary")
st.info(summary_text)

# ---------- Tab 4: Tables ----------
with tab_tables:
    st.subheader("Tables")

    with st.expander("📊 Metrics", expanded=True):
        st.dataframe(safe_df_for_st(metrics_table), use_container_width=True, height=340)

    c1, c2 = st.columns(2)
    with c1:
        with st.expander("🔗 Correlation (monthly %)", expanded=False):
            st.dataframe(safe_df_for_st(corr_matrix.round(2) if isinstance(corr_matrix, pd.DataFrame) else corr_matrix),
                         use_container_width=True, height=350)
    with c2:
        with st.expander("β vs GLI (Monthly OLS)", expanded=False):
            st.dataframe(
                safe_df_for_st(betas_df.round(3) if isinstance(betas_df, pd.DataFrame) else betas_df).reset_index().rename(columns={"index":"Asset"}),
                use_container_width=True, height=350
            )

    with st.expander("📈 Monthly closes (preview)", expanded=False):
        st.dataframe(safe_df_for_st(monthly.tail(12) if isinstance(monthly, pd.DataFrame) else monthly),
                     use_container_width=True, height=320)
