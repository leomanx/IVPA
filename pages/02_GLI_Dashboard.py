"""
pages/02_GLI_Dashboard.py — GLI Dashboard v2
Sections: Overview | Rolling | Regime | Tables | Auto Summary
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── import gli_lib ────────────────────────────────────────────
try:
    import gli_lib as gl
except Exception as _e:
    st.error(f"ไม่สามารถ import gli_lib: {_e}")
    st.stop()

# ── Custom RdYlGn color helper (ไม่ต้องการ matplotlib) ──────────
def _rdylgn_style(vmin: float, vmax: float, alpha: float = 0.35):
    """
    คืน styling function สำหรับ Styler.map() ที่ให้สี Red→Yellow→Green
    ใช้แทน background_gradient(cmap='RdYlGn') เพื่อหลีกเลี่ยง matplotlib dependency
    """
    _range = max(float(vmax - vmin), 1e-9)
    def _fn(v):
        if not isinstance(v, (int, float)) or pd.isna(v):
            return ""
        t = max(0.0, min(1.0, (float(v) - vmin) / _range))
        if t < 0.5:
            s = t * 2                      # Red(0) → Yellow(0.5)
            r = int(214 + (255 - 214) * s) # 214→255
            g = int(39  + (215 - 39)  * s) # 39→215
            b = int(40  - 40 * s)          # 40→0
        else:
            s = (t - 0.5) * 2              # Yellow(0.5) → Green(1)
            r = int(255 - (255 - 44)  * s) # 255→44
            g = int(215 - (215 - 160) * s) # 215→160
            b = int(44 * s)                # 0→44
        return f"background-color:rgba({r},{g},{b},{alpha})"
    return _fn

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="GLI Dashboard", layout="wide", page_icon="🌊")

# ─── colour palette ─────────────────────────────────────────
_COLORS = {
    "GLI":    ("#1f77b4", "solid",   3.0),
    "NASDAQ": ("#ff7f0e", "solid",   1.8),
    "SP500":  ("#2ca02c", "dot",     1.8),
    "GOLD":   ("#c8b400", "dashdot", 1.8),
    "BTC":    ("#f7931a", "dash",    1.8),
    "ETH":    ("#627eea", "dash",    1.8),
}
def _color(name):
    c = _COLORS.get(name, ("#888888", "solid", 1.5))
    return dict(color=c[0], dash=c[1], width=c[2])

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")
    start_yr   = st.slider("Start Year", 2004, 2022, 2008)
    start      = f"{start_yr}-01-01"
    years_n    = st.slider("CAGR Window (years)", 5, 20, 10)
    rf_pct     = st.number_input("Risk-free Rate (%/yr)", 0.0, 10.0, 2.0, step=0.25)
    rf_annual  = rf_pct / 100.0
    st.divider()
    roll_win   = st.slider("Rolling Window (months)", 6, 36, 12)
    n_q        = st.slider("GLI Quantiles (Regime)", 3, 5, 5)
    max_lag    = st.slider("Max Lead/Lag (months)", 6, 18, 12)
    fwd_horiz  = st.multiselect("Forward Horizons (months)", [1,3,6,12,24], default=[1,3,6,12])
    if not fwd_horiz:
        fwd_horiz = [3]
    fwd_horiz = sorted(fwd_horiz)
    st.divider()
    show_norm  = st.checkbox("GLI Normalisation (M2 / Z-Score)", value=False,
                             help="เรียก FRED เพิ่มสำหรับ M2SL — อาจช้ากว่า")
    extra_ast  = st.checkbox("Extra Assets (Copper, DXY)", value=False,
                             help="เพิ่ม Copper + DXY Broad เข้า asset universe")
    show_plumb = st.checkbox("🔧 Fed Plumbing Panel", value=False,
                             help="Reserves, BTFP, MMF, HY Spread, Yield Curve, DXY, Copper")
    show_yen   = st.checkbox("🇯🇵 Yen Carry Trade Panel", value=False,
                             help="USDJPY, Carry State, Unwind Detection, VIX, BOJ Impact on GLI")
    if st.button("♻️ Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# API KEY
# ═══════════════════════════════════════════════════════════════
_raw   = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", ""))
fred_k = gl.sanitize_fred_key(_raw)

if not fred_k:
    st.warning("⚠️ ยังไม่พบ FRED_API_KEY — ใส่ใน App → Settings → Secrets")
    st.stop()
if not gl.validate_fred_key_format(fred_k):
    st.error(f"❌ รูปแบบ FRED_API_KEY ไม่ถูกต้อง ({gl.mask_key(fred_k)})")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# DATA LOADING  (cached, network-heavy)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3_600, show_spinner="⏳ ดึงข้อมูล FRED + Yahoo Finance…")
def _load(key, start, yrs, rf, norm, extra):
    return gl.load_all(
        fred_api_key=key, start=start,
        years_for_cagr=yrs, risk_free_annual=rf,
        normalize=norm, extra_assets=extra,
        pboc_series_id=None,
    )

try:
    D = _load(fred_k, start, int(years_n), rf_annual, show_norm, extra_ast)
except gl.FredKeyError as e:
    st.error("❌ ปัญหา FRED API Key")
    st.code(str(e))
    st.stop()
except Exception as e:
    st.error("เกิดข้อผิดพลาดระหว่างโหลดข้อมูล")
    st.exception(e)
    st.stop()

wk         = D["wk"]
monthly    = D["monthly"]
mr         = D["monthly_rets"]
annual     = D["annual"]
met_tbl    = D["metrics_table"]
corr_mx    = D["corr_matrix"]
betas_df   = D["betas_df"]
rebased_m  = D["rebased_m"]
ann_fig    = D["annual_yoy_fig"]
wk_norm    = D.get("wk_norm")

# ═══════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS  (cached per data+params, fast computation)
# ═══════════════════════════════════════════════════════════════
_adv_key = f"{start}_{fred_k[:4]}_{max_lag}_{tuple(fwd_horiz)}_{n_q}"

if st.session_state.get("_adv_key") != _adv_key:
    with st.spinner("🔬 คำนวณ Advanced Analysis…"):
        st.session_state["ll"]     = gl.lead_lag_analysis(mr, max_lag=max_lag)
        st.session_state["stats"]  = gl.statistical_tests(monthly, mr, max_lag_granger=6)
        st.session_state["fwd"]    = gl.forward_return_analysis(
                                         monthly, mr,
                                         horizons=fwd_horiz, n_quantiles=n_q)
        st.session_state["reg"]    = gl.regime_and_events(monthly, mr)
        st.session_state["prf"]    = gl.perf_regime_table(
                                         mr, st.session_state["reg"]["regime_df"])
        st.session_state["_adv_key"] = _adv_key

ll_res  = st.session_state["ll"]
st_res  = st.session_state["stats"]
fwd_res = st.session_state["fwd"]
reg     = st.session_state["reg"]
prf_tbl = st.session_state["prf"]

# ─── unpack regime ───────────────────────────────────────────
reg_df    = reg["regime_df"]
exp_per   = reg["expansion_periods"]
evt_up    = reg["evt_up"]
evt_dn    = reg["evt_down"]

# ═══════════════════════════════════════════════════════════════
# KPI STRIP
# ═══════════════════════════════════════════════════════════════
st.title("🌊 GLI Dashboard")
st.caption(f"Global Liquidity Index · อัปเดต {wk.index[-1].strftime('%d %b %Y')}")

gli_now  = wk["GLI_INDEX"].iloc[-1]
gli_wow  = wk["GLI_INDEX"].pct_change().iloc[-1] * 100
gli_yoy  = wk["GLI_INDEX"].pct_change(52).iloc[-1] * 100
is_exp   = gli_yoy > 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("GLI Index",  f"{gli_now:.1f}",  f"{gli_wow:+.2f}% WoW")
k2.metric("GLI YoY %",  f"{gli_yoy:+.1f}%",
          "🟢 Expansion" if is_exp else "🔴 Contraction")

if wk_norm is not None and "GLI_ZSCORE_36M" in wk_norm.columns:
    zs = wk_norm["GLI_ZSCORE_36M"].dropna().iloc[-1]
    z_note = ("🔥 Extreme high" if zs > 1.5 else
              "❄️ Extreme low"  if zs < -1.5 else "Normal range")
    k3.metric("Z-Score 36M", f"{zs:+.2f}", z_note)
    if "GLI_ACC" in wk_norm.columns:
        acc = wk_norm["GLI_ACC"].dropna().iloc[-1]
        k4.metric("GLI Acceleration", f"{acc:+.2f}%/mo",
                  "⬆️ Speeding up" if acc > 0 else "⬇️ Slowing down")
    else:
        k4.metric("GLI Acceleration", "—", "")
else:
    k3.metric("Z-Score 36M",   "—", "Enable Normalise ↙")
    k4.metric("GLI Acceleration", "—", "Enable Normalise ↙")

fwd_3m = fwd_res["fwd_yoy"].get("3M")
if fwd_3m is not None and not fwd_3m.dropna(how="all").empty:
    top_q    = fwd_3m.index[-1]
    best_a   = fwd_3m.loc[top_q].dropna().idxmax()
    best_v   = fwd_3m.loc[top_q].dropna().max()
    gli_pct  = ((monthly["GLI_INDEX"].pct_change(12).dropna()*100) <= gli_yoy).mean()*100
    k5.metric(f"Best Asset @ {top_q}", f"{best_a} ({best_v:+.1f}%)",
              f"GLI at {gli_pct:.0f}th pctile now")
else:
    k5.metric("Best Fwd Asset", "—", "")

st.divider()

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_ov, tab_roll, tab_reg, tab_tbl = st.tabs([
    "📈 Overview", "📉 Rolling", "🧭 Regime", "📋 Tables"
])

# ───────────────────────────────────────────────────────────────
# TAB 1 : OVERVIEW
# ───────────────────────────────────────────────────────────────
with tab_ov:
    # ── Rebased chart ─────────────────────────────────────────
    st.subheader("Price Index — GLI & Assets (Base = 100)")
    fig_reb = go.Figure()
    for col in rebased_m.columns:
        name = col if col != "GLI_INDEX" else "GLI"
        fig_reb.add_trace(go.Scatter(
            x=rebased_m.index, y=rebased_m[col].round(2),
            mode="lines", name=name, line=_color(name),
            hovertemplate=f"<b>{name}</b>: %{{y:.1f}}<extra></extra>",
        ))
    for s, e in exp_per:
        fig_reb.add_vrect(x0=s, x1=e, fillcolor="rgba(0,180,0,0.06)", line_width=0)
    fig_reb.update_layout(
        hovermode="x unified", height=460,
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis_title="Index (Base=100)",
    )
    st.plotly_chart(fig_reb, use_container_width=True)
    st.caption("พื้นที่เขียวอ่อน = GLI Expansion  |  Base = วันแรกของข้อมูล")

    # ── Annual YoY + Metrics side-by-side ─────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Annual YoY %")
        st.plotly_chart(ann_fig, use_container_width=True)
    with c2:
        st.subheader("Performance Summary")
        if not met_tbl.empty:
            num_cols = [c for c in met_tbl.columns if c != "Asset"]
            mdd_cols = [c for c in num_cols if "MaxDD" in c]
            pos_cols = [c for c in num_cols if c not in mdd_cols]
            styled = (
                met_tbl.style
                    .map(lambda v: "color:#2ca02c" if isinstance(v,(int,float)) and v>0
                              else ("color:#d62728" if isinstance(v,(int,float)) and v<0 else ""),
                              subset=pos_cols)
                    .map(lambda v: f"color:rgb({min(int(abs(v if isinstance(v,(int,float)) else 0)/80*255),255)},0,0)"
                              if isinstance(v,(int,float)) else "", subset=mdd_cols)
                    .format({c: "{:.2f}" for c in num_cols}, na_rep="—")
            )
            st.dataframe(styled, use_container_width=True, hide_index=True, height=320)

# ───────────────────────────────────────────────────────────────
# TAB 2 : ROLLING
# ───────────────────────────────────────────────────────────────
with tab_roll:
    roll_data  = gl.rolling_corr_beta_alpha(mr, window=roll_win)
    assets_av  = [c for c in mr.columns if c != "GLI_INDEX"]
    sel        = st.multiselect("เลือก Asset", assets_av, default=assets_av, key="roll_sel")
    if not sel:
        sel = assets_av

    _roll_panels = [
        ("corr",  f"Rolling Correlation vs GLI (window={roll_win}M)",  "Pearson r",   0.0),
        ("beta",  f"Rolling Beta vs GLI (window={roll_win}M)",          "Beta",        None),
        ("alpha", f"Rolling Alpha — residual return (window={roll_win}M)", "Alpha %/mo", 0.0),
    ]
    for key, title, ytitle, zero_line in _roll_panels:
        df_r = roll_data[key][sel].dropna(how="all")
        fig  = go.Figure()
        for col in df_r.columns:
            fig.add_trace(go.Scatter(x=df_r.index, y=df_r[col].round(4),
                                     mode="lines", name=col, line=_color(col)))
        if zero_line is not None:
            fig.add_hline(y=zero_line, line_dash="dash", line_color="rgba(120,120,120,0.5)")
        for s, e in exp_per:
            fig.add_vrect(x0=s, x1=e, fillcolor="rgba(0,180,0,0.05)", line_width=0)
        fig.update_layout(
            title=title, hovermode="x unified", height=300,
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(rangeslider=dict(visible=(key == "alpha"))),
            yaxis_title=ytitle,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Current Rolling Values (latest month)"):
        last_vals = {}
        for key in ("corr", "beta", "alpha"):
            last_vals[key] = roll_data[key][sel].dropna(how="all").iloc[-1].round(3)
        st.dataframe(pd.DataFrame(last_vals, index=sel).rename(
            columns={"corr":"Corr","beta":"Beta","alpha":"Alpha %/mo"}
        ), use_container_width=True)

# ───────────────────────────────────────────────────────────────
# TAB 3 : REGIME  (enhanced)
# ───────────────────────────────────────────────────────────────
with tab_reg:

    # ── 3-A  GLI YoY + Expansion Shading ──────────────────────
    st.subheader("GLI Cycle — Expansion / Contraction")
    fig_regime = gl.gli_yoy_vs_gold(monthly, mr, reg_df, exp_per)
    st.plotly_chart(fig_regime, use_container_width=True)

    # Expansion stats
    n_exp = sum((e - s).days for s, e in exp_per)
    n_tot = (wk.index[-1] - wk.index[0]).days
    st.caption(
        f"🟢 Expansion: {len(exp_per)} episodes · "
        f"{100*n_exp/max(n_tot,1):.0f}% of total time  |  "
        f"Avg duration: {n_exp//max(len(exp_per),1)//30:.0f} months"
    )

    # ── 3-B  Z-Score + Acceleration ───────────────────────────
    if wk_norm is not None:
        st.subheader("GLI Normalisation Signals")
        c_z, c_a = st.columns(2)

        with c_z:
            if "GLI_ZSCORE_36M" in wk_norm.columns:
                z = wk_norm["GLI_ZSCORE_36M"].dropna()
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(x=z.index, y=z.values, mode="lines", name="Z-Score",
                    fill="tozeroy", fillcolor="rgba(31,119,180,0.12)",
                    line=dict(color="#1f77b4", width=1.5)))
                for lvl, col, lbl in [(1.5,"#d62728","Extreme High"), (-1.5,"#1a56db","Extreme Low")]:
                    fig_z.add_hline(y=lvl, line_dash="dot", line_color=col,
                                    annotation_text=lbl, annotation_position="right")
                fig_z.update_layout(title="GLI Z-Score (36M rolling)", height=300,
                                    yaxis_title="Z", hovermode="x unified")
                st.plotly_chart(fig_z, use_container_width=True)

        with c_a:
            if "GLI_ACC" in wk_norm.columns:
                acc = wk_norm["GLI_ACC"].dropna()
                colors_acc = np.where(acc.values > 0, "#2ca02c", "#d62728")
                fig_a = go.Figure()
                fig_a.add_trace(go.Bar(x=acc.index, y=acc.values, name="Acceleration",
                                        marker_color=colors_acc))
                fig_a.add_hline(y=0, line_color="gray", line_width=0.8)
                fig_a.update_layout(title="GLI Acceleration Δ(YoY%)/month", height=300,
                                    yaxis_title="Δ YoY% per month", hovermode="x unified")
                st.plotly_chart(fig_a, use_container_width=True)

        if "GLI_M2_INDEX" in wk_norm.columns:
            st.markdown("**GLI / M2 Index** (raw GLI adjusted for US M2 growth)")
            m2i = wk_norm["GLI_M2_INDEX"].dropna()
            fig_m2 = go.Figure()
            fig_m2.add_trace(go.Scatter(x=m2i.index, y=m2i.values, mode="lines",
                                         name="GLI/M2 Index", line=dict(color="#9467bd", width=1.8)))
            fig_m2.add_trace(go.Scatter(x=wk["GLI_INDEX"].index, y=wk["GLI_INDEX"].values,
                                         mode="lines", name="GLI Index (raw)",
                                         line=dict(color="#1f77b4", dash="dot", width=1.2), opacity=0.6))
            fig_m2.update_layout(height=280, hovermode="x unified",
                                  legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig_m2, use_container_width=True)
    else:
        st.info("💡 เปิด **GLI Normalisation** ใน Sidebar เพื่อดู Z-Score, Acceleration, GLI/M2")

    st.divider()

    # ── 3-C  Lead / Lag ───────────────────────────────────────
    st.subheader("📡 Lead / Lag Analysis")
    c_ccf, c_opt = st.columns([3, 2])

    with c_ccf:
        st.plotly_chart(ll_res["fig"], use_container_width=True)

    with c_opt:
        st.markdown("**Optimal Lag per Asset**")
        opt_df = ll_res["optimal_lags"]
        def _dir_color(v):
            if "GLI leads" in str(v): return "color:#2ca02c;font-weight:bold"
            if "Asset leads" in str(v): return "color:#d62728"
            return ""
        def _sig_color(v):
            return "color:#2ca02c;font-weight:bold" if v == "✅" else ""
        st.dataframe(
            opt_df[["Asset","Optimal_Lag(mo)","Direction","Max_|Corr|","Sig_95%"]]
              .style.map(_dir_color, subset=["Direction"])
                    .map(_sig_color, subset=["Sig_95%"])
                    .format({"Optimal_Lag(mo)": "{:.0f}", "Max_|Corr|": "{:.3f}"}, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
        ci = ll_res["ci95"]
        n_mo = len(mr.dropna())
        st.caption(f"95% CI: ±{ci:.3f}  |  N = {n_mo} months")

        # Quick interpretation
        leaders = opt_df[opt_df["Optimal_Lag(mo)"] > 0]
        if not leaders.empty:
            best = leaders.sort_values("Max_|Corr|", ascending=False).iloc[0]
            st.success(
                f"GLI นำ **{best['Asset']}** ที่ lag **{int(best['Optimal_Lag(mo)'])} เดือน** "
                f"(r = {best['Max_|Corr|']:.3f})"
            )

    st.divider()

    # ── 3-D  Forward Return Heatmaps ──────────────────────────
    st.subheader("🔮 Predictive — Forward Return by GLI Quantile")

    heat_yoy, heat_mom = st.tabs(["📊 GLI YoY Signal (Cycle)", "⚡ GLI MoM Signal (Momentum)"])

    def _draw_heatmap(df, title, h, y_suffix, n_q):
        """Draw a single forward-return heatmap."""
        if df is None or df.dropna(how="all").empty:
            st.info("ข้อมูลไม่เพียงพอสำหรับ heatmap นี้")
            return
        _max = max(abs(df.stack().dropna().abs().max()), 0.01)
        z    = df.values.tolist()
        txt  = [[f"{v:.1f}%" if pd.notna(v) else "—" for v in row] for row in z]
        ylab = [f"{idx}  ({y_suffix})" for idx in df.index]
        fig  = go.Figure(go.Heatmap(
            z=z, x=df.columns.tolist(), y=ylab,
            colorscale="RdYlGn", zmid=0, zmin=-_max, zmax=_max,
            text=txt, texttemplate="%{text}",
            colorbar=dict(title=f"{h}M Fwd %", len=0.8),
            hovertemplate="Asset: %{x}<br>GLI Bin: %{y}<br>Avg Fwd Return: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{title}<br><sup>Q1 / M1 = GLI weakest · Q{n_q} / M{n_q} = GLI strongest</sup>",
            height=320, margin=dict(l=170),
        )
        st.plotly_chart(fig, use_container_width=True)

    with heat_yoy:
        h_y = st.select_slider("Horizon", fwd_horiz,
                                value=min(3, max(fwd_horiz)), key="hy")
        _draw_heatmap(fwd_res["fwd_yoy"].get(f"{h_y}M"),
                      f"Avg {h_y}-Month Forward Return — GLI YoY Quantile",
                      h_y, "GLI YoY%", n_q)

        hit_y = fwd_res["hit_rate_yoy"].get(f"{h_y}M")
        if hit_y is not None:
            with st.expander(f"Hit Rate — % เดือนที่ผลตอบแทน > 0  (horizon {h_y}M)"):
                st.dataframe(
                    hit_y.style.map(_rdylgn_style(30, 70))
                               .format("{:.0f}%", na_rep="—"),
                    use_container_width=True,
                )

        with st.expander("ทุก Horizon — GLI YoY Signal"):
            for h in fwd_horiz:
                df_h = fwd_res["fwd_yoy"].get(f"{h}M")
                if df_h is not None:
                    st.markdown(f"**{h}M:**")
                    _v = df_h.stack().dropna()
                    _vmin, _vmax = (float(_v.min()), float(_v.max())) if not _v.empty else (-20, 20)
                    st.dataframe(
                        df_h.style.map(_rdylgn_style(_vmin, _vmax))
                                  .format("{:.1f}%", na_rep="—"),
                        use_container_width=True,
                    )

    with heat_mom:
        h_m = st.select_slider("Horizon", fwd_horiz,
                                value=min(3, max(fwd_horiz)), key="hm")
        _draw_heatmap(fwd_res["fwd_mom"].get(f"{h_m}M"),
                      f"Avg {h_m}-Month Forward Return — GLI MoM Quantile",
                      h_m, "GLI MoM%", n_q)

        hit_m = fwd_res["hit_rate_mom"].get(f"{h_m}M")
        if hit_m is not None:
            with st.expander(f"Hit Rate — % เดือนที่ผลตอบแทน > 0  (horizon {h_m}M)"):
                st.dataframe(
                    hit_m.style.map(_rdylgn_style(30, 70))
                               .format("{:.0f}%", na_rep="—"),
                    use_container_width=True,
                )

    st.divider()

    # ── 3-E  Performance by Regime ────────────────────────────
    st.subheader("📊 Performance by GLI Regime")
    c_prf, c_evt = st.columns(2)

    with c_prf:
        st.markdown("**Avg Monthly Return (%) — Expansion vs Contraction**")
        try:
            avg_r = prf_tbl["Avg_%/mo"].copy()
            std_r = prf_tbl["Std_%/mo"].copy()
            if "GLI_INDEX" in avg_r.columns:
                avg_r = avg_r.drop(columns=["GLI_INDEX"])
                std_r = std_r.drop(columns=["GLI_INDEX"])
            avg_r.index = avg_r.index.map({True: "🟢 Expansion", False: "🔴 Contraction"})
            std_r.index = std_r.index.map({True: "🟢 Expansion", False: "🔴 Contraction"})
            st.dataframe(
                avg_r.style.map(_rdylgn_style(
                    float(avg_r.stack().dropna().min()) if not avg_r.stack().dropna().empty else -5,
                    float(avg_r.stack().dropna().max()) if not avg_r.stack().dropna().empty else  5,
                )).format("{:.2f}%", na_rep="—"),
                use_container_width=True,
            )
            with st.expander("Std Dev by Regime"):
                st.dataframe(std_r.style.format("{:.2f}%", na_rep="—"),
                             use_container_width=True)
        except Exception as _e:
            st.dataframe(prf_tbl, use_container_width=True)

    with c_evt:
        st.markdown("**Event Study — Cumulative Return After Regime Change**")
        ev_up_tab, ev_dn_tab = st.tabs(["↑ After Upturn (C→E)", "↓ After Downturn (E→C)"])

        def _style_event(df):
            _v = df.stack().dropna()
            _lo = float(_v.min()) if not _v.empty else -20
            _hi = float(_v.max()) if not _v.empty else  20
            return (df.style
                      .map(_rdylgn_style(_lo, _hi))
                      .format("{:.1f}%", na_rep="—"))

        with ev_up_tab:
            if evt_up is not None and not evt_up.empty:
                st.dataframe(_style_event(evt_up), use_container_width=True)
                st.caption("ค่าเฉลี่ยผลตอบแทนสะสมหลัง GLI พลิกกลับจาก Contraction → Expansion")
            else:
                st.info("ข้อมูลไม่พอสำหรับ event study")

        with ev_dn_tab:
            if evt_dn is not None and not evt_dn.empty:
                st.dataframe(_style_event(evt_dn), use_container_width=True)
                st.caption("ค่าเฉลี่ยผลตอบแทนสะสมหลัง GLI พลิกกลับจาก Expansion → Contraction")
            else:
                st.info("ข้อมูลไม่พอสำหรับ event study")

# ───────────────────────────────────────────────────────────────
# TAB 4 : TABLES  (enhanced)
# ───────────────────────────────────────────────────────────────
with tab_tbl:

    # ── 4-A  Metrics ──────────────────────────────────────────
    st.subheader("📊 Performance Metrics")
    if not met_tbl.empty:
        num_cols = [c for c in met_tbl.columns if c != "Asset"]
        mdd_cols = [c for c in num_cols if "MaxDD" in c]
        pos_cols = [c for c in num_cols if c not in mdd_cols]
        st.dataframe(
            met_tbl.style
                .map(lambda v: "color:#2ca02c" if isinstance(v,(int,float)) and v>0
                          else ("color:#d62728" if isinstance(v,(int,float)) and v<0 else ""),
                          subset=pos_cols)
                .map(lambda v: f"background-color:rgba(214,39,40,"
                          f"{min(abs(v if isinstance(v,(int,float)) else 0)/80,1)*0.35})"
                          if isinstance(v,(int,float)) else "", subset=mdd_cols)
                .format({c: "{:.2f}" for c in num_cols}, na_rep="—"),
            use_container_width=True, hide_index=True, height=240,
        )

    # ── 4-B  Correlation Heatmap ──────────────────────────────
    st.subheader("🔗 Correlation Matrix — Monthly Returns")
    cols_c = corr_mx.columns.tolist()
    fig_cx = go.Figure(go.Heatmap(
        z=corr_mx.round(3).values.tolist(),
        x=cols_c, y=cols_c,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr_mx.round(2).values.tolist(),
        texttemplate="%{text}",
        colorbar=dict(title="r", len=0.8),
    ))
    fig_cx.update_layout(
        height=380,
        xaxis=dict(side="bottom"),
        title="Pearson Correlation (monthly %MoM, full period)",
    )
    st.plotly_chart(fig_cx, use_container_width=True)

    # ── 4-C  Beta / Alpha ─────────────────────────────────────
    st.subheader("📐 Beta & Alpha vs GLI — Full Period OLS")
    if not betas_df.empty:
        def _beta_bg(v):
            if not isinstance(v, (int, float)): return ""
            intensity = min(abs(v) / 2.5, 1.0) * 0.35
            return (f"background-color:rgba(44,160,44,{intensity:.2f})" if v > 0
                    else f"background-color:rgba(214,39,40,{intensity:.2f})")
        st.dataframe(
            betas_df.style
                .map(_beta_bg, subset=["Beta_vs_GLI"])
                .map(lambda v: "color:#2ca02c" if isinstance(v,(int,float)) and v>0
                          else ("color:#d62728" if isinstance(v,(int,float)) and v<0 else ""),
                          subset=["Alpha_%/mo"])
                .format("{:.4f}", na_rep="—"),
            use_container_width=True,
        )

    st.divider()

    # ── 4-D  Statistical Validity ─────────────────────────────
    st.subheader("🔬 Statistical Validity Tests")

    def _sig_style(styler, col, true_val="✅"):
        return styler.map(
            lambda v: "color:#2ca02c;font-weight:bold" if v == true_val else "",
            subset=[col]
        )

    s_adf, s_coint, s_gr = st.tabs([
        "ADF — Stationarity",
        "Engle-Granger — Cointegration",
        "Granger Causality",
    ])

    with s_adf:
        st.markdown("""
**ADF (Augmented Dickey-Fuller)** ทดสอบว่า series เป็น I(0) หรือ I(1)

| ผล | ความหมาย |
|---|---|
| ✅ Stationary | ใช้ regression level ได้โดยตรง |
| ❌ Unit root (level) | ควร first-difference ก่อน หรือตรวจ cointegration |
| ❌ Unit root (return) | ⚠️ return ยัง non-stationary — ระวัง spurious regression |
        """)
        adf = st_res.get("adf_table", pd.DataFrame())
        if not adf.empty:
            st.dataframe(
                _sig_style(
                    adf.style.format({"ADF_stat":"{:.3f}","p_value":"{:.4f}",
                                      "CV_1%":"{:.3f}","CV_5%":"{:.3f}"}, na_rep="—"),
                    "Stationary"
                ),
                use_container_width=True, hide_index=True,
            )

    with s_coint:
        st.markdown("""
**Engle-Granger Cointegration** ทดสอบว่า GLI และ Asset มี long-run equilibrium

| ผล | ความหมาย |
|---|---|
| ✅ Cointegrated | long-run tie มีอยู่ → ECM ใช้ได้, regression level ไม่ spurious |
| — Not cointegrated | ใช้ returns เท่านั้น, regression level อาจ spurious |
        """)
        coint = st_res.get("coint_table", pd.DataFrame())
        if not coint.empty:
            st.dataframe(
                _sig_style(
                    coint.style.format({"EG_stat":"{:.3f}","p_value":"{:.4f}",
                                        "CV_5%":"{:.3f}"}, na_rep="—"),
                    "Cointegrated"
                ),
                use_container_width=True, hide_index=True,
            )

    with s_gr:
        st.markdown("""
**Granger Causality: GLI%MoM → Asset%MoM**
ทดสอบว่า *ข้อมูลอดีตของ GLI* ช่วยพยากรณ์ Asset ได้จริงหรือไม่
(เชิง predictive; ไม่ใช่เชิงสาเหตุ mechanistic)

| ผล | ความหมาย |
|---|---|
| ✅ p < 0.05 | GLI อดีตมีนัยสำคัญในการพยากรณ์ Asset นั้น |
| —  p ≥ 0.05 | ไม่มีหลักฐาน Granger-predictive |
        """)
        gr = st_res.get("granger_table", pd.DataFrame())
        if not gr.empty:
            disp = ["GLI → Asset","Best_Lag(mo)","F_stat","p_value","GLI_causes_Asset"]
            st.dataframe(
                _sig_style(
                    gr[disp].style.format({"F_stat":"{:.3f}","p_value":"{:.4f}",
                                           "Best_Lag(mo)":"{:.0f}"}, na_rep="—"),
                    "GLI_causes_Asset"
                ),
                use_container_width=True, hide_index=True,
            )
            with st.expander("p-value ทุก lag"):
                st.dataframe(
                    gr[["GLI → Asset","All_lag_p"]].rename(
                        columns={"All_lag_p":"p-value per lag"}),
                    use_container_width=True, hide_index=True,
                )

        # ── Lead/Lag summary table (cross-reference) ──────────
        st.markdown("---")
        st.markdown("**Lead/Lag Optimal Lags (quick reference)**")
        opt = ll_res["optimal_lags"]
        st.dataframe(
            _sig_style(
                opt[["Asset","Optimal_Lag(mo)","Direction","Max_|Corr|","Corr_lag0","Sig_95%"]]
                   .style.format({"Optimal_Lag(mo)":"{:.0f}","Max_|Corr|":"{:.3f}",
                                  "Corr_lag0":"{:.3f}"}, na_rep="—"),
                "Sig_95%"
            ),
            use_container_width=True, hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════
# AUTO SUMMARY  (bottom, always visible)
# ═══════════════════════════════════════════════════════════════
st.divider()
st.subheader("📝 Auto Summary")

col_c, col_a = st.columns(2)

with col_c:
    with st.expander("📊 Classic Summary", expanded=True):
        try:
            txt = gl.auto_summary(met_tbl, betas_df, evt_up, evt_dn, prf_tbl)
            st.text(txt)
        except Exception as _e:
            st.warning(f"Classic summary error: {_e}")

with col_a:
    with st.expander("🔬 Advanced Summary", expanded=True):
        try:
            txt = gl.advanced_summary(ll_res, st_res, fwd_res)
            st.text(txt)
        except Exception as _e:
            st.warning(f"Advanced summary error: {_e}")

# ── Full Report ───────────────────────────────────────────────
with st.expander("📋 Full Narrative Report (Download)"):
    try:
        lines = [
            "═══ GLI DASHBOARD FULL REPORT ═══",
            f"Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC",
            f"Data range: {wk.index[0].strftime('%d %b %Y')} → {wk.index[-1].strftime('%d %b %Y')}",
            f"Settings  : start={start}  CAGR={years_n}Y  Rf={rf_pct:.2f}%  Q={n_q}  Lag={max_lag}M",
            "",
            "──── 1. CURRENT GLI STATE ────────────────────────────────",
            f"GLI Index : {gli_now:.1f}   WoW: {gli_wow:+.2f}%   YoY: {gli_yoy:+.1f}%",
            f"Regime    : {'🟢 EXPANSION' if is_exp else '🔴 CONTRACTION'}",
        ]

        if wk_norm is not None and "GLI_ZSCORE_36M" in wk_norm.columns:
            zs = wk_norm["GLI_ZSCORE_36M"].dropna().iloc[-1]
            lines.append(f"Z-Score   : {zs:+.2f}  "
                         f"({'Extreme High' if zs>1.5 else 'Extreme Low' if zs<-1.5 else 'Normal'})")
        lines.append("")

        # Lead/Lag
        opt = ll_res["optimal_lags"]
        lines.append("──── 2. LEAD / LAG FINDINGS ──────────────────────────────")
        for _, r in opt.iterrows():
            sig = "✅ Sig" if r.get("Sig_95%") == "✅" else "—"
            lines.append(f"  {r['Asset']:8s} lag={int(r['Optimal_Lag(mo)']):+3d}M  "
                         f"r={r['Max_|Corr|']:.3f}  {r['Direction']}  {sig}")
        lines.append("")

        # Granger
        gr = st_res.get("granger_table", pd.DataFrame())
        lines.append("──── 3. GRANGER CAUSALITY (GLI → Asset) ─────────────────")
        for _, r in gr.iterrows():
            lines.append(f"  {r['GLI → Asset']:8s}  p={r['p_value']:.4f}  "
                         f"lag={r['Best_Lag(mo)']}M  {r['GLI_causes_Asset']}")
        lines.append("")

        # Predictive outlook
        gli_pctile = ((monthly["GLI_INDEX"].pct_change(12).dropna()*100) <= gli_yoy).mean()*100
        lines.append("──── 4. PREDICTIVE OUTLOOK ────────────────────────────────")
        lines.append(f"Current GLI YoY: {gli_yoy:+.1f}%  (historical {gli_pctile:.0f}th percentile)")
        for h in fwd_horiz:
            df_h = fwd_res["fwd_yoy"].get(f"{h}M")
            if df_h is None: continue
            top_row = df_h.iloc[-1].dropna().sort_values(ascending=False)
            bot_row = df_h.iloc[0].dropna().sort_values(ascending=False)
            lines.append(f"  {h:2d}M fwd if GLI=Q{n_q} (strong): "
                         + "  ".join(f"{a}:{v:+.1f}%" for a, v in top_row.items()))
            lines.append(f"  {h:2d}M fwd if GLI=Q1 (weak) : "
                         + "  ".join(f"{a}:{v:+.1f}%" for a, v in bot_row.items()))
        lines.append("")

        # Classic + Advanced
        try:
            lines.append("──── 5. CLASSIC ANALYSIS ──────────────────────────────────")
            lines.append(gl.auto_summary(met_tbl, betas_df, evt_up, evt_dn, prf_tbl))
            lines.append("")
        except Exception:
            pass
        try:
            lines.append("──── 6. ADVANCED ANALYSIS ─────────────────────────────────")
            lines.append(gl.advanced_summary(ll_res, st_res, fwd_res))
        except Exception:
            pass

        report = "\n".join(lines)
        st.text(report)
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report,
            file_name=f"GLI_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )
    except Exception as _e:
        st.warning(f"Report error: {_e}")

st.caption("© 2025 — GLI Dashboard v2  |  FRED + Yahoo Finance  |  gl.advanced_summary()")

# ═══════════════════════════════════════════════════════════════
# 🔧 FED PLUMBING PANEL  (optional, below main dashboard)
# ═══════════════════════════════════════════════════════════════
if show_plumb:
    st.divider()
    st.header("🔧 Fed Plumbing & Stealth Liquidity")
    st.markdown("""
> **Fed มีเครื่องมือควบคุมสภาพคล่องแบบ "อ้อมๆ"** ที่ไม่ปรากฏใน headline WALCL
> แต่ส่งผลต่อตลาดอย่างมีนัยสำคัญ — ติดตามรายการเหล่านี้เพื่อเห็น "น้ำจริง" ในระบบ
    """)

    # Fetch (cached separately, start from 2020 เพื่อเห็น COVID + BTFP cycle)
    plumb_start = st.selectbox("Plumbing Start Year", ["2015","2018","2020","2022","2024","2025","2026"],
                               index=2, key="plumb_start")

    @st.cache_data(ttl=3_600, show_spinner="🔬 ดึง Fed Plumbing data…")
    def _plumb(key, s):
        return gl.fed_plumbing(key, start=s)

    try:
        P = _plumb(fred_k, f"{plumb_start}-01-01")

        # ── Summary Table ─────────────────────────────────────
        st.subheader("📋 Current Readings")
        tbl = P["summary_tbl"]
        if not tbl.empty:
            def _mom_color(v):
                if not isinstance(v, (int, float)): return ""
                return "color:#2ca02c" if v > 0 else "color:#d62728"
            num_cols = ["Latest", "MoM %", "YoY %"]
            st.dataframe(
                tbl.style
                    .map(_mom_color, subset=["MoM %", "YoY %"])
                    .format({c: "{:.2f}" for c in num_cols}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )

        st.caption(
            "Reserve Balances + BTFP = stealth inject  |  "
            "MMF ⬆ = เงินหนีความเสี่ยง  |  "
            "HY Spread ⬇ = liquidity ส่งถึงตลาด  |  "
            "DXY ⬆ = ดูด global dollar"
        )

        st.divider()

        # ── 3 charts row ─────────────────────────────────────
        c_inj, c_str = st.columns([3, 2])
        with c_inj:
            st.plotly_chart(P["fig_inject"], use_container_width=True)
        with c_str:
            st.plotly_chart(P["fig_stress"], use_container_width=True)

        c_glb, c_cu = st.columns([3, 2])
        with c_glb:
            st.plotly_chart(P["fig_global"], use_container_width=True)
        with c_cu:
            st.plotly_chart(P["fig_copper"], use_container_width=True)

        # ── Net Fed Liquidity ─────────────────────────────────
        net = P.get("net_fed_liq", pd.Series(dtype=float))
        if not net.dropna().empty:
            st.subheader("🧮 Net Fed Liquidity = Reserves + BTFP")
            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(
                x=net.index, y=net.round(1).values,
                mode="lines", name="Net Fed Liq (B USD)",
                fill="tozeroy", fillcolor="rgba(31,119,180,0.13)",
                line=dict(color="#1f77b4", width=2),
            ))
            fig_net.update_layout(
                height=280, hovermode="x unified",
                yaxis_title="Billions USD",
                xaxis=dict(rangeslider=dict(visible=True)),
            )
            st.plotly_chart(fig_net, use_container_width=True)
            st.caption(
                "Net Fed Liq = Reserve Balances + Emergency Loans(BTFP)  "
                "↑ สูง = ธนาคารมีสภาพคล่องมาก → risk-on  "
                "↓ ต่ำ = ธนาคารระวังตัว → ระวัง stress event"
            )

        # ── Interpretation Guide ─────────────────────────────
        with st.expander("📖 วิธีอ่าน Fed Plumbing"):
            st.markdown("""
| Instrument | Signal บวก (inject) | Signal ลบ (drain) |
|---|---|---|
| **Reserve Balances** | สูง = ธนาคารมีเงินสำรองมาก → กล้าปล่อยสินเชื่อ/risk | ต่ำ = ธนาคารระวัง → credit tightening |
| **BTFP/Emergency Loans** | พุ่งขึ้น = Fed กำลัง inject แบบเงียบๆ (SVB 2023) | ≈ 0 = ไม่มีวิกฤต |
| **MMF Assets** | ลด = เงินไหลออกจาก MMF → risk assets ได้รับเงิน | เพิ่ม = เงินหนีความเสี่ยง จอดใน MMF |
| **HY Spread** | ต่ำ (<300bps) = ตลาดเปิดรับความเสี่ยง | สูง (>500bps) = credit stress แม้ GLI ดี ก็ไม่ส่งผ่าน |
| **10Y-2Y Curve** | บวก + steepen = growth expected | ลบ (inversion) = recession signal |
| **DXY Broad** | อ่อน = dollar ไหลออกสู่โลก → EM/crypto ดี | แข็ง = ดูด global dollar กลับ US |
| **China FX** | เพิ่ม = PBOC inject / ไม่ต้องป้องหยวน | ลด = PBOC ขาย USD ป้องค่าหยวน = ดูด global liquidity |
| **Copper** | ขึ้น = demand จริง → GLI expansion มักตามมา | ลง = ความต้องการชะลอ → GLI อาจหด |

**วิธีอ่านแบบรวม:**
- GLI ขึ้น + Reserves ขึ้น + HY Spread แคบ = **Full Liquidity Expansion** → risk-on strong
- GLI ขึ้น + MMF สูง + HY Spread กว้าง = **Liquidity Trapped** → เงินมีแต่ยังไม่ถึงตลาด
- GLI ลง + BTFP พุ่ง = **Stealth Support** → Fed อาจกำลัง backstop วิกฤตเงียบๆ
- DXY แข็ง + China FX ลด = **Global Dollar Squeeze** → EM/BTC ระวัง
            """)

    except gl.FredKeyError as e:
        st.error(f"FRED Key Error: {e}")
    except Exception as e:
        st.error("Fed Plumbing data load error")
        st.exception(e)

# ═══════════════════════════════════════════════════════════════
# 🇯🇵 YEN CARRY TRADE PANEL
# ═══════════════════════════════════════════════════════════════
if show_yen:
    st.divider()
    st.header("🇯🇵 Yen Carry Trade & Unwind Analysis")
    st.markdown("""
> **Yen Carry Trade** = กู้ JPY ดอกเบี้ย ~0% → ลงทุนใน USD risk assets
> เมื่อ JPY แข็งกลับอย่างรวดเร็ว → **Unwind** = ขาย risk assets บังคับพร้อมกันทั่วโลก
> 
> 🔗 **GLI Connection:** BOJ_USD = BOJ_JPY ÷ USDJPY
> ดังนั้น JPY แข็ง → BOJ_USD ขึ้น → GLI formula ขึ้น **แต่ตลาดร่วง** — Contradiction ที่ต้องระวัง
    """)

    yen_start = st.selectbox("Yen Analysis Start Year",
                             ["2012","2015","2018","2020","2022"],
                             index=2, key="yen_start")

    @st.cache_data(ttl=3_600, show_spinner="🇯🇵 วิเคราะห์ Yen Carry Trade…")
    def _yen(wk_json, key, s):
        import io
        _wk = pd.read_json(io.StringIO(wk_json))
        _wk.index = pd.to_datetime(_wk.index)
        return gl.yen_carry_analysis(_wk, fred_api_key=key, start=s)

    try:
        YC = _yen(wk.to_json(), fred_k, f"{yen_start}-01-01")
        cs = YC["current_state"]

        # ── KPI Strip ─────────────────────────────────────────
        st.subheader("📊 Current Yen Carry State")
        y1, y2, y3, y4, y5 = st.columns(5)
        y1.metric("USD/JPY Now",
                  f"{cs['USDJPY']:.2f}",
                  f"{cs['MoM_%']:+.2f}% (4W)" if pd.notna(cs['MoM_%']) else "—")
        y2.metric("3-Month Change",
                  f"{cs['3M_%']:+.1f}%" if pd.notna(cs['3M_%']) else "—",
                  cs["Unwind_Status"])
        y3.metric("Carry State", cs["Carry_State"], "")
        y4.metric("VIX",
                  f"{cs['VIX_latest']:.1f}" if pd.notna(cs['VIX_latest']) else "—",
                  "🔴 Panic" if pd.notna(cs['VIX_latest']) and cs['VIX_latest'] > 30
                  else ("⚠️ Caution" if pd.notna(cs['VIX_latest']) and cs['VIX_latest'] > 20 else "🟢 Calm"))
        y5.metric("BOJ Share of GLI",
                  f"{cs['BOJ_GLI_share_%']:.1f}%" if pd.notna(cs['BOO_GLI_share_%'] if 'BOO_GLI_share_%' in cs else cs.get('BOJ_GLI_share_%')) else "—",
                  "ยิ่งสูง = USDJPY มีผลกับ GLI มากขึ้น")

        # ── Interpretation Box ────────────────────────────────
        unwind_txt = cs["Unwind_Status"]
        if "MAJOR" in unwind_txt or "🚨" in unwind_txt:
            st.error(f"🚨 {unwind_txt} — พิจารณา defensive allocation ทันที")
        elif "Minor" in unwind_txt or "⚠️" in unwind_txt:
            st.warning(f"⚠️ {unwind_txt} — ติดตามอย่างใกล้ชิด")
        else:
            st.success(f"✅ {unwind_txt}")

        st.divider()

        # ── Charts Row 1 ──────────────────────────────────────
        st.plotly_chart(YC["fig_usdjpy"], use_container_width=True)

        c_mom, c_vix = st.columns(2)
        with c_mom:
            st.plotly_chart(YC["fig_mom"], use_container_width=True)
        with c_vix:
            st.plotly_chart(YC["fig_vix"], use_container_width=True)

        # ── BOJ Impact on GLI ─────────────────────────────────
        st.plotly_chart(YC["fig_boj_impact"], use_container_width=True)

        # ── Unwind Events Table ───────────────────────────────
        st.subheader("📋 Unwind Events History")
        uw = YC["unwind_events"]
        if not uw.empty:
            def _sev_bg(v):
                if v == "Severe": return "background-color:rgba(214,39,40,0.25);font-weight:bold"
                if v == "Major":  return "background-color:rgba(255,127,14,0.20)"
                if v == "Minor":  return "background-color:rgba(255,200,0,0.15)"
                return ""

            uw_disp = uw.copy()
            for col in ["Start","Peak"]:
                if col in uw_disp.columns:
                    uw_disp[col] = pd.to_datetime(uw_disp[col]).dt.strftime("%d %b %Y")
            st.dataframe(
                uw_disp.style.map(_sev_bg, subset=["Severity"])
                             .format({"USDJPY_at_Peak": "{:.2f}",
                                      "USDJPY_drop_%": "{:.2f}%"}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )
            st.caption(f"พบ {len(uw)} unwind events | "
                       f"Severe: {(uw['Severity']=='Severe').sum()} | "
                       f"Major: {(uw['Severity']=='Major').sum()} | "
                       f"Minor: {(uw['Severity']=='Minor').sum()}")
        else:
            st.info("ไม่พบ unwind events ในช่วงเวลาที่เลือก")

        # ── Yen Carry Guide ───────────────────────────────────
        with st.expander("📖 คู่มือ Yen Carry Trade สำหรับมือใหม่"):
            st.markdown("""
**Yen Carry Trade คืออะไร?**
1. กู้เงิน JPY ในอัตราดอกเบี้ย ~0% (BOJ คงนโยบายผ่อนคลายมาทศวรรษ)
2. แปลง JPY → USD (ขาย JPY, ซื้อ USD → JPY อ่อนค่า USDJPY ขึ้น)
3. ลงทุนใน US Stocks, Bonds, หรือ risk assets อื่นๆ
4. รับกำไรจากส่วนต่างดอกเบี้ย + capital gain จาก USD assets
5. เมื่อ USD อ่อน/JPY แข็ง → กำไรน้อยลง → ถ้าขาดทุน → **ต้องปิด position**

**ทำไม Unwind ถึงรุนแรง?**
- Carry trade เป็น leveraged position ขนาดใหญ่มาก (ประมาณ $4 Trillion)
- เมื่อ JPY แข็งกะทันหัน → ทุกคน unwind พร้อมกัน → ขาย risk assets พร้อมกัน
- ก่อให้เกิด cascade: JPY แข็ง → BTC ลง → NASDAQ ลง → VIX พุ่ง → panic ยิ่งใหญ่

**Unwind Signals:**
| Signal | ค่า | ความหมาย |
|---|---|---|
| USDJPY 4W change | < −3% | Minor unwind เริ่ม |
| USDJPY 4W change | < −7% | Major unwind — defensive |
| USDJPY 4W change | < −12% | Severe (Aug 2024 level) |
| VIX | > 20 | ตลาดเริ่มตื่นกลัว |
| VIX | > 30 | Panic zone — carry unwind มักอยู่ที่นี่ |
| VIX | > 40 | Crisis level |

**Historical Unwind Events:**
| ปี | USDJPY Drop | สาเหตุ | ผลกระทบ |
|---|---|---|---|
| 2008 | 127→88 (−31%) | GFC | หุ้นโลกร่วง 50%+ |
| 2011 | ค่าเงินผันผวนหลังสึนามิ | Disaster + BOJ intervention | เยนแข็งมาก |
| 2016 | Brexit + Trump | ความไม่แน่นอนทางการเมือง | เยนแข็งสั้นๆ |
| 2022 | BOJ เริ่มขึ้นดอกเบี้ย | YCC policy change | เยนอ่อนก่อน แล้วแข็ง |
| ส.ค. 2024 | 157→141 (−10.2%) | BOJ ขึ้น rate surprise | BTC −20%, Nikkei −12% ใน 3 วัน |

**GLI Contradiction:**
- เมื่อ JPY แข็ง → BOJ_USD (ส่วนประกอบของ GLI) เพิ่มขึ้น mechanically
- ทำให้ GLI formula คำนวณได้ว่า "สภาพคล่องเพิ่ม"
- แต่ในความเป็นจริง carry unwind = สภาพคล่องลด + risk-off
- → ดู **GLI Acceleration** ประกอบ: ถ้าลบ = สภาพคล่องจริงลด แม้ GLI formula ขึ้น
            """)

    except Exception as _e:
        st.error("Yen Carry Analysis error")
        st.exception(_e)

st.caption("© 2025 — GLI Dashboard v2  |  FRED + Yahoo Finance")
