# pages/02_GLI_Dashboard.py
import os, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import gli_lib as gl

# ------------- Page config -------------
st.set_page_config(page_title="GLI Dashboard", layout="wide")

# ------------- Helpers -------------
def _fmt_pct(x):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "—"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

def _flatten_index(idx) -> pd.Index:
    """แปลง MultiIndex/DatetimeIndex เป็น flat string สำหรับแสดงผล/Arrow"""
    if isinstance(idx, pd.MultiIndex):
        return pd.Index([" / ".join([str(x) for x in tup]) for tup in idx.tolist()])
    if isinstance(idx, pd.DatetimeIndex):
        return pd.Index([d.strftime("%Y-%m-%d") for d in idx.to_pydatetime()])
    return pd.Index([str(x) for x in idx.tolist()])

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """แปลงคอลัมน์ให้เป็นสตริงเสมอ (กัน Arrow)"""
    if not isinstance(df, pd.DataFrame):
        return df
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            cols.append(" / ".join([str(x) for x in c]))
        else:
            cols.append(str(c))
    out = df.copy()
    out.columns = cols
    return out

def _to_display_df(obj) -> pd.DataFrame:
    """บังคับเป็น DataFrame + flatten index/cols + แปลง dtype เป็นแสดงผลได้ (กัน Arrow)"""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.Series):
        out = obj.to_frame(name=getattr(obj, "name", "value"))
    elif isinstance(obj, pd.DataFrame):
        out = obj.copy()
    else:
        return pd.DataFrame()

    out = _flatten_cols(out)

    # แปลง index -> คอลัมน์ text "Index" เพื่อเลี่ยง Arrow issues
    try:
        out = out.copy()
        out.insert(0, "Index", _flatten_index(out.index))
        out = out.reset_index(drop=True)
    except Exception:
        # ถ้า insert ไม่ได้ก็ reset_index ปกติ
        out = out.reset_index()

    # แปลง object columns ที่ปน list/tuple/datetime ให้เป็น string
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            try:
                out[c] = out[c].apply(lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else (str(v) if not isinstance(v, (int, float, np.number)) else v))
            except Exception:
                out[c] = out[c].astype(str)

    return out

def _col_of(df: pd.DataFrame, logical_name: str) -> str | None:
    """หา 'ชื่อคอลัมน์จริง' ใน df สำหรับ logical_name ('GLI_INDEX','NASDAQ','GOLD') แบบยืดหยุ่น"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = [str(c) for c in df.columns]

    ALIASES = {
        "GLI_INDEX": ["GLI_INDEX", "GLI", "GLI INDEX"],
        "NASDAQ":    ["NASDAQ", "^IXIC", "NDX", "NASDAQCOM"],
        "GOLD":      ["GOLD", "GC=F", "XAU", "XAUUSD", "GLD"],
    }
    targets = [logical_name] + ALIASES.get(logical_name, [])

    # exact match ก่อน
    for t in targets:
        for c in cols:
            if c.strip().upper() == t.strip().upper():
                return c
    # contains match (รองรับชื่อยาวเช่น 'NASDAQ: FRED (NASDAQCOM)')
    for t in targets:
        tU = t.strip().upper()
        for c in cols:
            if tU in c.strip().upper():
                return c
    return None

def _pick_series(df: pd.DataFrame, logical_name: str) -> pd.Series:
    """หยิบ series ตาม logical_name ถ้าไม่เจอคืน Series ว่าง"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    col = _col_of(df, logical_name)
    if col and col in df.columns:
        return df[col]
    return pd.Series(dtype=float)

def _annual_series_from_monthly(monthly_df: pd.DataFrame, logical_name: str) -> pd.Series:
    """สร้าง Annual close (A-DEC) จาก monthly panel ถ้า annual ไม่มี"""
    if not isinstance(monthly_df, pd.DataFrame) or monthly_df.empty:
        return pd.Series(dtype=float)
    col = _col_of(monthly_df, logical_name)
    if not col:
        return pd.Series(dtype=float)
    try:
        return monthly_df[[col]].resample("A-DEC").last()[col]
    except Exception:
        return pd.Series(dtype=float)

def _safe_get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if isinstance(df, pd.DataFrame) and col in df.columns:
        return df[col]
    return pd.Series(dtype=float)

# ------------- Sidebar -------------
st.sidebar.caption("GLI: Fed + ECB + BoJ − TGA − ONRRP (+PBoC optional)")
start     = st.sidebar.text_input("Start (YYYY-MM-DD)", "2008-01-01")
years_n   = st.sidebar.number_input("CAGR lookback (years)", 5, 25, 10, step=1)
rf_annual = st.sidebar.number_input("Risk-free (annual)", 0.00, 0.10, 0.02, step=0.0025, format="%.4f")
win_m     = st.sidebar.slider("Rolling window (months)", 6, 36, 12, step=1)

# FRED key: secrets > env
fred_key = (st.secrets.get("FRED_API_KEY", "") or os.environ.get("FRED_API_KEY","")).strip()

# ปุ่มรีเฟรช cache
def _refresh_cache():
    st.session_state.pop("gli_data_cache", None)
    st.cache_data.clear()
st.sidebar.button("🔄 Refresh cache", on_click=_refresh_cache)

# ------------- Load data (persist ข้ามการสลับเพจ) -------------
param_key = (start, int(years_n), float(rf_annual), fred_key)
cache_pack = st.session_state.get("gli_data_cache")
if cache_pack is None or cache_pack.get("params") != param_key:
    with st.spinner("Loading GLI & assets..."):
        data = gl.load_all(
            fred_api_key=fred_key,
            start=start,
            end=None,
            years_for_cagr=int(years_n),
            risk_free_annual=float(rf_annual),
            include_pboc=False,
            pboc_series_id=None
        )
        st.session_state["gli_data_cache"] = {"params": param_key, "data": data}
else:
    data = cache_pack["data"]

# ปลอดภัยเผื่อ None
wk              = data.get("wk", pd.DataFrame())
monthly         = data.get("monthly", pd.DataFrame())
monthly_rets    = data.get("monthly_rets", pd.DataFrame())
annual          = data.get("annual", pd.DataFrame())
metrics_table   = data.get("metrics_table", pd.DataFrame())
corr_matrix     = data.get("corr_matrix", pd.DataFrame())
betas_df        = data.get("betas_df", pd.DataFrame())
rebased_m       = data.get("rebased_m", pd.DataFrame())
annual_yoy_fig  = data.get("annual_yoy_fig", None)

# Rolling
roll = gl.rolling_corr_beta_alpha(monthly_rets, window=int(win_m))
roll_corr_m_df, roll_beta_m_df, roll_alpha_m_df = roll["corr"], roll["beta"], roll["alpha"]

# Regime + events
reg = gl.regime_and_events(monthly, monthly_rets)
regime_df      = reg["regime_df"]
exp_periods    = reg["expansion_periods"]
evt_up, evt_down = reg["evt_up"], reg["evt_down"]

# ------------- Title -------------
st.title("GLI Dashboard")

# ------------- KPI row (แก้ไขส่วนนี้) -------------
colA, colB, colC, colD, colE = st.columns(5)

# ซีรีส์ annual โดยเน้น GLI/NASDAQ/GOLD (ถ้า annual ไม่มี ให้ fallback จาก monthly)
gli_ser_a  = _pick_series(annual, "GLI_INDEX")
nas_ser_a  = _pick_series(annual, "NASDAQ")
gold_ser_a = _pick_series(annual, "GOLD")

if gli_ser_a.empty:
    gli_ser_a  = _annual_series_from_monthly(monthly, "GLI_INDEX")
if nas_ser_a.empty:
    nas_ser_a  = _annual_series_from_monthly(monthly, "NASDAQ")
if gold_ser_a.empty:
    gold_ser_a = _annual_series_from_monthly(monthly, "GOLD")

# ฟังก์ชันสำรอง CAGR calculation
def _calc_cagr_safe(series: pd.Series) -> float:
    """คำนวณ CAGR แบบปลอดภัย fallback ถ้า gl.cagr_from_series() ไม่ทำงาน"""
    if series.empty or len(series) < 2:
        return np.nan
    try:
        # ลองใช้ฟังก์ชันจาก library ก่อน
        result = gl.cagr_from_series(series)
        if pd.notna(result) and result is not None:
            return float(result)
    except:
        pass
    
    # Fallback: คำนวณเอง
    try:
        series_clean = series.dropna()
        if len(series_clean) < 2:
            return np.nan
        start_val = series_clean.iloc[0]
        end_val = series_clean.iloc[-1]
        if start_val <= 0 or end_val <= 0:
            return np.nan
        years = len(series_clean) / 12.0  # สมมติเป็นข้อมูลรายเดือน
        if years <= 0:
            return np.nan
        cagr = (end_val / start_val) ** (1/years) - 1
        return cagr
    except:
        return np.nan

# คำนวณ CAGR
gli_full = _calc_cagr_safe(gli_ser_a)
nas_full = _calc_cagr_safe(nas_ser_a)
gold_full = _calc_cagr_safe(gold_ser_a)

# สำหรับ GLI N years - ลองใช้ library function ก่อน
try:
    gli_n = gl.cagr_last_n_years(gli_ser_a, int(years_n))
    if gli_n is None or pd.isna(gli_n):
        # Fallback: หา N years ล่าสุด
        gli_n_years = gli_ser_a.tail(int(years_n) * 12) if len(gli_ser_a) >= int(years_n) * 12 else gli_ser_a
        gli_n = _calc_cagr_safe(gli_n_years)
except:
    gli_n = np.nan

# Debug info (แสดงค่าใน sidebar เพื่อตรวจสอบ)
with st.sidebar:
    st.write("Debug Info:")
    st.write(f"GLI CAGR: {gli_full} (type: {type(gli_full)})")
    st.write(f"NASDAQ CAGR: {nas_full} (type: {type(nas_full)})")
    st.write(f"GOLD CAGR: {gold_full} (type: {type(gold_full)})")
    st.write(f"GLI series length: {len(gli_ser_a)}")
    st.write(f"NASDAQ series length: {len(nas_ser_a)}")
    st.write(f"GOLD series length: {len(gold_ser_a)}")
    
    # แสดงคอลัมน์ที่พบ
    st.write("Found columns in monthly:")
    if isinstance(monthly, pd.DataFrame):
        st.write(list(monthly.columns))
        st.write(f"GLI col: {_col_of(monthly, 'GLI_INDEX')}")
        st.write(f"NASDAQ col: {_col_of(monthly, 'NASDAQ')}")
        st.write(f"GOLD col: {_col_of(monthly, 'GOLD')}")
    
    # แสดงคอลัมน์ที่พบใน annual  
    st.write("Found columns in annual:")
    if isinstance(annual, pd.DataFrame):
        st.write(list(annual.columns))
    
    # แสดงตัวอย่างข้อมูล
    if not gli_ser_a.empty:
        st.write(f"GLI first/last: {gli_ser_a.iloc[0]:.2f} / {gli_ser_a.iloc[-1]:.2f}")
    if not nas_ser_a.empty:
        st.write(f"NASDAQ first/last: {nas_ser_a.iloc[0]:.2f} / {nas_ser_a.iloc[-1]:.2f}")
    if not gold_ser_a.empty:
        st.write(f"GOLD first/last: {gold_ser_a.iloc[0]:.2f} / {gold_ser_a.iloc[-1]:.2f}")

# แสดง KPIs
colA.metric("GLI CAGR (full)", _fmt_pct(gli_full))
colB.metric(f"GLI CAGR ({int(years_n)}y)", _fmt_pct(gli_n))

# คำนวณ liquidity premium - แก้ใหม่
nas_liq = np.nan
gold_liq = np.nan

st.sidebar.write(f"Before calc - GLI: {gli_full}, NASDAQ: {nas_full}, GOLD: {gold_full}")

# ตรวจสอบและคำนวณ NASDAQ - GLI
if gli_full is not None and nas_full is not None and not pd.isna(gli_full) and not pd.isna(nas_full):
    try:
        nas_liq = float(nas_full) - float(gli_full)
        st.sidebar.write(f"NASDAQ - GLI calculated: {nas_liq}")
    except Exception as e:
        st.sidebar.write(f"Error calc NASDAQ-GLI: {e}")
else:
    st.sidebar.write(f"Cannot calc NASDAQ-GLI: GLI={gli_full}, NASDAQ={nas_full}")

# ตรวจสอบและคำนวณ GOLD - GLI  
if gli_full is not None and gold_full is not None and not pd.isna(gli_full) and not pd.isna(gold_full):
    try:
        gold_liq = float(gold_full) - float(gli_full)
        st.sidebar.write(f"GOLD - GLI calculated: {gold_liq}")
    except Exception as e:
        st.sidebar.write(f"Error calc GOLD-GLI: {e}")
else:
    st.sidebar.write(f"Cannot calc GOLD-GLI: GLI={gli_full}, GOLD={gold_full}")

colC.metric("NASDAQ − GLI (CAGR)", _fmt_pct(nas_liq))
colD.metric("GOLD − GLI (CAGR)",   _fmt_pct(gold_liq))

# Sharpe ratio
shp_gli = np.nan
try:
    gli_col = _col_of(monthly_rets, "GLI_INDEX")
    if gli_col and isinstance(monthly_rets, pd.DataFrame) and gli_col in monthly_rets.columns:
        shp_gli = gl.sharpe(monthly_rets[gli_col], float(rf_annual), 12)
except Exception:
    shp_gli = np.nan

colE.metric("Sharpe (GLI)", f"{shp_gli:.2f}" if pd.notna(shp_gli) else "—")

# ------------- Tabs -------------
tab_main, tab_roll, tab_regime, tab_tables = st.tabs(
    ["📈 Rebased + Annual YoY", "📉 Rolling", "🧭 Regime & Events", "📋 Tables"]
)

# ---------- Tab 1: Rebased + Annual YoY ----------
with tab_main:
    st.subheader("(Monthly) GLI vs NASDAQ / S&P500 / GOLD / BTC / ETH — Rebased = 100")

    # toggle แสดง/ซ่อน series
    options_all = list(rebased_m.columns) if isinstance(rebased_m, pd.DataFrame) else []
    sel = st.multiselect(
        "เลือกเส้นที่ต้องการแสดง",
        options=options_all,
        default=options_all,
        key="rebased_sel",
        help="ซ่อน/แสดงซีรีส์ที่ต้องการเปรียบเทียบ"
    )
    sel_set = set(sel)

    fig_rebase = go.Figure()
    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        for col in rebased_m.columns:
            fig_rebase.add_trace(
                go.Scatter(
                    x=rebased_m.index, y=rebased_m[col],
                    mode="lines", name=str(col),
                    visible=True if col in sel_set else "legendonly"
                )
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
    # fallback ถ้าใน gli_lib ไม่ได้ส่งรูปมา
    if annual_yoy_fig is None:
        try:
            ann = monthly.resample("A-DEC").last().pct_change().dropna() * 100.0
            ann = ann.rename(columns={"GLI_INDEX": "GLI_%YoY"})
            fig_ann = go.Figure()
            if "GLI_%YoY" in ann.columns:
                fig_ann.add_trace(go.Scatter(x=ann.index, y=ann["GLI_%YoY"],
                                             mode="lines+markers", name="GLI_%YoY"))
            for c in [c for c in ann.columns if c != "GLI_%YoY"]:
                fig_ann.add_trace(go.Bar(x=ann.index, y=ann[c], name=f"{c}_%YoY"))
            fig_ann.update_layout(
                title="Annual YoY: GLI (line) vs Assets (bars)",
                barmode="group", hovermode="x unified",
                legend=dict(orientation="h", y=1.05),
                xaxis=dict(rangeslider=dict(visible=True))
            )
            annual_yoy_fig = fig_ann
        except Exception:
            annual_yoy_fig = go.Figure()
    st.plotly_chart(annual_yoy_fig, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 2: Rolling ----------
with tab_roll:
    st.subheader(f"Rolling {int(win_m)}-Month Statistics vs GLI (Monthly Returns)")
    c1, c2 = st.columns(2)

    # Rolling Corr
    with c1:
        fig_rc = go.Figure()
        if isinstance(roll_corr_m_df, pd.DataFrame) and not roll_corr_m_df.empty:
            for col in [c for c in roll_corr_m_df.columns if c != "GLI_INDEX"]:
                fig_rc.add_trace(go.Scatter(x=roll_corr_m_df.index, y=roll_corr_m_df[col], mode="lines", name=str(col)))
        fig_rc.update_layout(title=f"Rolling {int(win_m)}M Correlation vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)),
                             yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})

    # Rolling Beta
    with c2:
        fig_rb = go.Figure()
        if isinstance(roll_beta_m_df, pd.DataFrame) and not roll_beta_m_df.empty:
            for col in [c for c in roll_beta_m_df.columns if c != "GLI_INDEX"]:
                fig_rb.add_trace(go.Scatter(x=roll_beta_m_df.index, y=roll_beta_m_df[col], mode="lines", name=str(col)))
        fig_rb.update_layout(title=f"Rolling {int(win_m)}M Beta vs GLI",
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02),
                             xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_rb, use_container_width=True, config={"displaylogo": False})

    # Rolling Alpha
    fig_ra = go.Figure()
    if isinstance(roll_alpha_m_df, pd.DataFrame) and not roll_alpha_m_df.empty:
        for col in [c for c in roll_alpha_m_df.columns if c != "GLI_INDEX"]:
            fig_ra.add_trace(go.Scatter(x=roll_alpha_m_df.index, y=roll_alpha_m_df[col], mode="lines", name=str(col)))
    fig_ra.update_layout(title=f"Rolling {int(win_m)}M Alpha vs GLI (approx, %/mo)",
                         hovermode="x unified",
                         legend=dict(orientation="h", y=1.02),
                         xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_ra, use_container_width=True, config={"displaylogo": False})

# ---------- Tab 3: Regime & Events ----------
with tab_regime:
    st.subheader("GLI Regime (YoY>0 = Expansion) & Event Study")
    # Rebased + shaded expansion
    fig_reg = go.Figure()
    if isinstance(rebased_m, pd.DataFrame) and not rebased_m.empty:
        for col in rebased_m.columns:
            fig_reg.add_trace(go.Scatter(x=rebased_m.index, y=rebased_m[col], mode="lines", name=str(col)))
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
    except Exception:
        fig_gold_yoy = go.Figure()
    st.plotly_chart(fig_gold_yoy, use_container_width=True, config={"displaylogo": False})

    st.markdown("##### Event Study — ผลตอบแทนสะสมโดยเฉลี่ยหลังจุดเปลี่ยนระบอบ")
    st.caption("**Upturn** = GLI จากหดตัว → ขยายตัว, **Downturn** = GLI จากขยายตัว → หดตัว; วัดผลสะสมถัดไป 3/6/12 เดือน")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**หลัง Upturn**")
        st.dataframe(_to_display_df(evt_up.round(2) if isinstance(evt_up, pd.DataFrame) else evt_up), use_container_width=True)
    with c2:
        st.markdown("**หลัง Downturn**")
        st.dataframe(_to_display_df(evt_down.round(2) if isinstance(evt_down, pd.DataFrame) else evt_down), use_container_width=True)

    # Auto summary (Thai) - แก้ไขให้ปลอดภัยขึ้น
    st.markdown("#### 📌 Auto Summary")
    try:
        # สร้าง performance regime table แบบปลอดภัย
        perf_regime = None
        try:
            perf_regime = gl.perf_regime_table(monthly_rets, regime_df)
        except:
            pass
            
        # เรียก auto_summary แบบปลอดภัย
        summary_text = gl.auto_summary(metrics_table, betas_df, evt_up, evt_down, perf_regime)
        
        # ถ้าได้ summary กลับมา
        if summary_text and isinstance(summary_text, str) and len(summary_text.strip()) > 0:
            st.info(summary_text)
        else:
            # สร้าง summary เอง
            custom_summary = []
            
            # GLI CAGR info
            if pd.notna(gli_full):
                custom_summary.append(f"• GLI CAGR (เต็มช่วง): {_fmt_pct(gli_full)}")
            if pd.notna(gli_n):
                custom_summary.append(f"• GLI CAGR ({int(years_n)} ปี): {_fmt_pct(gli_n)}")
                
            # Liquidity premium
            if pd.notna(nas_liq):
                custom_summary.append(f"• NASDAQ เหนือ GLI: {_fmt_pct(nas_liq)}")
            if pd.notna(gold_liq):
                custom_summary.append(f"• GOLD เหนือ GLI: {_fmt_pct(gold_liq)}")
                
            # Beta info from betas_df
            if isinstance(betas_df, pd.DataFrame) and not betas_df.empty:
                try:
                    # หา asset ที่มี beta สูงสุด/ต่ำสุด
                    beta_col = None
                    for col in ['beta', 'Beta', 'BETA']:
                        if col in betas_df.columns:
                            beta_col = col
                            break
                    
                    if beta_col:
                        betas_clean = betas_df[beta_col].dropna()
                        if not betas_clean.empty:
                            max_beta_idx = betas_clean.idxmax()
                            min_beta_idx = betas_clean.idxmin()
                            custom_summary.append(f"• Beta สูงสุด: {max_beta_idx} ({betas_clean[max_beta_idx]:.2f})")
                            custom_summary.append(f"• Beta ต่ำสุด: {min_beta_idx} ({betas_clean[min_beta_idx]:.2f})")
                except:
                    pass
                    
            if custom_summary:
                st.info("สรุปหลักๆ:\n" + "\n".join(custom_summary))
            else:
                st.info("สรุปย่อ: ไม่สามารถสร้างสรุปได้จากข้อมูลปัจจุบัน")
                
    except Exception as e:
        st.warning(f"ไม่สามารถสร้าง Auto Summary ได้: {str(e)}")
        st.info("กรุณาตรวจสอบข้อมูลในแท็บ Tables เพื่อดูรายละเอียด")

# ---------- Tab 4: Tables ----------
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
