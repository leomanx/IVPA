# streamlit_app.py
import os
import streamlit as st

st.set_page_config(page_title="IVP / GLI Suite", layout="wide")
st.title("IVP / GLI Suite")

st.markdown("""
**IVP Analyzer** (Portfolio Downside Protection)
**GLI Dashboard** (Global Liquidity vs NASDAQ / GOLD / S&P500 / BTC / ETH)
""")

# ----- Quick links to pages (requires Streamlit >= 1.24) -----
st.subheader("Pages")
cols = st.columns(2)
with cols[0]:
    st.page_link("pages/IVP-Analyzer.py", label="📊 IVP Analyzer")
with cols[1]:
    st.page_link("pages/02_GLI_Dashboard.py", label="🌊 GLI Dashboard")

# ----- Environment / Secrets check -----
fred_key = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", "")).strip()
st.subheader("Environment")
if fred_key:
    st.success("FRED_API_KEY: detected ✅ (พร้อมใช้งาน GLI)")
else:
    st.warning(
        "ยังไม่พบ **FRED_API_KEY**\n\n"
        "- รันท้องถิ่น: สร้าง `.streamlit/secrets.toml` แล้วใส่ `FRED_API_KEY = \"...\"`\n"
        "- Streamlit Cloud: ไปที่ App → Settings → **Secrets** แล้วใส่ค่าเดียวกัน"
    )

# ----- Tips -----
with st.expander("How to"):
    st.markdown("""
1) ไปหน้า **GLI Dashboard** เพื่อดู GLI vs สินทรัพย์หลัก, Rolling Corr/Beta/Alpha, Regime & Event Study  
2) ไปหน้า **IVP Analyzer** อัปโหลดไฟล์พอร์ต (CSV/XLSX) แล้วกด **Run Analysis**  
3) ถ้าข้อมูลไม่อัปเดต ลองกดปุ่ม ♻️ Refresh cache ในแต่ละหน้า
""")

st.caption("© 2025")
