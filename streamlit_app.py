# streamlit_app.py — IVP / GLI Suite  Landing Page + Full Guide
import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IVP / GLI Suite",
    page_icon="🌊",
    layout="wide",
)

try:
    import gli_lib as gl
    _GL_OK = True
except Exception:
    _GL_OK = False

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='margin-bottom:0'>🌊 IVP / GLI Suite</h1>
<p style='color:#6b7280;font-size:1.05rem;margin-top:4px'>
Global Liquidity Intelligence + Portfolio Downside Protection
</p>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/02_GLI_Dashboard.py", label="🌊  GLI Dashboard — สภาพคล่องโลก vs สินทรัพย์")
with c2:
    st.page_link("pages/IVP-Analyzer.py",     label="📊  IVP Analyzer — ป้องกัน Downside ของพอร์ต")

_raw = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", ""))
if _GL_OK:
    _key = gl.sanitize_fred_key(_raw)
    if _key and gl.validate_fred_key_format(_key):
        st.success(f"FRED_API_KEY ✅ ({gl.mask_key(_key)}) — พร้อมดึงข้อมูลทั้งหมด")
    elif _key:
        st.error(f"FRED_API_KEY รูปแบบผิด ({gl.mask_key(_key)}) — ต้องเป็น alphanumeric 32 ตัว")
    else:
        st.warning("ยังไม่พบ FRED_API_KEY — App → Settings → Secrets → `FRED_API_KEY = \"...\"`")
else:
    if _raw.strip():
        st.success("FRED_API_KEY ✅ detected")
    else:
        st.warning("ยังไม่พบ FRED_API_KEY")

st.divider()

# ═══════════════════════════════════════════════════════════════
# GUIDE TABS
# ═══════════════════════════════════════════════════════════════
st.subheader("📚 คู่มือการใช้งาน")

t1, t2, t3, t4 = st.tabs([
    "🗺️  GLI คืออะไร",
    "📊  ช่วงค่าอ้างอิง",
    "🔍  วิธีอ่านสัญญาณ",
    "🎯  สถานการณ์ตลาด",
])

# ───────────────────────────────────────────────────────────────
# TAB 1 : GLI คืออะไร
# ───────────────────────────────────────────────────────────────
with t1:
    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown("### GLI = Global Liquidity Index")
        st.markdown("""
**GLI วัดว่าเงินทั้งโลกมากหรือน้อยแค่ไหน** โดยรวมเงินที่ธนาคารกลางหลักของโลก
"สร้าง" เข้าสู่ระบบ แล้วหักด้วยเงินที่รัฐบาลสหรัฐ "ดูด" กลับไป

> **เมื่อ GLI ขึ้น → เงินล้นระบบ → สินทรัพย์เสี่ยงมักขึ้น**
> **เมื่อ GLI ลง → เงินตึงระบบ → สินทรัพย์เสี่ยงมักลง**
        """)
        st.markdown("#### สูตร GLI (USD)")
        st.code(
            "GLI  =  Fed_WALCL           ← พิมพ์เงิน (QE)\n"
            "      + ECB_Assets × USDEUR  ← ยุโรปพิมพ์เงิน\n"
            "      + BOJ_Assets ÷ USDJPY  ← ญี่ปุ่นพิมพ์เงิน\n"
            "      + PBOC_Assets*          ← จีนพิมพ์เงิน (optional)\n"
            "      ─ TGA                   ← รัฐบาลสหรัฐดูดเงิน\n"
            "      ─ ONRRP                 ← Fed ดูดเงินคืนชั่วคราว\n\n"
            "* PBOC ต้องใส่ series ID เอง (ไม่มีบน FRED โดยตรง)",
            language="text",
        )
    with c_right:
        st.markdown("#### ทิศทางของแต่ละ Component")
        df_dir = pd.DataFrame([
            ("🏦 FED WALCL",   "Fed พิมพ์เงิน (QE/TLTRO)",   "⬆ inject", "⬇ drain (QT)"),
            ("🇪🇺 ECB Assets", "ECB พิมพ์เงิน (APP/PEPP)",    "⬆ inject", "⬇ drain"),
            ("🇯🇵 BOJ Assets", "BOJ พิมพ์เงิน (YCC/QQE)",     "⬆ inject", "⬇ drain"),
            ("🏛️ TGA",         "คลังสหรัฐดูดเงิน",             "⬇ inject", "⬆ drain"),
            ("🔄 ONRRP",       "MMF จอดเงินกับ Fed ชั่วคราว",  "⬇ inject", "⬆ drain"),
            ("💱 USD/EUR",      "แปลง ECB → USD",               "ใช้ convert", ""),
            ("💱 USD/JPY",      "แปลง BOJ → USD",               "ใช้ convert", ""),
        ], columns=["Component", "หน้าที่", "📗 ทิศ Inject", "📕 ทิศ Drain"])
        st.dataframe(
            df_dir.style.map(
                lambda v: "color:#2ca02c;font-weight:bold" if "inject" in str(v) else
                          ("color:#d62728;font-weight:bold" if "drain" in str(v) else ""),
                subset=["📗 ทิศ Inject", "📕 ทิศ Drain"]),
            hide_index=True, use_container_width=True, height=265,
        )

    st.divider()
    st.markdown("### 🔧 Fed Plumbing — เครื่องมือ inject/drain แบบ 'อ้อมๆ'")
    st.markdown("""
นอกจาก WALCL/TGA/RRP ในสูตร GLI — Fed ยังมีเครื่องมือที่ **ไม่ปรากฏใน headline**
แต่ส่งผลต่อสภาพคล่องจริงในตลาด ติดตามได้ใน 🔧 Fed Plumbing Panel
    """)
    df_plumb = pd.DataFrame([
        ("🏦 Reserve Balances (WRESBAL)",
         "เงินสำรองที่ธนาคารฝากไว้กับ Fed",
         "สูง = ธนาคารมีสภาพคล่อง กล้าปล่อยสินเชื่อ",
         "ต่ำ (<$2T) = ระวัง — อาจเกิด repo stress"),
        ("🚨 Emergency Loans / BTFP (WLCFLPCL)",
         "Fed ปล่อยกู้ฉุกเฉินให้ธนาคาร\n(รวม Bank Term Funding Program)",
         "พุ่งขึ้น = Fed inject แบบเงียบๆ\n(ผลเหมือน QE แต่ไม่ประกาศ)",
         "≈ $0 = ระบบปกติ ไม่มีวิกฤต"),
        ("💰 Money Market Fund (WRMFSL)",
         "AUM ของกองทุน Money Market\n($6T+ ที่จอดรอโอกาส)",
         "ลดลง = เงินไหลออก MMF\n→ สู่ risk assets",
         "สูงขึ้น = เงินหนีเสี่ยง\nจอดใน MMF ไม่เข้าตลาด"),
        ("📉 HY Credit Spread (BAMLH0A0HYM2)",
         "Spread ของ High Yield Bond vs Treasury\n(วัดว่าตลาดกล้าเสี่ยงไหม)",
         "< 350 bps = risk appetite สูง\nliquidity ส่งถึงตลาดได้ดี",
         "> 500 bps = stress\nเงินมีแต่ไม่ไหลสู่ตลาด"),
        ("📊 Yield Curve 10Y-2Y (T10Y2Y)",
         "ส่วนต่างดอกเบี้ยระยะยาว-สั้น\n(barometer ของ growth expectations)",
         "> 0% (Steep) = ตลาดคาด growth\nระบบเงินสมดุล",
         "< 0% (Inverted) = recession signal\nนำก่อน recession 6-18 เดือน"),
        ("💵 DXY Broad USD (DTWEXBGS)",
         "ค่าเงินดอลลาร์เทียบ basket สกุลเงินโลก\n(inverse ของ global liquidity)",
         "อ่อนลง = dollar ไหลออกสู่โลก\nEM/BTC/Commodities ได้ประโยชน์",
         "แข็งขึ้น = ดูด dollar global กลับ US\nกดดัน EM/BTC/Gold"),
        ("🇨🇳 China FX Reserves (RRFXRBCNM)",
         "ทุนสำรองเงินตราต่างประเทศของจีน\n(PBOC proxy รายเดือน)",
         "เพิ่มขึ้น = PBOC ไม่ต้องป้องหยวน\nglobal liquidity ไม่ถูกดูดออก",
         "ลดลง = PBOC ขาย USD ป้องหยวน\n= ดูด global dollar supply"),
        ("🔶 Copper (Yahoo: HG=F)",
         "ราคาทองแดง — Dr. Copper\nleading indicator ความต้องการจริง",
         "ขึ้น = demand จริง\nมักนำ GLI expansion ประมาณ 1 ไตรมาส",
         "ลง = ความต้องการชะลอ\nอาจนำหน้า GLI หด"),
    ], columns=["Instrument", "คือ...", "📗 Signal บวก", "📕 Signal ลบ"])
    st.dataframe(
        df_plumb.style.set_properties(**{"white-space": "pre-wrap", "font-size": "12px"}),
        hide_index=True, use_container_width=True, height=310,
    )

# ───────────────────────────────────────────────────────────────
# TAB 2 : ช่วงค่าอ้างอิง
# ───────────────────────────────────────────────────────────────
with t2:
    st.markdown("### 📊 ช่วงค่าอ้างอิงสำหรับแต่ละ FRED Series")
    st.caption("ช่วงเหล่านี้อิงข้อมูลประวัติศาสตร์ — ใช้เป็น 'ไกด์ไลน์' ประกอบการตัดสินใจ ไม่ใช่กฎตายตัว")

    st.markdown("#### 🏛️ GLI Core Series")
    df_core = pd.DataFrame([
        ("FED WALCL",  "M USD",
         "< $4T\n(pre-2020)",  "$4–7T\npost-COVID base", "> $7T\naggressive QE",
         "$8.9T\nApr 2022", "$3.8T\nJan 2020",
         "เปลี่ยนแปลง:\n+$120B/M = aggressive QE\n−$95B/M = max QT\n< −$40B/M = gradual QT"),
        ("ECB Assets", "M EUR",
         "< €4T", "€4–6T", "> €6T",
         "€8.8T\nJan 2023", "€2.0T\n2012",
         "PEPP/APP ทำให้พุ่งสูง\nปัจจุบัน unwind ช้าๆ\nEUR/USD ส่งผลต่อ GLI ด้วย"),
        ("BOJ Assets", "B JPY",
         "< ¥400T", "¥400–600T", "> ¥600T",
         "¥754T\n2024 (ยังขึ้น)", "¥150T\n2008",
         "BOJ พิเศษ — ยังขยาย\nYield Curve Control (YCC)\nJPY อ่อนมาก = BOJ inject มาก"),
        ("TGA",        "B USD",
         "< $150B\ndebt ceiling drain", "$200–600B\nปกติ", "> $800B\nตึงตลาด",
         "$1.8T\nJul 2020", "$~5B\ndebt ceiling 2023",
         "Debt ceiling: TGA แห้ง\nเมื่อแก้ได้ TGA พุ่ง = drain\nผลตรงข้าม RRP"),
        ("O/N RRP",    "B USD",
         "< $100B\nliquidity ไหลดี", "$100–500B", "> $1T\nexcess liquidity คั่ง",
         "$2.55T\nDec 2022", "$0\npre-2021",
         "RRP ลด = เงินออก MMF สู่ตลาด\n= bullish signal\nRRP ต่ำ + TGA ขึ้น = net drain"),
    ], columns=["Series", "หน่วย",
                "🔵 ต่ำ / ตึง", "🟡 ปกติ", "🔴 สูง / ผ่อน",
                "All-time High", "All-time Low", "💡 อ่านอย่างไร"])
    st.dataframe(
        df_core.style.set_properties(**{"white-space": "pre-wrap", "font-size": "12px"}),
        hide_index=True, use_container_width=True, height=230,
    )

    st.divider()
    st.markdown("#### 🔧 Fed Plumbing Series")
    df_pb = pd.DataFrame([
        ("Reserve Balances\nWRESBAL", "B USD",
         "< $2T 🔴\nrepo stress risk", "$2.5–3.5T 🟡", "> $3.5T 🟢\nample liquidity",
         "$3.9T\n2021", "$1.4T\nSep 2019",
         "< $2T เคยทำ repo crisis 2019\nFormula: WALCL − TGA − RRP ≈ Reserves"),
        ("BTFP/Emergency\nWLCFLPCL", "M USD",
         "≈ $0 🟢\nระบบปกติ", "$10–50B 🟡", "> $100B 🔴\nวิกฤตซ่อน",
         "$165B\nMay 2023 (SVB)", "≈ $0\nก่อน 2023",
         "พุ่งหลัง SVB = stealth QE\nแต่สัญญาณว่ามีปัญหาจริง"),
        ("MMF Assets\nWRMFSL", "B USD",
         "< $4T 🟢\nเงินกำลังลงทุน", "$4–5.5T 🟡", "> $5.5T 🔴\nเงินหนีเสี่ยง",
         "$6.4T\n2024", "$2.7T\n2020",
         "ลดลงหลัง rate cut = bullish\nเพิ่มขึ้น = เงินไม่เข้าตลาด"),
        ("HY OAS Spread\nBAMLH0A0HYM2", "bps",
         "< 300 bps 🟢\nrisk appetite สูง", "300–500 bps 🟡", "> 500 bps 🔴\ncredit stress",
         "~2000 bps\n2008 GFC", "~250 bps\n2021",
         "Widening = liquidity ติดค้าง\nGLI สูงแต่ spread กว้าง = signal ไม่ส่งผ่าน"),
        ("Yield Curve 10Y-2Y\nT10Y2Y", "%",
         "< 0% 🔴\nINVERSION\nrecession warning", "0–1% 🟡\nflat/re-steepen", "> 1.5% 🟢\nhealthy growth",
         "+3.0%\n2011", "−1.08%\nJul 2023",
         "Un-invert จาก negative\n= near-term risk ยังสูง\nLead recession 6-18 เดือน"),
        ("DXY Broad\nDTWEXBGS", "Index\n(2006=100)",
         "< 95 🟢\ndollar อ่อน = global ease", "95–115 🟡", "> 115 🔴\ndollar แข็ง = global squeeze",
         "~128\nOct 2022", "~84\n2018",
         "DXY > 115 กดดัน EM/BTC/Gold\nInverse ของ GLI global"),
        ("China FX Reserves\nRRFXRBCNM", "B USD",
         "< $3.0T 🔴\nPBOC ต้องขาย USD", "$3.0–3.2T 🟡", "> $3.2T 🟢\nstable/accumulate",
         "$4.0T\nJun 2014", "$3.1T\nJan 2017",
         "ลดเร็ว = global dollar squeeze\nRรายเดือน (lag ~4 สัปดาห์)"),
        ("Copper HG=F", "USD/lb",
         "< $3.0 🔴\ndemand อ่อนมาก", "$3.0–4.5 🟡\nnormal range", "> $4.5 🟢\nstrong demand",
         "$5.20\nMay 2024 ATH", "$1.94\nMar 2020",
         "Lead GLI expansion 1-2 ไตรมาส\nDr. Copper = real economy proxy"),
    ], columns=["Series", "หน่วย",
                "📕 ต่ำ / เสี่ยง", "🟡 กลาง / ปกติ", "📗 สูง / ดี",
                "Historical High", "Historical Low", "💡 Key Note"])
    st.dataframe(
        df_pb.style.set_properties(**{"white-space": "pre-wrap", "font-size": "12px"}),
        hide_index=True, use_container_width=True, height=330,
    )

    st.divider()
    st.markdown("#### 📈 Asset Reference")
    df_ast = pd.DataFrame([
        ("NASDAQ (^IXIC)",  "Index",   "Corr vs GLI: 0.7–0.9",  "รับ liquidity ได้เร็วสุด",      "Beta สูง = ขึ้นเร็ว ลงเร็ว"),
        ("S&P 500 (^GSPC)", "Index",   "Corr vs GLI: 0.6–0.8",  "Diversified กว่า NASDAQ",        "Beta ปานกลาง = stable กว่า"),
        ("GOLD (GC=F)",     "USD/oz",  "Corr vs GLI: 0.4–0.6",  "ประโยชน์ทั้ง GLI up+down",      "Hedge เงินเฟ้อ + geopolitics"),
        ("BTC (BTC-USD)",   "USD",     "Corr vs GLI: 0.5–0.8",  "Highly volatile — Beta ~2–5×",   "Lead GLI signal 2-4 สัปดาห์"),
        ("ETH (ETH-USD)",   "USD",     "Corr vs GLI: 0.5–0.8",  "Corr กับ BTC สูง ~0.85+",       "Beta vs GLI สูงกว่า BTC"),
        ("Copper (HG=F)",   "USD/lb",  "Corr vs GLI: 0.5–0.7",  "Leading indicator demand",        "นำ GLI 1-2 ไตรมาส"),
        ("DXY Broad",       "Index",   "Corr vs GLI: −0.5~−0.8","**Inverse** — แข็งขึ้น = ตึง",   "ใช้ยืนยัน / ต้าน GLI signal"),
    ], columns=["Asset", "หน่วย", "ความสัมพันธ์กับ GLI", "ลักษณะพิเศษ", "Note"])
    st.dataframe(df_ast, hide_index=True, use_container_width=True, height=280)

# ───────────────────────────────────────────────────────────────
# TAB 3 : วิธีอ่านสัญญาณ
# ───────────────────────────────────────────────────────────────
with t3:
    st.markdown("### 🔍 วิธีอ่านสัญญาณจาก GLI Dashboard")

    st.markdown("#### 1. GLI YoY Quantile — อยู่ในระดับไหน?")
    df_q = pd.DataFrame([
        ("Q5 (Top 20%)", "> +8% YoY",  "BTC, NASDAQ, ETH",   "Risk-On เต็มที่",  "GLI กำลังขยายตัวแรง — historical bull window สำหรับ crypto/tech"),
        ("Q4",           "+3%–+8%",     "NASDAQ, SP500, GOLD","Risk-On moderate", "สภาพคล่องดีแต่ไม่สุด — portfolio ปกติ"),
        ("Q3",           "−1%–+3%",     "SP500, GOLD",        "Neutral",           "GLI ไม่ชัดเจน — ดู HY Spread + DXY ประกอบ"),
        ("Q2",           "−5%–−1%",     "GOLD, Cash",         "Defensive",         "GLI กำลังหด — ระวัง drawdown"),
        ("Q1 (Bot 20%)", "< −5% YoY",  "Cash, Gold, Short",  "Risk-Off",          "Historical worst window — BTC/crypto ระวังมาก"),
    ], columns=["GLI Quantile", "ช่วง YoY %", "Asset ที่มักชนะ", "แนวทาง", "คำอธิบาย"])

    def _q_bg(v):
        if "Q5" in str(v) or "เต็มที่" in str(v): return "background-color:rgba(44,160,44,0.18)"
        if "Q4" in str(v) or "moderate" in str(v): return "background-color:rgba(44,160,44,0.08)"
        if "Q3" in str(v) or "Neutral" in str(v):  return "background-color:rgba(255,200,0,0.12)"
        if "Q2" in str(v) or "Defensive" in str(v):return "background-color:rgba(214,39,40,0.08)"
        if "Q1" in str(v) or "Risk-Off" in str(v): return "background-color:rgba(214,39,40,0.18)"
        return ""

    st.dataframe(df_q.style.map(_q_bg), hide_index=True, use_container_width=True, height=220)

    st.divider()
    c_ll, c_gr = st.columns(2)

    with c_ll:
        st.markdown("#### 2. Lead/Lag Analysis")
        st.markdown("""
**วิธีอ่าน CCF Chart:**
- แกน X = lag (เดือน)
- r บวกสูงที่ lag > 0 = **GLI นำ** (GLI ขึ้นก่อน → asset ตามมา)
- r บวกสูงที่ lag < 0 = Asset นำ GLI
- เส้นประแดง = 95% CI — ต่ำกว่าเส้นนี้ = ไม่มีนัยสำคัญ

**ค่าประมาณทั่วไป:**
| Asset  | Lag โดยประมาณ |
|--------|--------------|
| BTC    | GLI นำ 2–4 เดือน |
| NASDAQ | GLI นำ 1–3 เดือน |
| SP500  | GLI นำ 1–2 เดือน |
| GOLD   | GLI นำ 0–2 เดือน |
| Copper | Copper นำ GLI 1–2 ไตรมาส |

**ข้อควรระวัง:**
- Lag เปลี่ยนตาม regime (2020 vs 2022 ต่างกัน)
- ดู Rolling Corr ประกอบ — ถ้า corr ลด แสดงว่า lag กำลังเปลี่ยน
        """)

    with c_gr:
        st.markdown("#### 3. Statistical Tests")
        st.markdown("""
**ADF Test (Stationarity)**
| ผล | ความหมาย |
|---|---|
| ✅ p < 0.05 | Series stationary ใช้ regression ได้โดยตรง |
| ❌ Level | Unit root — ควรใช้ %return |

→ ปกติจะเป็น I(1) ที่ level และ I(0) ที่ return — **เป็นเรื่องปกติ**

**Engle-Granger Cointegration**
| ผล | ความหมาย |
|---|---|
| ✅ p < 0.05 | มี long-run tie → ECM ใช้ได้ |
| — | ใช้ returns เท่านั้น |

**Granger Causality: GLI → Asset**
| ผล | ความหมาย |
|---|---|
| ✅ p < 0.05 | GLI อดีตช่วยพยากรณ์ Asset |
| — | ไม่มีหลักฐาน predictive |

→ ✅ ไม่ใช่ "cause" เชิง mechanistic — แค่บอกว่า GLI มีข้อมูลเพิ่มที่มีประโยชน์

**Forward Return Heatmap:**
- สีเขียว = avg return บวกในอดีต เมื่อ GLI อยู่ระดับนั้น
- ดู Hit Rate คู่กัน (% เดือนที่ return > 0)
- Hit Rate > 60% + avg return บวก = สัญญาณน่าเชื่อถือกว่า
        """)

    st.divider()
    st.markdown("#### 4. Fed Plumbing — อ่านยืนยันสัญญาณ GLI")
    st.markdown("""
| GLI Signal | ยืนยันด้วย | ความหมายรวม |
|---|---|---|
| GLI ขึ้น | Reserves ขึ้น + HY Spread แคบลง | **Full Expansion** — เงินถึงตลาดจริง |
| GLI ขึ้น | MMF สูง + HY Spread กว้าง | **Trapped Liquidity** — เงินมีแต่ยังไม่ลงทุน |
| GLI flat | BTFP พุ่ง + Reserves ขึ้น | **Stealth QE** — Fed inject แบบเงียบๆ |
| GLI ลง | DXY สูง + China FX ลด | **Global Dollar Squeeze** — ระวัง EM/BTC |
| GLI ดูดี | Yield Curve ยัง Inverted | **รอ un-invert** — recession risk ยังสูง |
    """)

# ───────────────────────────────────────────────────────────────
# TAB 4 : สถานการณ์ตลาด
# ───────────────────────────────────────────────────────────────
with t4:
    st.markdown("### 🎯 5 สถานการณ์หลักของ GLI")
    st.caption("วิธีอ่านหลายมุมมองพร้อมกัน: GLI + Reserves + HY Spread + DXY + Yield Curve")

    scenarios = [
        ("🟢 สถ. 1: Full Expansion — Bull Signal", "#f0fff4", "#2ca02c", [
            ("GLI YoY",          "บวก +5% ขึ้นไป และกำลังเร่งตัว"),
            ("Reserve Balances", "> $3T และสูงขึ้น"),
            ("O/N RRP",          "ลดลง (เงินไหลออก MMF)"),
            ("HY Spread",        "< 350 bps และแคบลง"),
            ("Yield Curve",      "steepening จาก flat/inversion"),
            ("DXY",              "อ่อนลงหรือคงที่"),
        ], "📈 Risk-On เต็มที่\n→ overweight BTC, NASDAQ, ETH\n→ เพิ่ม position ตาม Q5 forward return\n→ ระวัง FOMO: ดู rolling corr ว่ายังสูง",
           "2020 Q4 – 2021: Fed QE + TGA ลด + RRP ≈ 0\n→ BTC: $10K → $69K (+590%)"),

        ("🟡 สถ. 2: Trapped Liquidity — Mixed Signal", "#fffef0", "#d4a017", [
            ("GLI YoY",    "บวก แต่เร่งตัวช้า"),
            ("MMF Assets", "สูงและยังไม่ลด"),
            ("HY Spread",  "400–550 bps หรือกว้างขึ้น"),
            ("Yield Curve","ยัง flat หรือ inverted"),
            ("DXY",        "แข็งหรือผันผวน"),
        ], "⏸️ Neutral — ไม่รีบตัดสินใจ\n→ รอสัญญาณ: MMF เริ่มลด หรือ HY spread แคบ\n→ ถือ GOLD เป็น hedge\n→ position sizing เล็กลง",
           "ต้นปี 2023: GLI ขึ้น แต่ MMF สูง + HY spread กว้าง\n→ ตลาดผันผวนหนัก"),

        ("🔶 สถ. 3: Stealth Tightening — Hidden Drain", "#fff8f0", "#ff7f0e", [
            ("GLI YoY",    "flat หรือลบ แม้ดูเหมือน neutral"),
            ("TGA",        "กำลังสูงขึ้น (Treasury สะสม cash)"),
            ("RRP",        "ต่ำแล้ว ไม่มีที่ absorb เพิ่ม"),
            ("Reserves",   "ลดลงแม้ WALCL คงที่"),
            ("DXY",        "แข็งขึ้นต่อเนื่อง"),
        ], "📉 Defensive\n→ ลด risk assets\n→ ระวัง: headline 'Fed ไม่ขึ้นดอกเบี้ย' แต่ตลาดตึงจาก TGA\n→ เพิ่ม cash, GOLD ลด BTC/Tech",
           "Q4 2022 – Q1 2023: QT + TGA rebuild\n→ NASDAQ ลง 33% ในปี 2022"),

        ("🔵 สถ. 4: Stealth QE — Hidden Inject", "#f0f8ff", "#1f77b4", [
            ("WALCL",          "ลดลง (QT ดำเนินการ) แต่..."),
            ("BTFP/Emergency", "พุ่งสูง (Fed ปล่อยกู้ฉุกเฉิน)"),
            ("Net Fed Liq",    "Reserves + BTFP กำลังขึ้น"),
            ("HY Spread",      "แคบลงหลังจากกว้าง (วิกฤตผ่าน)"),
            ("GLI",            "อาจ flat แต่ net liquidity จริงขึ้น"),
        ], "🚨 ระวัง + โอกาส\n→ มีวิกฤต แต่ Fed กำลัง backstop\n→ BTC/risk มักขึ้นแรงหลัง stealth inject รับรู้\n→ จับตา BTFP amount เป็นพิเศษ",
           "มี.ค. 2023 (SVB): BTFP พุ่ง $165B ใน 2 สัปดาห์\n→ BTC ขึ้น 40%+ ใน 30 วัน แม้ข่าวลบ"),

        ("🔴 สถ. 5: Global Dollar Squeeze — EM/Risk Under Pressure", "#fff5f5", "#d62728", [
            ("DXY Broad",       "> 115 และยังขึ้น"),
            ("China FX",        "ลดต่อเนื่อง 3+ เดือน"),
            ("HY Spread",       "> 500 bps"),
            ("GLI YoY",         "ลบและลึก"),
            ("Copper",          "ลดลงหรือ flat"),
        ], "🛑 Risk-Off\n→ เพิ่ม Cash, USD, GOLD\n→ หลีกเลี่ยง BTC, EM, Commodities\n→ รอจนกว่า DXY กลับทิศ หรือ Fed pivot ชัด\n→ 'Don't catch a falling knife'",
           "2022: DXY ทำ ATH 128, BTC ลง 65%, NASDAQ ลง 33%\nEM currencies collapse"),
    ]

    for title, bg, border, sigs, action, example in scenarios:
        with st.expander(title, expanded=False):
            c_s, c_a = st.columns([2, 1])
            with c_s:
                st.markdown(f"**Checklist สัญญาณ — {title.split(':')[1].strip().split('—')[0].strip()}:**")
                for ind, val in sigs:
                    st.markdown(f"- **{ind}**: {val}")
            with c_a:
                st.info(f"**แนวทางรับมือ:**\n\n{action}")
                st.caption(f"📖 ตัวอย่างอดีต:\n{example}")

    st.divider()
    st.markdown("### 📋 Quick Score Matrix — ประเมิน Bias รวม")
    st.caption("Score แต่ละ signal แล้วรวมกัน → ≥ +4 = bullish / ≤ −4 = bearish")

    df_score = pd.DataFrame([
        ("GLI YoY",        "> +5%",     "+2", "−1% ถึง +5%",  "0",  "< −1%",      "−2"),
        ("Reserve Balances","> $3.5T",  "+1", "$2.5–3.5T",    "0",  "< $2.5T",    "−1"),
        ("O/N RRP",        "ลดลง",      "+1", "Flat",          "0",  "สูงขึ้น",    "−1"),
        ("BTFP/Emergency", "≈ $0",      "+1", "$10–50B",       "0",  "> $100B",    "−1"),
        ("MMF Assets",     "ลดลง",      "+1", "Flat",          "0",  "สูงขึ้น",    "−1"),
        ("HY Spread",      "< 300 bps", "+2", "300–500 bps",   "0",  "> 500 bps",  "−2"),
        ("Yield Curve",    "> +1%",     "+1", "0–+1%",         "0",  "< 0%",       "−1"),
        ("DXY Broad",      "อ่อนลง",   "+1", "Flat",          "0",  "แข็งขึ้น",   "−1"),
        ("China FX",       "เพิ่ม",     "+1", "Flat",          "0",  "ลดลง",       "−1"),
        ("Copper",         "ขึ้น",      "+1", "Flat",          "0",  "ลง",         "−1"),
    ], columns=["Signal", "Bull Cond.", "🟢 Score", "Neutral", "🟡 Score", "Bear Cond.", "🔴 Score"])

    def _sc(v):
        s = str(v)
        if s.startswith("+") and s != "+0": return "color:#2ca02c;font-weight:bold"
        if s.startswith("−"):               return "color:#d62728;font-weight:bold"
        return ""

    st.dataframe(
        df_score.style.map(_sc, subset=["🟢 Score","🟡 Score","🔴 Score"]),
        hide_index=True, use_container_width=True, height=380,
    )
    st.markdown("""
**ผลรวม Score:**  
🟢 ≥ +6 = Strong Bull · ✅ +3 ถึง +5 = Moderate Bull · 🟡 −2 ถึง +2 = Neutral
⚠️ −3 ถึง −5 = Moderate Bear · 🔴 ≤ −6 = Strong Bear
    """)

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.divider()
with st.expander("⚙️ วิธีตั้งค่าและการใช้งาน"):
    st.markdown("""
**ขั้นตอนเริ่มต้น:**
1. ขอ FRED API Key ฟรีที่ https://fred.stlouisfed.org/docs/api/api_key.html
2. ใส่ key ใน Streamlit Secrets: `FRED_API_KEY = "your_32_char_key"`
3. เปิดหน้า **🌊 GLI Dashboard** → ระบบจะดึงข้อมูลอัตโนมัติ

**Sidebar Options ของ GLI Dashboard:**
| Option | คำอธิบาย |
|---|---|
| Start Year | เลือกช่วงเริ่มต้น (2008 = เห็น GFC; 2020 = เห็น COVID cycle) |
| CAGR Window | คำนวณ CAGR ย้อนหลัง (แนะนำ 10Y สำหรับ long-term) |
| GLI Quantiles | 5 = ละเอียด; 3 = แบ่งง่าย Bull/Neutral/Bear |
| GLI Normalisation | เปิดดู Z-Score + Acceleration (ใช้ M2 เป็น denominator) |
| Extra Assets | เพิ่ม Copper + DXY เข้า universe |
| 🔧 Fed Plumbing Panel | Reserves + BTFP + MMF + HY Spread + Yield Curve + DXY + Copper |

**Tips:**
- กด ♻️ Clear Cache ถ้าข้อมูลไม่อัปเดต
- Tab **🧭 Regime** มีข้อมูลเชิงลึกสุด — Forward Return Heatmap + Lead/Lag
- Tab **📋 Tables → Statistical Tests** บอกว่าความสัมพันธ์มีนัยสำคัญจริงหรือไม่
- **🔧 Fed Plumbing** เปิดผ่าน Sidebar ด้านซ้ายของ GLI Dashboard
    """)

st.caption("© 2025 IVP / GLI Suite · Data: FRED (St. Louis Fed) + Yahoo Finance · ไม่ใช่คำแนะนำการลงทุน")
