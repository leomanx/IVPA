import os
import sys
import json
import requests
import subprocess
import tempfile
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Ensure we can import gli_lib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gli_lib as gl

env_path = "/home/mmkpi4/netguard/.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# Hardcoded configuration (no need to put these in .env)
RCLONE_REMOTE = "tdv_screener_result:"
RCLONE_DEST_PATH = "MMK/Quant/Global-Liquidity-Index/Weekly"

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Skipping message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    r = requests.post(url, json=payload)
    if not r.ok:
        print(f"Failed to send message: {r.text}")

def send_telegram_media_group(image_paths):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or not image_paths:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup"
    
    media = []
    files = {}
    for i, path in enumerate(image_paths):
        name = f"photo{i}"
        media.append({"type": "photo", "media": f"attach://{name}"})
        files[name] = open(path, "rb")
        
    payload = {"chat_id": TELEGRAM_CHAT_ID, "media": json.dumps(media)}
    r = requests.post(url, data=payload, files=files)
    for f in files.values():
        f.close()
    if not r.ok:
        print(f"Failed to send media group: {r.text}")

def rclone_upload(local_dir):
    dest = f"{RCLONE_REMOTE}{RCLONE_DEST_PATH}"
    print(f"Uploading to rclone destination: {dest}")
    try:
        subprocess.run(["rclone", "copy", local_dir, dest], check=True)
        print("Rclone upload successful.")
    except Exception as e:
        print(f"Rclone upload failed: {e}")

def run_bot():
    if not FRED_API_KEY:
        print("Error: FRED_API_KEY is not set.")
        return
        
    start_date = "2022-01-01" # Keep it recent for mobile readability
    errors = []
    
    print("Loading Core Data...")
    try:
        core_data = gl.load_all(FRED_API_KEY, start=start_date, normalize=True)
    except Exception as e:
        err = f"Core Data Error: {e}"
        print(err)
        send_telegram_message(f"🚨 <b>GLI Bot Error</b>\n{err}")
        return
        
    print("Loading Plumbing Data...")
    try:
        plumb_data = gl.fed_plumbing(FRED_API_KEY, start=start_date)
    except Exception as e:
        plumb_data = None
        errors.append(f"Plumbing Error: {e}")
        
    print("Loading Yen Carry Data...")
    try:
        yen_data = gl.yen_carry_analysis(core_data["wk"], fred_api_key=FRED_API_KEY, start=start_date)
    except Exception as e:
        yen_data = None
        errors.append(f"Yen Carry Error: {e}")
        
    # --- Generate Narrative Text ---
    wk = core_data["wk"]
    gli_usd = wk["GLI_USD"].dropna()
    gli_now = gli_usd.iloc[-1]
    gli_wow = ((gli_now / gli_usd.iloc[-2]) - 1) * 100
    gli_yoy = ((gli_now / gli_usd.iloc[-53]) - 1) * 100 if len(gli_usd) >= 53 else 0
    is_exp = gli_yoy > 0
    
    text = f"📊 <b>GLI Weekly Report</b>\n"
    text += f"📅 <i>{datetime.now().strftime('%d %b %Y')}</i>\n\n"
    
    reg_emoji = "🟢" if is_exp else "🔴"
    text += "<b>1. Current GLI State</b>\n"
    text += "<pre>"
    text += f"GLI Index : {gli_now:.1f} B\n"
    text += f"WoW       : {gli_wow:+.2f}%\n"
    text += f"YoY       : {gli_yoy:+.1f}%\n"
    text += f"Regime    : {reg_emoji} {'EXPANSION' if is_exp else 'CONTRACTION'}"
    text += "</pre>\n\n"
    
    if plumb_data:
        text += "<b>2. Fed Plumbing & Stress</b>\n"
        text += "<pre>"
        ptbl = plumb_data["summary_tbl"]
        for _, r in ptbl.iterrows():
            if pd.isna(r['Latest']): continue
            mom = f"{r['MoM %']:+.1f}%" if pd.notna(r['MoM %']) else "-"
            # Simplify name for monospace alignment
            name = str(r['Instrument']).replace(" Spread", "").replace(" Curve", "").replace(" Balances", "")
            if len(name) > 10: name = name[:10]
            val = f"{r['Latest']}"
            sig = str(r['Signal Note']).replace("Inject", "💉Inject").replace("Drain", "🩸Drain").replace("Risk-Off", "🛡️Risk-Off").replace("Inversion", "⚠️Inv").replace("Normal", "✅Norm").replace("Stress", "🔥Stress")
            text += f"{name:<10}: {val:<7} ({mom})\n  └ {sig}\n"
        text += "</pre>\n\n"
        
    if yen_data:
        text += "<b>3. Yen Carry Trade Status</b>\n"
        cs = yen_data["current_state"]
        st = cs['Carry_State']
        st_icon = "🟢" if "Expanding" in st else "🔴" if "Unwinding" in st else "🟡"
        al_icon = "🚨" if "SEVERE" in str(cs['Unwind_Status']).upper() else ("⚠️" if "MINOR" in str(cs['Unwind_Status']).upper() else "✅")
        text += "<pre>"
        text += f"USDJPY : {float(cs['USDJPY']):.2f} ({cs['MoM_%']:+.1f}%)\n"
        text += f"State  : {st_icon} {st}\n"
        text += f"Alert  : {al_icon} {cs['Unwind_Status']}\n"
        if pd.notna(cs['VIX_latest']):
            text += f"VIX    : {float(cs['VIX_latest']):.2f}\n"
        text += "</pre>\n\n"
        
    if core_data and "annual" in core_data:
        text += "<b>4. Asset Performance (Year-to-Date / %YoY)</b>\n"
        text += "<pre>"
        annual_rets = core_data["annual"].pct_change().dropna() * 100
        if not annual_rets.empty:
            latest = annual_rets.iloc[-1]
            yr = annual_rets.index[-1]
            text += f"--- Year {yr} ---\n"
            for col, val in latest.items():
                if pd.notna(val):
                    name = str(col).replace("_INDEX", "")
                    text += f"{name:<8}: {val:+.1f}%\n"
        text += "</pre>\n\n"
        
    if errors:
        text += "⚠️ <b>Warnings:</b>\n"
        for e in errors:
            text += f"- {e}\n"
            
    # --- Helper to fix Kaleido Timestamp bug ---
    def clean_fig(f):
        import plotly.io as pio
        return pio.from_json(pio.to_json(f))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save text report to file for Rclone (with date/time to keep history)
        date_str = datetime.now().strftime('%Y-%m-%d_%H%M')
        report_path = os.path.join(tmpdir, f"report_{date_str}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<pre>", "").replace("</pre>", ""))
            
        img_paths = []
        
        print("Saving charts...")
        # 1. Main Rebased Chart (using annual YoY figure or recreate a simple one)
        fig_reb = clean_fig(core_data["annual_yoy_fig"])
        fig_reb.update_layout(title="Annual YoY: GLI vs Assets", width=800, height=500)
        p1 = os.path.join(tmpdir, "01_gli_yoy.png")
        fig_reb.write_image(p1, scale=2)
        img_paths.append(p1)
        
        if plumb_data:
            f_inj = clean_fig(plumb_data["fig_inject"])
            f_inj.update_layout(width=800, height=500, title="Stealth Liquidity (Reserves + BTFP + MMF)")
            p2 = os.path.join(tmpdir, "02_plumbing.png")
            f_inj.write_image(p2, scale=2)
            img_paths.append(p2)
            
            f_str = clean_fig(plumb_data["fig_stress"])
            f_str.update_layout(width=800, height=500)
            p3 = os.path.join(tmpdir, "03_stress.png")
            f_str.write_image(p3, scale=2)
            img_paths.append(p3)
            
        if yen_data:
            f_yen = clean_fig(yen_data["fig_usdjpy"])
            f_yen.update_layout(width=800, height=500)
            p4 = os.path.join(tmpdir, "04_usdjpy.png")
            f_yen.write_image(p4, scale=2)
            img_paths.append(p4)
            
            if "fig_vix" in yen_data:
                f_vix = clean_fig(yen_data["fig_vix"])
                f_vix.update_layout(width=800, height=500)
                p5 = os.path.join(tmpdir, "05_vix_unwind.png")
                f_vix.write_image(p5, scale=2)
                img_paths.append(p5)
                
            if "fig_boj" in yen_data:
                f_boj = clean_fig(yen_data["fig_boj"])
                f_boj.update_layout(width=800, height=500)
                p6 = os.path.join(tmpdir, "06_boj_impact.png")
                f_boj.write_image(p6, scale=2)
                img_paths.append(p6)
            
        print("Sending to Telegram...")
        send_telegram_message(text)
        if img_paths:
            send_telegram_media_group(img_paths)
            
        if RCLONE_REMOTE and RCLONE_DEST_PATH:
            rclone_upload(tmpdir)
            
    print("Done.")

if __name__ == "__main__":
    run_bot()
