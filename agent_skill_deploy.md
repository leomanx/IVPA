# Agent Skill: Automated GLI Weekly Report

This document contains the instructions and requirements for deploying the `gli_telegram_bot.py` script on a Raspberry Pi 4 running Linux, ensuring it runs automatically via Cronjob.

## 1. Prerequisites (Raspberry Pi Setup)

Connect to your Raspberry Pi via SSH and run the following commands to prepare the environment:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Rclone (if not already installed)
sudo apt install rclone -y

# Install dependencies for image generation (Kaleido requires some system libraries)
sudo apt install libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 -y
```

## 2. Python Environment Setup

Clone your repository to the Raspberry Pi, or copy the files (`gli_lib.py`, `gli_telegram_bot.py`). Then set up a virtual environment:

```bash
# Navigate to your project directory
cd /home/mmkpi4/Automate_GLI_Analysis (or wherever you placed the files)

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required Python packages
pip install pandas numpy plotly yfinance requests python-dotenv
pip install fredapi statsmodels scipy

# **Crucial for generating image files from Plotly**
pip install -U kaleido
```

## 3. Environment Variables (`.env`)

Create a file named `.env` in the same directory as your script:

```bash
nano /home/mmkpi4/netguard/.env
```

Paste your credentials into this file:

```env
FRED_API_KEY=your_fred_api_key_here
TELEGRAM_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

## 4. Test the Script Manually

Before setting up the automation, verify that the script runs successfully and sends the message to Telegram:

```bash
# Ensure you are still in the virtual environment
python gli_telegram_bot.py
```
Check your Telegram group for the message and charts. Also, check your Rclone destination to see if the files were uploaded.

## 5. Automate with Cronjob

Once verified, set up a cronjob to run the script automatically every week.

```bash
crontab -e
```

Add the following line to the end of the file. This example runs the script every Monday at 07:00 AM. Adjust the paths if your directory is different.

```bash
# Run every Monday at 07:00 AM
0 7 * * 1 cd /home/mmkpi4/Automate_GLI_Analysis && /home/mmkpi4/Automate_GLI_Analysis/venv/bin/python gli_telegram_bot.py >> /home/mmkpi4/logs/gli_analysis/gli-analysis.log 2>&1
```

> [!TIP]
> **Cron Syntax Breakdown:**
> `0 7 * * 1` means:
> - `0`: Minute 0
> - `7`: Hour 7 (7 AM)
> - `*`: Any day of the month
> - `*`: Any month
> - `1`: Monday

> [!NOTE]
> The `>> bot_cron.log 2>&1` part ensures that any output or errors from the script are saved to a log file, making it easy to troubleshoot if the bot doesn't send the message.
