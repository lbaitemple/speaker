# Speaker — Discord Voice Bot for Mini Pupper

## Prerequisites

Install system dependencies once (required before first run):

```bash
sudo apt install python3.12-venv portaudio19-dev python3-pyaudio -y
```

## Setup & Run

Create a `.env` file in this directory with your credentials:

```
DISCORD_BOT_TOKEN=your_token_here
DISCORD_LISTENER_TOKEN=your_listener_token_here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google-creds.json
TARGET_APP_USER_IDS=123456789
TARGET_CHANNEL_ID=987654321
```

Then run:

```bash
bash start.sh
```

`start.sh` will automatically:
1. Enable system-site-packages in the venv (gives access to MangDang LCD drivers)
2. Install all Python dependencies from `requirements.txt`
3. Start the bot
4. Restart `robot.service` when the bot exits (Ctrl+C or crash)

## Stopping the bot

**Option 1 — Ctrl+C** (clean shutdown):

Press **Ctrl+C** once. The bot shuts down and `robot.service` is restarted automatically.

**Option 2 — Ctrl+Z then kill** (if Ctrl+C is unresponsive):

```bash
# Suspend the job
Ctrl+Z

# List suspended jobs to find the job number
jobs

# Kill it (replace 1 with the actual job number)
kill %1
```

Either way, `start.sh` will restart `robot.service` after the process exits.
