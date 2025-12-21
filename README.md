# Research Paper Audiobook Converter

A desktop app that turns research PDFs into a clean, single-voice narration you can listen to. It structures the paper into a spoken-friendly script using Gemini, lets you review/edit the script, and then synthesizes it to MP3 using a TTS provider.

### Features
- Convert PDFs (default limit: 20 pages, configurable in Settings) into a natural narration script
- Pre‑flight editor to review and tweak the script
- Text‑to‑speech to MP3
- Audio providers: Google TTS (Studio) or ElevenLabs (default selection is Google TTS in code)
- Conversion modes: Summarized (Gemini-structured) or Verbatim (Gemini-cleaned raw text)

## Prerequisites
- Python 3.10+ recommended
- macOS, Windows, or Linux with Tk (Tkinter) available
- Internet access and API keys:
  - Gemini API key (for structuring the script)
  - Google TTS API key (Studio REST) and/or ElevenLabs API key

## Setup
1) Clone and enter the project directory
```bash
git clone <your-fork-or-repo-url>
cd article-reader
```

2) (Recommended) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

## Run the app
```bash
python app.py
```

If Tk fails to start due to missing Tkinter, install a Python build with Tk support (on macOS via python.org installer; on Linux, your distro's tkinter package).

## Configure API keys

### 1. Gemini API Key (required)
The Gemini API is used for text processing (summarization/cleaning). Set it in the app's Settings.

### 2. TTS Provider (choose one)

**Inworld AI (default):**
Create a `.env.local` file in the project directory:
```bash
INWORLD_API_KEY=your_base64_encoded_key_here
```

Available voices: Ashley, Brian, Cora, David, Emma, George, Hailey, Isaac, Julia, Kevin

**Gemini TTS:**
Uses the same Gemini API key. Select "gemini" as the TTS provider in Settings.

### In-app Settings
Open `Settings` (menu bar → Settings) to configure:
- Gemini API Key
- TTS Provider (Inworld or Gemini)
- Voice selection for each provider
- Default citation style and conversion mode

Note: A `config.json` file is automatically created to store your settings locally. Do not commit this file. To reset settings, close the app and delete `config.json`.

## Usage
1) File → Select PDF, choose a paper (page limit configurable in Settings, default: 20)
2) Choose Mode: `Summarized` (uses Gemini to structure) or `Verbatim` (uses Gemini to clean raw text for TTS)
3) Click `Convert`
   - The app estimates tokens and shows an estimated Gemini cost. Confirm to proceed. Both modes use Gemini (structuring or cleaning).
   - After processing, the script appears in the editor.
4) Review and optionally edit the script in the editor
5) Click `Generate Audio`
   - Choose a save location for the resulting MP3

### Limits and notes
- PDF limit: 20 pages by default (configurable in Settings → Conversion Defaults → Max PDF Pages)
- Google TTS Studio uses short input chunks; the app automatically splits the script into byte‑safe parts and concatenates the MP3
- Costs: You are responsible for any API usage fees for Gemini, Google TTS, or ElevenLabs

### Cost estimation
- Before Convert in Summarized mode: shows an estimated Gemini cost based on tokens.
- Before Generate Audio: shows an estimated audio cost based on characters and your selected provider. Rates are configurable via `usd_per_million_chars_google_tts` and `usd_per_million_chars_elevenlabs` in `config.json` (defaults are approximations).

## Command-line usage
You can run the full two-step conversion (Gemini → TTS) without opening the GUI. The CLI prints estimated costs for both steps and shows progress with tqdm progress bars.

```bash
python cli.py /path/to/article.pdf \
  --mode Summarized \
  --citations "Ignore" \
  --output /path/to/output.mp3
```

- `--mode`: `Summarized` (default) structures content; `Verbatim` cleans raw text with image descriptions.
- `--citations`: `Ignore` (default) or `Subtle Mention`.
- `--output`: optional MP3 destination; default is `<article_basename>.mp3` in the same directory as the input PDF.
- `--config`: optional path to `config.json` (defaults to the project's `config.json`).
- `--compress-script`: Save script as `.txt.gz` instead of plain `.txt`.
- `--quiet` / `-q`: Suppress progress bars.
- `--verbose` / `-v`: Show detailed progress messages.

The CLI will:
- Print estimated Gemini cost (based on token estimation of the PDF)
- Run Gemini to produce the script (not printed)
- Print estimated TTS cost (based on character count of the produced script)
- Generate the MP3 (streaming to disk for memory efficiency)
- Save the script text next to it as `.txt` or `.txt.gz`

## Batch processing (for large jobs)
For processing many PDFs, use the batch processor. It supports:
- **Background execution** — continues even if you close the terminal or your computer sleeps
- **Checkpoint/resume** — saves progress after each PDF, can resume if interrupted
- **Status monitoring** — check progress without interrupting the job

### Start a batch job
```bash
# Process all PDFs in a folder
python batch.py start /path/to/papers/*.pdf --output ./audio_papers/

# With options
python batch.py start papers/*.pdf --mode Verbatim --citations "Subtle Mention"
```

### Run in background (daemon mode)
This lets you close the terminal and the job continues:
```bash
python batch.py start papers/*.pdf --daemon

# Or use nohup
nohup python batch.py start papers/*.pdf &
```

### Check status
```bash
python batch.py status

# With more detail
python batch.py status --verbose --log
```

Example output:
```
==================================================
BATCH JOB: 20250620_143022
==================================================

Status: 🟢 RUNNING
PID: 12345

Progress: 5/20 completed
[████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25.0%

  ✅ Completed: 5
  ❌ Failed:    0
  ⏳ Pending:   15
```

### Resume interrupted job
If your computer restarts or the process is interrupted:
```bash
python batch.py resume
```

### Other commands
```bash
# Cancel a running job (saves progress)
python batch.py cancel

# Retry only the failed PDFs
python batch.py retry-failed

# Clear state to start fresh
python batch.py clear
```

### Batch files
The batch processor creates these files in the project directory:
- `.batch_state.json` — Job progress (do not edit while running)
- `.batch_lock` — Prevents multiple jobs running simultaneously
- `batch.log` — Detailed log of all operations

## Troubleshooting
- Missing API Key errors: Ensure keys are set in Settings and saved
- Google TTS: If you see errors like "No audioContent in response" or quota/permission errors, verify the Studio API key, project billing, and TTS API enablement
- ElevenLabs: Voice fetch failures usually indicate an invalid key or network issues
- PDF parse errors: Ensure the PDF is readable; very image‑heavy or scanned PDFs may extract little text

## Optional: Verify TTS with the included tests
Run these small scripts to sanity‑check your keys (they will make billable API calls):
```bash
python test_google_studio_tts.py   # requires Google TTS API key
python test_elevenlabs.py          # requires ElevenLabs API key
```

## Security
- Keys are stored locally in `config.json`. Keep this file private and out of version control.
- Avoid sharing logs that include request details or keys.

## Disclaimer
- You supply and control your own API keys. You are solely responsible for how this tool is used, for any content you process or generate with it, for complying with the terms/policies of the API providers (e.g., Gemini, Google TTS Studio, ElevenLabs), and for any associated fees or charges.
- The author/maintainers do not control, monitor, or assume responsibility for your usage. No warranty is provided; use at your own risk. The author/maintainers are not liable for misuse, violations of third‑party policies, or costs incurred.

## Non‑Commercial Use
- This project is intended for personal and educational use only and is not intended for commercial use.
- Do not use the outputs to create, distribute, or monetize content in a commercial context.
- You are responsible for ensuring you have the necessary rights to any source material you process. The author/maintainers are not responsible for copyright violations or related disputes arising from your use of this tool.
