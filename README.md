# Research Paper Audiobook Converter

A desktop app that turns research PDFs into a clean, single-voice narration you can listen to. It structures the paper into a spoken-friendly script using Gemini, lets you review/edit the script, and then synthesizes it to MP3 using a TTS provider.

### Features
- Convert PDFs (up to 30 pages) into a natural narration script
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

## Configure API keys (in‑app)
On first run, open `Settings` (menu bar → Settings) and fill in:

- Gemini API Key: required to structure the paper into a narration script.
- Audio Provider: choose `google_tts` or `elevenlabs`.

For Google TTS (Studio):
- Google TTS API Key: required if using Google TTS.
- Voice Name (e.g., `en-US-Studio-O`), Language Code (e.g., `en-US`)
- Speaking Rate and Pitch

For ElevenLabs:
- ElevenLabs API Key
- Click `Fetch Voices`, then choose a voice in the dropdown
- Adjust Stability and Clarity (similarity boost)

Note: A `config.json` file is automatically created to store your settings locally. Do not commit this file. To reset settings, close the app and delete `config.json`.

## Usage
1) File → Select PDF, choose a paper (≤ 30 pages)
2) Choose Mode: `Summarized` (uses Gemini to structure) or `Verbatim` (uses Gemini to clean raw text for TTS)
3) Click `Convert`
   - The app estimates tokens and shows an estimated Gemini cost. Confirm to proceed. Both modes use Gemini (structuring or cleaning).
   - After processing, the script appears in the editor.
4) Review and optionally edit the script in the editor
5) Click `Generate Audio`
   - Choose a save location for the resulting MP3

### Limits and notes
- PDF limit: 30 pages (enforced)
- Google TTS Studio uses short input chunks; the app automatically splits the script into byte‑safe parts and concatenates the MP3
- Costs: You are responsible for any API usage fees for Gemini, Google TTS, or ElevenLabs

### Cost estimation
- Before Convert in Summarized mode: shows an estimated Gemini cost based on tokens.
- Before Generate Audio: shows an estimated audio cost based on characters and your selected provider. Rates are configurable via `usd_per_million_chars_google_tts` and `usd_per_million_chars_elevenlabs` in `config.json` (defaults are approximations).

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
