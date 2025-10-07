# Research Paper Audiobook Converter

A desktop app that turns research PDFs into a clean, single-voice narration you can listen to. It structures the paper into a spoken-friendly script using Gemini, lets you review/edit the script, and then synthesizes it to MP3 using a TTS provider.

### Features
- Convert PDFs (up to 20 pages) into a natural narration script
- Pre‑flight editor to review and tweak the script
- Text‑to‑speech to MP3
- Audio providers: Google TTS (Studio) or ElevenLabs (default selection is Google TTS in code)

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
1) File → Select PDF, choose a paper (≤ 20 pages)
2) Click `Convert`
   - The app estimates tokens and shows an estimated Gemini cost. Confirm to proceed.
   - After processing, the structured script appears in the editor.
3) Review and optionally edit the script in the editor
4) Click `Generate Audio`
   - Choose a save location for the resulting MP3

### Limits and notes
- PDF limit: 20 pages (enforced)
- Google TTS Studio uses short input chunks; the app automatically splits the script into byte‑safe parts and concatenates the MP3
- Costs: You are responsible for any API usage fees for Gemini, Google TTS, or ElevenLabs

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
