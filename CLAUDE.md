# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research Paper Audiobook Converter - A Python desktop app that converts research PDFs into natural-sounding MP3 audiobooks. Uses Gemini AI for text structuring/cleaning and TTS providers (Inworld AI or Gemini TTS) for audio synthesis.

## Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
# GUI
python app.py

# CLI single conversion
python cli.py /path/to/article.pdf --mode Summarized --citations "Ignore" --output /path/to/output.mp3

# Batch processing
python batch.py start /path/to/pdfs/*.pdf --output ./audio/
python batch.py status
python batch.py resume
python batch.py cancel
```

### CLI Options
- `--mode`: `Summarized` (default) or `Verbatim`
- `--citations`: `Ignore` (default) or `Subtle Mention`
- `--compress-script`: Save script as `.txt.gz`
- `--quiet` / `--verbose`: Control output verbosity

## Architecture

### Core Files
| File | Purpose |
|------|---------|
| `app.py` | GUI application (CustomTkinter) - main window, settings, progress UI |
| `processing.py` | Core pipeline: PDF parsing, Gemini integration, TTS synthesis |
| `config.py` | Configuration management, pricing cache, API key handling |
| `cli.py` | Command-line interface with tqdm progress |
| `batch.py` | Batch processing with checkpointing and background execution |

### Processing Pipeline
```
PDF → parse_pdf() → Gemini (structure/clean) → TTS (Inworld/Gemini) → MP3
```

Two conversion modes:
- **Summarized**: `structure_with_gemini()` - AI-structured narration
- **Verbatim**: `clean_with_gemini()` - Raw text with AI cleanup

### Threading Model (GUI)
- Main thread: UI rendering
- Worker thread: `ConversionWorker.run()` via `threading.Thread`
- Communication: `queue.Queue` for thread-safe event passing
- Cancellation: `threading.Event` for graceful interruption
- UI updates: `_process_ui_queue()` via `after()` for thread-safe updates

### TTS Provider Fallback
Inworld AI is default. On quota exceeded (HTTP 429) or auth failures, automatically falls back to Gemini TTS.

### Key Classes
- `ConversionWorker` (processing.py): Static class orchestrating conversion
- `PDFCache` (processing.py): Thread-safe in-memory cache (5 PDFs max)
- `BatchStateManager` (batch.py): Checkpoint/resume functionality
- `ConfigManager` (config.py): JSON config persistence with defaults merging

### Configuration Files
- `config.json` - User settings (auto-created, do not commit)
- `.env.local` - API keys (optional, takes precedence)
- `.batch_state.json` - Batch job checkpoint
- `.pricing_cache.json` - Cached model pricing

## API Keys Required
- **Gemini API Key**: Required for text processing and TTS fallback (`GEMINI_API_KEY` in `.env.local`, or set in Settings)
- **Inworld API Key**: For Inworld TTS (base64-encoded, `INWORLD_API_KEY` in `.env.local`, or set in Settings)
- **Gemini TTS**: Uses same Gemini API key (select "gemini" as TTS provider)
