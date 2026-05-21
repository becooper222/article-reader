#!/usr/bin/env python3
"""
Web interface for Research Paper Audiobook Converter.
Mobile-friendly Flask app for converting PDFs to MP3 audiobooks.
"""
import json
import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

import requests as http_requests
from flask import Flask, Response, jsonify, render_template_string, request, send_file

from config import ConfigManager
from processing import (
    ConversionWorker,
    PDFCache,
    clean_with_gemini,
    parse_pdf,
    structure_with_gemini,
)

app = Flask(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
OUTPUTS_DIR = os.path.join(tempfile.gettempdir(), "article_reader_outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

_jobs: dict = {}
_jobs_lock = threading.Lock()


class Job:
    def __init__(self, filename: str):
        self.id = str(uuid.uuid4())
        self.events: queue.Queue = queue.Queue()
        self.output_path: str | None = None
        self.filename = filename
        self.done = False
        self.error: str | None = None
        self.cancel_event = threading.Event()

    def emit(self, event_type: str, data: str):
        self.events.put({"type": event_type, "data": data})

    def finish(self, output_path: str):
        self.output_path = output_path
        self.done = True
        self.emit("done", json.dumps({"download_url": f"/download/{self.id}"}))

    def fail(self, error: str):
        self.error = error
        self.done = True
        self.emit("error", error)


def run_conversion(job: Job, pdf_path: str, mode: str, citations: str, cleanup_pdf: bool):
    """Run the full conversion pipeline in a background thread."""
    try:
        cfg_mgr = ConfigManager(CONFIG_PATH)
        cfg_mgr.ensure_config()
        cfg = cfg_mgr.load()

        api_key = cfg.get("gemini_api_key", "")
        if not api_key:
            job.fail("Gemini API key not configured. Please visit /settings to add it.")
            return

        model_name = cfg.get("model_name", "gemini-2.0-flash")

        def log(msg):
            job.emit("status", msg)

        def progress_cb(current, total, status=""):
            job.emit("progress", json.dumps({
                "current": current,
                "total": total,
                "status": status,
            }))

        # Step 1: Parse PDF
        job.emit("status", "Parsing PDF...")
        try:
            text, image_locations = parse_pdf(pdf_path, progress_callback=progress_cb)
        except Exception as e:
            job.fail(f"PDF parsing failed: {e}")
            return

        if job.cancel_event.is_set():
            job.emit("cancelled", "Cancelled by user")
            return

        # Step 2: Generate script with Gemini
        job.emit("status", f"Generating script with Gemini ({mode} mode)...")
        try:
            if mode == "Verbatim":
                script = clean_with_gemini(
                    text=text,
                    pdf_path=pdf_path,
                    image_locations=image_locations,
                    api_key=api_key,
                    model_name=model_name,
                    citation_style=citations,
                    cancel_event=job.cancel_event,
                    log=log,
                    progress_callback=progress_cb,
                )
            else:
                script = structure_with_gemini(
                    text=text,
                    pdf_path=pdf_path,
                    image_locations=image_locations,
                    api_key=api_key,
                    model_name=model_name,
                    citation_style=citations,
                    cancel_event=job.cancel_event,
                    log=log,
                    progress_callback=progress_cb,
                )
        except Exception as e:
            job.fail(f"Script generation failed: {e}")
            return

        if job.cancel_event.is_set():
            job.emit("cancelled", "Cancelled by user")
            return

        # Step 3: Generate audio
        job.emit("status", "Generating audio...")
        stem = Path(job.filename).stem[:60]
        output_path = os.path.join(OUTPUTS_DIR, f"{stem}_{job.id[:8]}.mp3")

        try:
            allow_fallback = cfg.get("tts_fallback_enabled", True)
            file_size = ConversionWorker.generate_audio_streaming(
                script,
                output_path,
                cfg_mgr,
                job.cancel_event,
                log=log,
                progress_callback=progress_cb,
                allow_fallback=allow_fallback,
            )
        except Exception as e:
            job.fail(f"Audio generation failed: {e}")
            return

        if file_size is None:
            job.emit("cancelled", "Cancelled by user")
            return

        job.emit("status", f"Done! Audio file: {file_size / (1024*1024):.1f} MB")
        job.finish(output_path)

    finally:
        if cleanup_pdf and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except Exception:
                pass
        PDFCache.clear()


MAIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Article to Audio</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --accent: #5b8dee;
    --accent-hover: #4a7cd4;
    --text: #e8eaf0;
    --muted: #8b8fa8;
    --success: #4caf7d;
    --error: #e05c5c;
    --radius: 12px;
  }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    min-height: 100vh;
    padding: 0;
  }
  .topbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .topbar h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }
  .topbar .tagline { font-size: 12px; color: var(--muted); margin-top: 1px; }
  .topbar a {
    color: var(--muted);
    text-decoration: none;
    font-size: 14px;
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    transition: color .2s, border-color .2s;
  }
  .topbar a:hover { color: var(--text); border-color: var(--accent); }
  .container { max-width: 600px; margin: 0 auto; padding: 24px 16px 80px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
  }
  .card-title { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 14px; }
  label { display: block; font-size: 14px; color: var(--muted); margin-bottom: 6px; }
  input[type="url"], input[type="text"], input[type="password"], select, textarea {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 15px;
    padding: 11px 14px;
    outline: none;
    transition: border-color .2s;
    -webkit-appearance: none;
  }
  input:focus, select:focus, textarea:focus { border-color: var(--accent); }
  select { cursor: pointer; }
  .divider { text-align: center; color: var(--muted); font-size: 13px; margin: 12px 0; }
  .file-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 11px 14px;
    background: var(--bg);
    border: 1px dashed var(--border);
    border-radius: 8px;
    color: var(--muted);
    font-size: 14px;
    cursor: pointer;
    transition: border-color .2s, color .2s;
  }
  .file-btn:hover { border-color: var(--accent); color: var(--text); }
  .file-btn.has-file { border-color: var(--success); color: var(--success); border-style: solid; }
  input[type="file"] { display: none; }
  .options-row { display: flex; gap: 10px; }
  .options-row > div { flex: 1; }
  .btn {
    display: block;
    width: 100%;
    padding: 14px;
    background: var(--accent);
    color: #fff;
    font-size: 16px;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background .2s, opacity .2s;
    margin-top: 4px;
  }
  .btn:hover { background: var(--accent-hover); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-outline {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 14px;
    padding: 10px;
    margin-top: 10px;
  }
  .btn-outline:hover { border-color: var(--error); color: var(--error); background: transparent; }
  .btn-success { background: var(--success); }
  .btn-success:hover { background: #3a9e6a; }
  .progress-wrap { margin: 14px 0 4px; }
  .progress-bar-bg {
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 99px;
    transition: width .3s ease;
    width: 0%;
  }
  .status-msg {
    font-size: 13px;
    color: var(--muted);
    margin-top: 8px;
    min-height: 18px;
    word-break: break-word;
  }
  .hidden { display: none !important; }
  .error-box {
    background: rgba(224, 92, 92, 0.1);
    border: 1px solid var(--error);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 14px;
    color: var(--error);
    margin-top: 12px;
    word-break: break-word;
  }
  .note { font-size: 12px; color: var(--muted); margin-top: 8px; }
  @media (max-width: 400px) {
    .options-row { flex-direction: column; }
  }
</style>
</head>
<body>
<div class="topbar">
  <div>
    <div class="topbar h1" style="font-size:18px;font-weight:700">Article to Audio</div>
    <div class="tagline">Convert research PDFs to MP3</div>
  </div>
  <a href="/settings">Settings</a>
</div>

<div class="container">
  <div class="card" id="form-card">
    <div class="card-title">Source PDF</div>
    <label for="pdf-url">Paste a URL</label>
    <input type="url" id="pdf-url" placeholder="https://example.com/paper.pdf" autocorrect="off" autocapitalize="none" />
    <div class="divider">— or —</div>
    <label class="file-btn" id="file-label" for="pdf-file">
      <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      <span id="file-label-text">Choose PDF file</span>
    </label>
    <input type="file" id="pdf-file" accept=".pdf">
    <p class="note">Max 20 pages per article</p>
  </div>

  <div class="card">
    <div class="card-title">Options</div>
    <div class="options-row">
      <div>
        <label for="mode">Mode</label>
        <select id="mode">
          <option value="Summarized">Summarized</option>
          <option value="Verbatim">Verbatim</option>
        </select>
      </div>
      <div>
        <label for="citations">Citations</label>
        <select id="citations">
          <option value="Ignore">Ignore</option>
          <option value="Subtle Mention">Subtle Mention</option>
        </select>
      </div>
    </div>
  </div>

  <div id="progress-card" class="card hidden">
    <div class="card-title">Converting</div>
    <div class="progress-wrap">
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="progress-fill"></div></div>
    </div>
    <div class="status-msg" id="status-msg">Starting...</div>
    <button class="btn btn-outline" id="cancel-btn">Cancel</button>
  </div>

  <div id="error-card" class="hidden">
    <div class="error-box" id="error-msg"></div>
  </div>

  <div id="done-card" class="card hidden">
    <div class="card-title">Ready to download</div>
    <a class="btn btn-success" id="download-btn" href="#" download>Download MP3</a>
    <button class="btn btn-outline" id="convert-another-btn" style="margin-top:10px">Convert another</button>
  </div>

  <button class="btn" id="convert-btn">Convert to Audio</button>
</div>

<script>
let currentJobId = null;
let eventSource = null;

const urlInput = document.getElementById('pdf-url');
const fileInput = document.getElementById('pdf-file');
const fileLabel = document.getElementById('file-label');
const fileLabelText = document.getElementById('file-label-text');
const convertBtn = document.getElementById('convert-btn');
const progressCard = document.getElementById('progress-card');
const progressFill = document.getElementById('progress-fill');
const statusMsg = document.getElementById('status-msg');
const cancelBtn = document.getElementById('cancel-btn');
const errorCard = document.getElementById('error-card');
const errorMsg = document.getElementById('error-msg');
const doneCard = document.getElementById('done-card');
const downloadBtn = document.getElementById('download-btn');
const convertAnotherBtn = document.getElementById('convert-another-btn');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) {
    fileLabelText.textContent = fileInput.files[0].name;
    fileLabel.classList.add('has-file');
    urlInput.value = '';
  }
});

urlInput.addEventListener('input', () => {
  if (urlInput.value) {
    fileInput.value = '';
    fileLabelText.textContent = 'Choose PDF file';
    fileLabel.classList.remove('has-file');
  }
});

convertBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  const file = fileInput.files[0];
  if (!url && !file) {
    alert('Please provide a PDF URL or choose a file.');
    return;
  }

  const mode = document.getElementById('mode').value;
  const citations = document.getElementById('citations').value;

  convertBtn.disabled = true;
  showProgress();

  const formData = new FormData();
  if (file) formData.append('pdf_file', file);
  else formData.append('pdf_url', url);
  formData.append('mode', mode);
  formData.append('citations', citations);

  let jobId;
  try {
    const resp = await fetch('/convert', { method: 'POST', body: formData });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Unknown error');
    jobId = data.job_id;
  } catch (e) {
    showError(e.message);
    convertBtn.disabled = false;
    return;
  }

  currentJobId = jobId;
  listenToJob(jobId);
});

cancelBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  cancelBtn.disabled = true;
  await fetch(`/cancel/${currentJobId}`, { method: 'POST' });
  if (eventSource) eventSource.close();
  resetUI();
});

convertAnotherBtn.addEventListener('click', resetUI);

function showProgress() {
  progressCard.classList.remove('hidden');
  doneCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  progressFill.style.width = '5%';
  statusMsg.textContent = 'Starting...';
  cancelBtn.disabled = false;
}

function showError(msg) {
  progressCard.classList.add('hidden');
  doneCard.classList.add('hidden');
  errorCard.classList.remove('hidden');
  errorMsg.textContent = msg;
  convertBtn.disabled = false;
}

function showDone(downloadUrl) {
  progressCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  doneCard.classList.remove('hidden');
  downloadBtn.href = downloadUrl;
  convertBtn.disabled = false;
}

function resetUI() {
  if (eventSource) { eventSource.close(); eventSource = null; }
  currentJobId = null;
  progressCard.classList.add('hidden');
  doneCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  progressFill.style.width = '0%';
  convertBtn.disabled = false;
}

function listenToJob(jobId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/stream/${jobId}`);
  let progress = 5;

  eventSource.addEventListener('status', e => {
    statusMsg.textContent = e.data;
    if (progress < 85) { progress += 5; progressFill.style.width = progress + '%'; }
  });

  eventSource.addEventListener('progress', e => {
    try {
      const d = JSON.parse(e.data);
      if (d.total > 0) {
        const pct = Math.min(95, 10 + Math.round((d.current / d.total) * 75));
        progressFill.style.width = pct + '%';
      }
      if (d.status) statusMsg.textContent = d.status;
    } catch {}
  });

  eventSource.addEventListener('done', e => {
    eventSource.close();
    progressFill.style.width = '100%';
    statusMsg.textContent = 'Complete!';
    try {
      const d = JSON.parse(e.data);
      setTimeout(() => showDone(d.download_url), 500);
    } catch { showError('Conversion finished but download link unavailable.'); }
  });

  eventSource.addEventListener('error', e => {
    eventSource.close();
    showError(e.data || 'Conversion failed. Check your API key in Settings.');
  });

  eventSource.addEventListener('cancelled', e => {
    eventSource.close();
    resetUI();
  });

  eventSource.onerror = () => {
    if (eventSource.readyState === EventSource.CLOSED) return;
    eventSource.close();
    showError('Connection lost. The conversion may still be running.');
  };
}
</script>
</body>
</html>
"""

SETTINGS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Settings — Article to Audio</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --accent: #5b8dee;
    --text: #e8eaf0;
    --muted: #8b8fa8;
    --success: #4caf7d;
    --error: #e05c5c;
    --radius: 12px;
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  .topbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    position: sticky; top: 0; z-index: 10;
  }
  .topbar a { color: var(--muted); text-decoration: none; font-size: 20px; line-height: 1; }
  .topbar h1 { font-size: 18px; font-weight: 700; }
  .container { max-width: 600px; margin: 0 auto; padding: 24px 16px 80px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 16px; }
  .card-title { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 14px; }
  .field { margin-bottom: 14px; }
  .field:last-child { margin-bottom: 0; }
  label { display: block; font-size: 14px; color: var(--muted); margin-bottom: 6px; }
  input[type="text"], input[type="password"], select {
    width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); font-size: 15px; padding: 11px 14px; outline: none;
    transition: border-color .2s; -webkit-appearance: none; font-family: inherit;
  }
  input:focus, select:focus { border-color: var(--accent); }
  .hint { font-size: 12px; color: var(--muted); margin-top: 5px; }
  .btn {
    display: block; width: 100%; padding: 14px;
    background: var(--accent); color: #fff; font-size: 16px; font-weight: 600;
    border: none; border-radius: 8px; cursor: pointer; transition: background .2s;
    font-family: inherit;
  }
  .btn:hover { background: #4a7cd4; }
  .toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: var(--success); color: #fff; padding: 12px 20px;
    border-radius: 8px; font-size: 14px; font-weight: 600;
    opacity: 0; transition: opacity .3s; pointer-events: none;
    white-space: nowrap;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
<div class="topbar">
  <a href="/" title="Back">&#8592;</a>
  <h1>Settings</h1>
</div>

<div class="container">
  <form id="settings-form">
    <div class="card">
      <div class="card-title">API Keys</div>
      <div class="field">
        <label for="gemini_api_key">Gemini API Key <span style="color:var(--error)">*</span></label>
        <input type="password" id="gemini_api_key" name="gemini_api_key" placeholder="AIza..." value="{{ config.gemini_api_key }}" autocomplete="off" />
        <p class="hint">Required for text processing and Gemini TTS. Get one at <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:var(--accent)">aistudio.google.com</a></p>
      </div>
      <div class="field">
        <label for="inworld_api_key">Inworld API Key <span style="color:var(--muted)">(optional)</span></label>
        <input type="password" id="inworld_api_key" name="inworld_api_key" placeholder="Base64-encoded credentials" value="{{ config.inworld_api_key }}" autocomplete="off" />
        <p class="hint">Optional. If not set, Gemini TTS is used.</p>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Text Model</div>
      <div class="field">
        <label for="model_name">Gemini Model</label>
        <select id="model_name" name="model_name">
          {% for m in text_models %}
          <option value="{{ m }}" {% if config.model_name == m %}selected{% endif %}>{{ m }}</option>
          {% endfor %}
        </select>
        <p class="hint">Used for PDF summarization / cleaning. Flash models are faster and cheaper.</p>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Text-to-Speech</div>
      <div class="field">
        <label for="tts_provider">Provider</label>
        <select id="tts_provider" name="tts_provider" onchange="updateVoiceList()">
          <option value="inworld" {% if config.tts_provider == 'inworld' %}selected{% endif %}>Inworld AI (fallback to Gemini if no key)</option>
          <option value="gemini" {% if config.tts_provider == 'gemini' %}selected{% endif %}>Gemini TTS</option>
        </select>
      </div>
      <div class="field" id="gemini-voice-field" {% if config.tts_provider == 'inworld' %}style="display:none"{% endif %}>
        <label for="tts_voice_name">Gemini Voice</label>
        <select id="tts_voice_name" name="tts_voice_name">
          {% for v in gemini_voices %}
          <option value="{{ v.name }}" {% if config.tts_voice_name == v.name %}selected{% endif %}>{{ v.name }} — {{ v.style }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="field" id="inworld-voice-field" {% if config.tts_provider == 'gemini' %}style="display:none"{% endif %}>
        <label for="inworld_voice_id">Inworld Voice</label>
        <select id="inworld_voice_id" name="inworld_voice_id">
          {% for v in inworld_voices %}
          <option value="{{ v.name }}" {% if config.inworld_voice_id == v.name %}selected{% endif %}>{{ v.name }} — {{ v.style }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Defaults</div>
      <div class="field">
        <label for="conversion_mode">Conversion Mode</label>
        <select id="conversion_mode" name="conversion_mode">
          <option value="Summarized" {% if config.conversion_mode == 'Summarized' %}selected{% endif %}>Summarized — AI-structured narration</option>
          <option value="Verbatim" {% if config.conversion_mode == 'Verbatim' %}selected{% endif %}>Verbatim — cleaned original text</option>
        </select>
      </div>
      <div class="field">
        <label for="citation_style">Citation Style</label>
        <select id="citation_style" name="citation_style">
          <option value="Ignore" {% if config.citation_style == 'Ignore' %}selected{% endif %}>Ignore citations</option>
          <option value="Subtle Mention" {% if config.citation_style == 'Subtle Mention' %}selected{% endif %}>Subtle mention</option>
        </select>
      </div>
    </div>

    <button type="submit" class="btn">Save Settings</button>
  </form>
</div>

<div class="toast" id="toast">Settings saved!</div>

<script>
function updateVoiceList() {
  const provider = document.getElementById('tts_provider').value;
  document.getElementById('gemini-voice-field').style.display = provider === 'gemini' ? '' : 'none';
  document.getElementById('inworld-voice-field').style.display = provider === 'inworld' ? '' : 'none';
}

document.getElementById('settings-form').addEventListener('submit', async e => {
  e.preventDefault();
  const form = e.target;
  const data = {};
  new FormData(form).forEach((v, k) => { data[k] = v; });

  const resp = await fetch('/settings', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) });
  if (resp.ok) {
    const toast = document.getElementById('toast');
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2500);
  } else {
    alert('Failed to save settings.');
  }
});
</script>
</body>
</html>
"""

TEXT_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


@app.route("/")
def index():
    return render_template_string(MAIN_HTML)


@app.route("/settings", methods=["GET"])
def settings_get():
    mgr = ConfigManager(CONFIG_PATH)
    mgr.ensure_config()
    cfg = mgr.load()
    # Mask keys for display: show empty string, not the actual key
    display_cfg = dict(cfg)
    # We keep actual values so the password fields pre-fill (browser masks them)
    gemini_voices = ConfigManager.get_available_voices("gemini")
    inworld_voices = ConfigManager.get_available_voices("inworld")
    return render_template_string(
        SETTINGS_HTML,
        config=display_cfg,
        text_models=TEXT_MODELS,
        gemini_voices=gemini_voices,
        inworld_voices=inworld_voices,
    )


@app.route("/settings", methods=["POST"])
def settings_post():
    data = request.get_json(force=True)
    allowed = {
        "gemini_api_key", "inworld_api_key", "model_name",
        "tts_provider", "tts_voice_name", "inworld_voice_id",
        "conversion_mode", "citation_style",
    }
    updates = {k: v for k, v in data.items() if k in allowed}
    mgr = ConfigManager(CONFIG_PATH)
    mgr.ensure_config()
    mgr.save(updates)
    return jsonify({"ok": True})


@app.route("/convert", methods=["POST"])
def convert():
    mode = request.form.get("mode", "Summarized")
    citations = request.form.get("citations", "Ignore")

    pdf_url = request.form.get("pdf_url", "").strip()
    pdf_file = request.files.get("pdf_file")

    if not pdf_url and not pdf_file:
        return jsonify({"error": "No PDF provided"}), 400

    if pdf_file:
        filename = pdf_file.filename or "upload.pdf"
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf_file.save(tmp.name)
        pdf_path = tmp.name
        cleanup = True
    else:
        # Download the PDF from URL
        filename = pdf_url.split("/")[-1].split("?")[0] or "article.pdf"
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        try:
            resp = http_requests.get(pdf_url, timeout=60, stream=True)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            for chunk in resp.iter_content(chunk_size=65536):
                tmp.write(chunk)
            tmp.close()
            pdf_path = tmp.name
            cleanup = True
        except Exception as e:
            return jsonify({"error": f"Failed to download PDF: {e}"}), 400

    job = Job(filename)
    with _jobs_lock:
        _jobs[job.id] = job

    thread = threading.Thread(
        target=run_conversion,
        args=(job, pdf_path, mode, citations, cleanup),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job.id})


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job:
        job.cancel_event.set()
    return jsonify({"ok": True})


@app.route("/stream/<job_id>")
def stream(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return Response("data: Job not found\n\n", mimetype="text/event-stream")

    def generate():
        ping_interval = 15  # seconds
        last_ping = time.time()
        while True:
            try:
                event = job.events.get(timeout=1.0)
                yield f"event: {event['type']}\ndata: {event['data']}\n\n"
                if event["type"] in ("done", "error", "cancelled"):
                    break
            except queue.Empty:
                now = time.time()
                if now - last_ping >= ping_interval:
                    yield ": ping\n\n"
                    last_ping = now
                if job.done:
                    break

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/download/<job_id>")
def download(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or not job.output_path or not os.path.exists(job.output_path):
        return "File not found", 404

    stem = Path(job.filename).stem[:60]
    download_name = f"{stem}.mp3"
    return send_file(job.output_path, as_attachment=True, download_name=download_name,
                     mimetype="audio/mpeg")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Starting Article to Audio web app on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
