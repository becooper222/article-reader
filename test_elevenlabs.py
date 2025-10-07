#!/usr/bin/env python3
"""
Standalone ElevenLabs debug runner.

- Loads text from summarized_text_ex.txt by default
- Reads ElevenLabs config from config.json (or env overrides)
- Chunks text, sends requests with detailed logging
- Saves concatenated audio to debug_output.mp3

Usage examples:
  python test_elevenlabs.py
  ELEVEN_API_KEY=... python test_elevenlabs.py --first-n-chunks 1 --output /tmp/test.mp3
"""
import argparse
import json
import os
import sys
from typing import List

import requests


PROJECT_DIR = os.path.dirname(__file__)
DEFAULT_TEXT_PATH = os.path.join(PROJECT_DIR, 'summarized_text_ex.txt')
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, 'config.json')


def read_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] config.json not found at {config_path}; relying on env vars.")
        return {}
    except Exception as e:
        print(f"[WARN] Failed to read config: {e}")
        return {}


def load_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_text(text: str, max_chars: int) -> List[str]:
    # Simple sentence-aware chunking; mirrors processing._chunk_text_for_tts
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return [normalized]
    seps = ['. ', '? ', '! ', '\n']
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    i = 0
    n = len(normalized)
    while i < n:
        next_boundary = None
        for sep in seps:
            idx = normalized.find(sep, i)
            if idx != -1 and (next_boundary is None or idx < next_boundary):
                next_boundary = idx + len(sep)
        if next_boundary is None:
            next_boundary = n
        seg = normalized[i:next_boundary]
        if cur_len + len(seg) > max_chars and cur:
            chunks.append("".join(cur).strip())
            cur = []
            cur_len = 0
        cur.append(seg)
        cur_len += len(seg)
        i = next_boundary
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c]


def synthesize_chunk(text: str, api_key: str, voice_id: str, model_id: str,
                     stability: float, clarity: float, timeout: int) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(stability),
            "similarity_boost": float(clarity),
        },
    }
    try:
        with requests.post(url, headers=headers, json=payload, timeout=timeout, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                detail = None
                try:
                    detail = r.json()
                except Exception:
                    detail = r.text
                raise RuntimeError(f"HTTP {r.status_code} from ElevenLabs. Detail: {str(detail)[:1000]}")
            out: List[bytes] = []
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    out.append(chunk)
            return b"".join(out)
    except Exception as e:
        snippet = text[:300].replace('\n', ' ')
        raise RuntimeError(f"Chunk synthesis failed: {e}\nPayload size={len(text)} chars; snippet='{snippet}...'\n")


def main():
    parser = argparse.ArgumentParser(description="Debug ElevenLabs synthesis with local text file")
    parser.add_argument('--text', default=DEFAULT_TEXT_PATH, help='Path to input text file')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Path to config.json')
    parser.add_argument('--voice-id', default=None, help='Override voice_id')
    parser.add_argument('--model-id', default=None, help='Override model_id (e.g., eleven_multilingual_v2)')
    parser.add_argument('--stability', type=float, default=None, help='Override stability [0..1]')
    parser.add_argument('--clarity', type=float, default=None, help='Override clarity (similarity_boost) [0..1]')
    parser.add_argument('--max-chars', type=int, default=4500, help='Max characters per chunk')
    parser.add_argument('--first-n-chunks', type=int, default=0, help='If >0, synthesize only the first N chunks')
    parser.add_argument('--timeout', type=int, default=180, help='HTTP timeout seconds')
    parser.add_argument('--output', default=os.path.join(PROJECT_DIR, 'debug_output.mp3'), help='Output MP3 path')
    args = parser.parse_args()

    cfg = read_config(args.config)
    api_key = os.environ.get('ELEVEN_API_KEY') or cfg.get('eleven_api_key') or ''
    voice_id = args.voice_id or cfg.get('voice_id') or ''
    model_id = args.model_id or cfg.get('eleven_model_id') or 'eleven_multilingual_v2'
    stability = args.stability if args.stability is not None else float(cfg.get('stability', 0.5))
    clarity = args.clarity if args.clarity is not None else float(cfg.get('clarity', 0.5))

    if not api_key:
        print("[ERROR] Missing ElevenLabs API key. Set ELEVEN_API_KEY or config.json: eleven_api_key")
        sys.exit(2)
    if not voice_id:
        print("[ERROR] Missing voice_id. Provide --voice-id or set in config.json via the app Settings.")
        sys.exit(2)

    print("[INFO] Input text:", args.text)
    print("[INFO] Output MP3:", args.output)
    print("[INFO] Voice ID:", voice_id)
    print("[INFO] Model ID:", model_id)
    print(f"[INFO] Stability={stability:.2f} Clarity={clarity:.2f} MaxChars={args.max_chars}")

    try:
        text = load_text(args.text)
    except Exception as e:
        print(f"[ERROR] Failed to read text: {e}")
        sys.exit(1)

    chunks = chunk_text(text, args.max_chars)
    total = len(chunks)
    if args.first_n_chunks and args.first_n_chunks > 0:
        chunks = chunks[:args.first_n_chunks]
        print(f"[INFO] Limiting to first {len(chunks)}/{total} chunks for debug")
    else:
        print(f"[INFO] Total chunks: {total}")

    audio_parts: List[bytes] = []
    for i, ch in enumerate(chunks, 1):
        print(f"[STEP] Synthesizing chunk {i}/{len(chunks)} (len={len(ch)} chars)")
        try:
            part = synthesize_chunk(ch, api_key, voice_id, model_id, stability, clarity, args.timeout)
            audio_parts.append(part)
            print(f"[OK] Chunk {i} received {len(part)} bytes")
        except Exception as e:
            print(f"[ERROR] Chunk {i} failed: {e}")
            print("[HINT] Common causes: invalid voice_id, wrong model for the voice, text too long, or characters not allowed.")
            sys.exit(3)

    data = b"".join(audio_parts)
    try:
        with open(args.output, 'wb') as f:
            f.write(data)
        print(f"[DONE] Wrote {len(data)} bytes to {args.output}")
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}")
        sys.exit(4)


if __name__ == '__main__':
    main()


