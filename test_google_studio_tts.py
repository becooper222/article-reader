#!/usr/bin/env python3
"""
Quick script to synthesize the example text using Google TTS Studio voice (en-US-Studio-0)
Requires env var GOOGLE_TTS_API_KEY to be set.
"""
import os
import json
from typing import List
import requests
from base64 import b64decode
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(__file__)
DEFAULT_TEXT = os.path.join(PROJECT_DIR, 'summarized_text_ex.txt')
DEFAULT_OUT = os.path.join(PROJECT_DIR, 'gtts_output.mp3')


def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def split_text_to_byte_chunks(text: str, max_bytes: int) -> List[str]:
    if max_bytes <= 0:
        return [text]
    chunks: List[str] = []
    cur: List[str] = []
    size = 0
    for ch in text:
        enc = ch.encode('utf-8')
        if size + len(enc) > max_bytes:
            if cur:
                chunks.append(''.join(cur))
                cur = []
                size = 0
        cur.append(ch)
        size += len(enc)
    if cur:
        chunks.append(''.join(cur))
    return chunks


def synthesize_rest_chunk(api_key: str,
                          text: str,
                          voice_name: str,
                          language_code: str,
                          speaking_rate: float = 1.0,
                          pitch: float = 0.0,
                          audio_encoding: str = 'MP3') -> bytes:
    # Studio voices require v1beta1 API
    url = f'https://texttospeech.googleapis.com/v1beta1/text:synthesize?key={api_key}'
    body = {
        'input': {'text': text},
        'voice': {
            'name': voice_name,
            'languageCode': language_code,
        },
        'audioConfig': {
            'audioEncoding': audio_encoding,
            'speakingRate': speaking_rate,
            'pitch': pitch,
        },
    }
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    audio_b64 = data.get('audioContent')
    if not audio_b64:
        raise RuntimeError(f'No audioContent in response: {json.dumps(data)[:500]}')
    return b64decode(audio_b64)


def main():
    # Load .env.local if present so GOOGLE_TTS_API_KEY is available
    load_dotenv(os.path.join(PROJECT_DIR, '.env.local'))
    api_key = os.environ.get('GOOGLE_TTS_API_KEY')
    if not api_key:
        raise RuntimeError('Please set GOOGLE_TTS_API_KEY in your environment.')

    text = read_text(DEFAULT_TEXT)
    # Studio/REST input limit is 900 bytes; stay safely under
    chunks = split_text_to_byte_chunks(text, max_bytes=800)

    voice_name = 'en-US-Studio-O'
    language_code = 'en-US'

    with open(DEFAULT_OUT, 'wb') as f:
        for i, chunk in enumerate(chunks):
            try:
                audio = synthesize_rest_chunk(api_key=api_key,
                                              text=chunk,
                                              voice_name=voice_name,
                                              language_code=language_code)
                f.write(audio)
            except requests.HTTPError as e:
                detail = None
                try:
                    detail = e.response.json()
                except Exception:
                    detail = getattr(e, 'response', None).text if getattr(e, 'response', None) else str(e)
                raise RuntimeError(f'Chunk {i+1}/{len(chunks)} failed: {e} | detail: {detail}')
    print(f'[DONE] Wrote audio to {DEFAULT_OUT}')


if __name__ == '__main__':
    main()
