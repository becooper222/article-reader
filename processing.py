import base64
import io
import os
from typing import Callable, Dict, List, Optional, Tuple

import requests
import json as _json  # for error messages when Google TTS returns JSON without audio
from base64 import b64decode as _b64decode

# PDF parsing
import pdfplumber

# Optional image handling (best-effort)
from PIL import Image

# Gemini
import google.generativeai as genai

# Token estimation
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # fallback later


def _extract_images_from_page(page) -> List[Tuple[bytes, str]]:
    images: List[Tuple[bytes, str]] = []
    # Best-effort extraction: pdfplumber exposes page.images metadata; extraction is non-trivial.
    # We'll attempt rasterization of the page and crop per image bbox as a fallback.
    try:
        for im in page.images:
            # im has bbox: x0, top, x1, bottom
            x0, top, x1, bottom = im.get("x0"), im.get("top"), im.get("x1"), im.get("bottom")
            if None in (x0, top, x1, bottom):
                continue
            # Render page to bitmap and crop region
            try:
                pil_page = page.to_image(resolution=200).original  # PIL Image
                crop = pil_page.crop((x0, top, x1, bottom))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                images.append((buf.getvalue(), "image/png"))
            except Exception:
                continue
    except Exception:
        # If page.images not reliable, ignore silently
        pass
    return images


def parse_pdf(pdf_path: str, max_pages: int = 20) -> Tuple[str, List[Tuple[bytes, str]]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found")

    text_parts: List[str] = []
    all_images: List[Tuple[bytes, str]] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            if num_pages > max_pages:
                raise ValueError(f"PDF exceeds {max_pages} page limit (got {num_pages})")
            for page in pdf.pages:
                try:
                    # Extract text with layout awareness
                    txt = page.extract_text(layout=True) or page.extract_text() or ""
                    text_parts.append(txt.strip())
                except Exception:
                    text_parts.append("")
                # Extract images best-effort
                all_images.extend(_extract_images_from_page(page))
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")

    combined_text = "\n\n".join([t for t in text_parts if t])
    return combined_text, all_images


def estimate_tokens(pdf_path: str) -> int:
    # Lightweight estimation: read text quickly without images
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found")
    text_parts: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) > 20:
                raise ValueError("PDF exceeds 20 page limit")
            for page in pdf.pages:
                try:
                    txt = page.extract_text(layout=True) or page.extract_text() or ""
                except Exception:
                    txt = ""
                text_parts.append(txt)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF for estimation: {e}")
    text = "\n".join(text_parts)

    # Prefer tiktoken if available; otherwise approximate
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic: ~4 chars/token
    return max(1, int(len(text) / 4))


def _build_gemini_prompt(citation_style: str) -> str:
    cite = ("omit citations entirely" if citation_style == "Ignore" else
            "replace in-text citations with brief phrases like 'as a citation notes' without details")
    return (
        "You will receive the contents of an academic paper. Convert it into a natural, single-voice narration script suitable for direct text-to-speech.\n"
        "Strict formatting and style requirements (follow exactly):\n"
        "- Output plain text only. Do NOT include any markdown (no ###, **, lists), stage directions, brackets, or labels like 'Narrator:'.\n"
        "- Begin the script with ONE concise line that states the paper title and the authors, extracted from the provided content.\n"
        "  Format exactly: Title. By Author One, Author Two, Author Three.\n"
        "  Do not add extra words like 'Paper:' or 'Authors:'.\n"
        "- Structure the script as a sequence of sections. For each section, first say the section title as a short, spoken-friendly header, then immediately read the content of that section in clear, natural language.\n"
        "- To create a natural pause after each section title, put the title on its own line ending with a period, then insert a blank line before the section content.\n"
        "- Keep section titles concise (spoken-friendly); avoid decorative words or sound cues.\n"
        "- Do NOT include intro/outro segments, music cues, disclaimers, or any meta commentary.\n"
        "- Read ONLY the title and the content of each summarized section.\n"
        f"- For citations, {cite}.\n"
        "- Explain formulas in spoken language (e.g., E = mc^2 -> 'E equals m c squared').\n"
        "- When figures or tables are essential to understanding, briefly describe their key takeaway in plain speech.\n"
        "- Avoid reading URLs, reference lists, inline reference numbers, or footnotes.\n"
        "- Keep sentences conversational and fluid for listening; use punctuation to guide natural pauses.\n"
        "- Keep total length concise but comprehensive.\n"
    )


def _chunk_text_for_tts(text: str, max_chars: int = 4500) -> List[str]:
    """Split text into chunks under ElevenLabs limits, preferring sentence boundaries.

    ElevenLabs commonly enforces ~5k character limits; we keep a safety margin.
    """
    # Normalize whitespace a bit
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return [normalized]

    separators = ['. ', '? ', '! ', '\n']
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    i = 0
    while i < len(normalized):
        # Find next boundary
        next_boundary = None
        for sep in separators:
            idx = normalized.find(sep, i)
            if idx != -1 and (next_boundary is None or idx < next_boundary):
                next_boundary = idx + len(sep)
        if next_boundary is None:
            next_boundary = len(normalized)

        segment = normalized[i:next_boundary]
        if current_len + len(segment) > max_chars and current:
            chunks.append("".join(current).strip())
            current = []
            current_len = 0
        current.append(segment)
        current_len += len(segment)
        i = next_boundary

    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def _split_text_to_byte_chunks(text: str, max_bytes: int) -> List[str]:
    """Split text into chunks constrained by UTF-8 byte length.

    This mirrors the working helper in test_google_studio_tts.py, ensuring
    Studio REST input stays within ~900 byte limit (we keep safety margin).
    """
    if max_bytes <= 0:
        return [text]
    chunks: List[str] = []
    current_chars: List[str] = []
    current_size = 0
    for ch in text:
        enc = ch.encode('utf-8')
        if current_size + len(enc) > max_bytes:
            if current_chars:
                chunks.append(''.join(current_chars))
                current_chars = []
                current_size = 0
        current_chars.append(ch)
        current_size += len(enc)
    if current_chars:
        chunks.append(''.join(current_chars))
    return chunks


def synthesize_with_google_tts(text: str,
                               api_key: str,
                               voice_name: str,
                               language_code: str,
                               speaking_rate: float,
                               pitch: float,
                               max_bytes: int,
                               cancel_event,
                               log: Callable[[str], None]) -> Optional[bytes]:
    """Synthesize speech using Google Cloud TTS Studio via REST v1beta1.

    Splits input by UTF-8 bytes to satisfy Studio limits and concatenates MP3.
    """
    if not api_key:
        raise ValueError("Google TTS API key is missing. Set it in Settings.")

    if cancel_event.is_set():
        return None

    url = f'https://texttospeech.googleapis.com/v1beta1/text:synthesize?key={api_key}'

    # Prepare byte-constrained chunks
    chunks = _split_text_to_byte_chunks(" ".join(text.split()), max_bytes=max_bytes)
    total = len(chunks)
    audio_parts: List[bytes] = []

    for idx, chunk in enumerate(chunks, start=1):
        if cancel_event.is_set():
            return None
        log(f"Generating audio chunk {idx}/{total}...")
        body = {
            'input': {'text': chunk},
            'voice': {
                'name': voice_name,
                'languageCode': language_code,
            },
            'audioConfig': {
                'audioEncoding': 'MP3',
                'speakingRate': float(speaking_rate),
                'pitch': float(pitch),
            },
        }
        try:
            with requests.post(url, json=body, timeout=60) as r:
                r.raise_for_status()
                data = r.json() or {}
                audio_b64 = data.get('audioContent')
                if not audio_b64:
                    # Include short detail to aid debugging
                    raise RuntimeError(
                        f"No audioContent in response: {_json.dumps(data)[:500]}"
                    )
                audio_parts.append(_b64decode(audio_b64))
        except requests.HTTPError as e:
            detail = None
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text[:500] if getattr(e, 'response', None) is not None else str(e)
            raise RuntimeError(f"Google TTS error on chunk {idx}/{total}: {e} | detail: {detail}")
        except Exception as e:
            raise RuntimeError(f"Google TTS error on chunk {idx}/{total}: {e}")

    return b"".join(audio_parts)


def structure_with_gemini(text: str,
                           images: List[Tuple[bytes, str]],
                           api_key: str,
                           model_name: str,
                           citation_style: str,
                           cancel_event,
                           log: Callable[[str], None]) -> str:
    if not api_key:
        raise ValueError("Gemini API key is missing. Please set it in Settings.")
    if cancel_event.is_set():
        raise InterruptedError("Cancelled")

    genai.configure(api_key=api_key)
    prompt = _build_gemini_prompt(citation_style)

    # Build content parts: prompt + text + images
    contents: List = [
        {"role": "user", "parts": [
            {"text": prompt},
            {"text": text[:200000]}  # guard against overly large payloads
        ]}
    ]

    # Attach images (limit to a few to control cost)
    for idx, (img_bytes, mime) in enumerate(images[:6]):
        contents[0]["parts"].append({
            "inline_data": {"mime_type": mime or "image/png", "data": base64.b64encode(img_bytes).decode("utf-8")}
        })

    model = genai.GenerativeModel(model_name=model_name)
    log("Structuring content with Gemini...")
    if cancel_event.is_set():
        raise InterruptedError("Cancelled")

    try:
        resp = model.generate_content(contents)
        # SDK may stream or not; ensure we have text
        structured = getattr(resp, "text", None)
        if not structured:
            # Try candidates
            try:
                structured = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
            except Exception:
                structured = ""
        if not structured:
            raise RuntimeError("No content returned from Gemini")
        return structured.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")


def synthesize_with_elevenlabs(text: str,
                               api_key: str,
                               voice_id: str,
                               stability: float,
                               clarity: float,
                               cancel_event,
                               log: Callable[[str], None]) -> Optional[bytes]:
    if not api_key:
        raise ValueError("Eleven Labs API key is missing. Please set it in Settings.")
    if not voice_id:
        raise ValueError("Voice is not selected. Please choose a voice in Settings.")

    if cancel_event.is_set():
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "voice_settings": {
            "stability": float(stability),
            "similarity_boost": float(clarity),
        },
    }
    # Optional: include model_id from config if available
    try:
        # Lazy import to avoid circular
        from config import ConfigManager  # type: ignore
        # Not using a manager instance here; the worker will pass config via generate_audio_only
    except Exception:
        ConfigManager = None  # type: ignore

    log("Generating audio...")
    if cancel_event.is_set():
        return None

    try:
        with requests.post(url, headers=headers, json=payload, timeout=120, stream=True) as r:
            r.raise_for_status()
            chunks: List[bytes] = []
            for chunk in r.iter_content(chunk_size=8192):
                if cancel_event.is_set():
                    return None
                if chunk:
                    chunks.append(chunk)
            return b"".join(chunks)
    except requests.HTTPError as e:
        try:
            detail = e.response.text[:500] if e.response is not None else str(e)
        except Exception:
            detail = str(e)
        raise RuntimeError(f"Eleven Labs error: {e}. Details: {detail}")
    except Exception as e:
        raise RuntimeError(f"Audio generation failed: {e}")


class ConversionWorker:
    @staticmethod
    def run(pdf_path: str,
            citation_style: str,
            config_mgr,
            cancel_event,
            done_callback: Callable[[str, Optional[str]], None]) -> None:
        def send(event: str, payload: Optional[str] = None):
            try:
                done_callback(event, payload)
            except Exception:
                pass

        try:
            send("status", "Parsing PDF...")
            text, images = parse_pdf(pdf_path)
            if cancel_event.is_set():
                send("cancelled")
                return

            cfg = config_mgr.load()
            api_key = cfg.get("gemini_api_key", "")
            model_name = cfg.get("model_name", "gemini-2.5-pro")

            script = structure_with_gemini(
                text=text,
                images=images,
                api_key=api_key,
                model_name=model_name,
                citation_style=citation_style,
                cancel_event=cancel_event,
                log=lambda m: send("status", m),
            )
            if cancel_event.is_set():
                send("cancelled")
                return

            send("script", script)
            send("done")
        except InterruptedError:
            send("cancelled")
        except Exception as e:
            send("error", str(e))

    @staticmethod
    def generate_audio_only(text: str,
                             config_mgr,
                             cancel_event,
                             log: Callable[[str], None]) -> Optional[bytes]:
        cfg = config_mgr.load()
        provider = (cfg.get("audio_provider") or "google_tts").strip().lower()

        if provider == "google_tts":
            api_key = cfg.get("google_tts_api_key", "")
            voice_name = cfg.get("gtts_voice_name", "en-US-Studio-O")
            language_code = cfg.get("gtts_language_code", "en-US")
            speaking_rate = float(cfg.get("gtts_speaking_rate", 1.0))
            pitch = float(cfg.get("gtts_pitch", 0.0))
            max_bytes = int(cfg.get("gtts_max_bytes", 800))
            return synthesize_with_google_tts(
                text=text,
                api_key=api_key,
                voice_name=voice_name,
                language_code=language_code,
                speaking_rate=speaking_rate,
                pitch=pitch,
                max_bytes=max_bytes,
                cancel_event=cancel_event,
                log=log,
            )

        # If provider is not google_tts, enforce Google TTS anyway
        api_key = cfg.get("google_tts_api_key", "")
        voice_name = cfg.get("gtts_voice_name", "en-US-Studio-O")
        language_code = cfg.get("gtts_language_code", "en-US")
        speaking_rate = float(cfg.get("gtts_speaking_rate", 1.0))
        pitch = float(cfg.get("gtts_pitch", 0.0))
        max_bytes = int(cfg.get("gtts_max_bytes", 800))
        return synthesize_with_google_tts(
            text=text,
            api_key=api_key,
            voice_name=voice_name,
            language_code=language_code,
            speaking_rate=speaking_rate,
            pitch=pitch,
            max_bytes=max_bytes,
            cancel_event=cancel_event,
            log=log,
        )
