import json
import os
from typing import Any, Dict, List

import requests


class ConfigManager:
    def __init__(self, path: str):
        self.path = path

    def default_config(self) -> Dict[str, Any]:
        return {
            "gemini_api_key": "",
            "eleven_api_key": "",
            "model_name": "gemini-2.5-pro",
            "eleven_model_id": "eleven_multilingual_v2",
            "voice_id": "",
            "stability": 0.5,
            "clarity": 0.5,  # mapped to ElevenLabs similarity_boost
            "citation_style": "Ignore",
            # Conversion mode: 'Summarized' (Gemini-structured) or 'Verbatim' (raw text)
            "conversion_mode": "Summarized",
            "usd_per_million_tokens": 5.0,
            # Audio cost estimates (per million characters)
            "usd_per_million_chars_google_tts": 16.0,
            "usd_per_million_chars_elevenlabs": 15.0,
            # Audio provider selection: 'google_tts' (default) or 'elevenlabs'
            "audio_provider": "google_tts",
            # Google TTS (Studio) settings
            "google_tts_api_key": "",
            "gtts_voice_name": "en-US-Studio-O",
            "gtts_language_code": "en-US",
            "gtts_speaking_rate": 1.0,
            "gtts_pitch": 0.0,
            # Max bytes per chunk for Google TTS Studio REST (limit ~900 bytes)
            "gtts_max_bytes": 800,
        }

    def ensure_config(self) -> None:
        if not os.path.exists(self.path):
            data = self.default_config()
            self._write_file(data)

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure any new defaults exist
        defaults = self.default_config()
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data

    def save(self, updates: Dict[str, Any]) -> None:
        current = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                current = json.load(f)
        current.update(updates)
        self._write_file(current)

    def _write_file(self, data: Dict[str, Any]) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def estimate_gemini_cost(self, tokens: int) -> float:
        try:
            data = self.load()
            rate = float(data.get("usd_per_million_tokens", 5.0))
        except Exception:
            rate = 5.0
        return (tokens / 1_000_000.0) * rate

    def estimate_tts_cost(self, provider: str, num_chars: int) -> float:
        """Estimate text-to-speech cost based on configured per-million-char rates.

        This is an approximation; real billing may vary by provider, voice, or region.
        """
        try:
            data = self.load()
        except Exception:
            data = self.default_config()

        prov = (provider or "google_tts").strip().lower()
        if prov == "elevenlabs":
            rate = float(data.get("usd_per_million_chars_elevenlabs", 15.0))
        else:
            # Default to Google TTS Studio
            rate = float(data.get("usd_per_million_chars_google_tts", 16.0))

        return (max(0, int(num_chars)) / 1_000_000.0) * rate

    def fetch_elevenlabs_voices(self, api_key: str) -> List[Dict[str, Any]]:
        if not api_key:
            raise ValueError("Eleven Labs API key is required")
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": api_key}
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json() or {}
            items = payload.get("voices", [])
            # Normalize response
            voices = []
            for it in items:
                voices.append({
                    "voice_id": it.get("voice_id") or it.get("voice_id"),
                    "name": it.get("name", "")
                })
            return voices
        except requests.HTTPError as e:
            raise RuntimeError(f"Failed to fetch voices: {e}")
        except Exception as e:
            raise RuntimeError(f"Voice fetch error: {e}")
