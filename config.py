import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Load environment variables from .env.local
def load_env_local():
    """Load environment variables from .env.local file."""
    env_paths = [
        Path(__file__).parent / ".env.local",
        Path(__file__).parent / ".env",
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, _, value = line.partition('=')
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key and value:
                                os.environ.setdefault(key, value)
            except Exception:
                pass

# Load env on import
load_env_local()


# =============================================================================
# Known Model Pricing (USD per million units)
# =============================================================================

# Default pricing for known models
# These are approximate/estimated prices - users can update in config
DEFAULT_MODEL_PRICING = {
    # Gemini text models (per million tokens)
    "gemini-2.0-flash": {"type": "text", "input": 0.10, "output": 0.40, "unit": "tokens"},
    "gemini-2.0-flash-lite": {"type": "text", "input": 0.075, "output": 0.30, "unit": "tokens"},
    "gemini-1.5-flash": {"type": "text", "input": 0.075, "output": 0.30, "unit": "tokens"},
    "gemini-1.5-pro": {"type": "text", "input": 1.25, "output": 5.00, "unit": "tokens"},
    "gemini-2.5-pro": {"type": "text", "input": 1.25, "output": 10.00, "unit": "tokens"},
    "gemini-2.5-flash": {"type": "text", "input": 0.15, "output": 0.60, "unit": "tokens"},
    
    # Gemini TTS models (per million characters)
    "gemini-2.5-flash-preview-tts": {"type": "tts", "rate": 0.80, "unit": "chars"},
    "gemini-2.0-flash-preview-tts": {"type": "tts", "rate": 0.80, "unit": "chars"},
    
    # Inworld TTS models (per million characters) - estimated
    "inworld-tts-1": {"type": "tts", "rate": 0.30, "unit": "chars"},
    "inworld-tts-1-hd": {"type": "tts", "rate": 0.50, "unit": "chars"},
}


# =============================================================================
# TTS Provider: Inworld AI
# =============================================================================

# Available Inworld TTS voices
# Fallback Inworld voices (used if API fetch fails)
INWORLD_TTS_VOICES_FALLBACK = {
    "Cove": "Male, American",
    "Dennis": "Male, American", 
    "Hazel": "Female, American",
    "Jett": "Male, American",
    "Lucia": "Female, American",
    "Miranda": "Female, British",
    "Sebastian": "Male, British",
}

# This will be populated dynamically from the API
INWORLD_TTS_VOICES = INWORLD_TTS_VOICES_FALLBACK.copy()

# Inworld TTS models
INWORLD_TTS_MODELS = {
    "inworld-tts-1": "Standard quality",
    "inworld-tts-1-max": "Maximum quality",
}


def fetch_inworld_voices(api_key: str) -> Dict[str, str]:
    """Fetch available voices from Inworld API.
    
    Args:
        api_key: Inworld API key (base64 encoded)
        
    Returns:
        Dict mapping voice ID to description
    """
    import requests
    
    try:
        headers = {
            "Authorization": f"Basic {api_key}",
        }
        params = {
            "filter": "language=en"
        }
        
        response = requests.get(
            "https://api.inworld.ai/tts/v1/voices",
            headers=headers,
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            voices = data.get("voices", [])
            
            result = {}
            for voice in voices:
                voice_id = voice.get("voiceId", "")
                display_name = voice.get("displayName", voice_id)
                description = voice.get("description", "")
                gender = voice.get("gender", "")
                accent = voice.get("accent", "")
                
                # Build a nice description
                desc_parts = []
                if gender:
                    desc_parts.append(gender)
                if accent:
                    desc_parts.append(accent)
                if description:
                    desc_parts.append(description)
                
                result[voice_id] = ", ".join(desc_parts) if desc_parts else display_name
            
            if result:
                print(f"[Config] Fetched {len(result)} Inworld voices from API")
                return result
        else:
            print(f"[Config] Failed to fetch Inworld voices: {response.status_code}")
            
    except Exception as e:
        print(f"[Config] Error fetching Inworld voices: {e}")
    
    return INWORLD_TTS_VOICES_FALLBACK.copy()


# =============================================================================
# TTS Provider: Gemini
# =============================================================================

# Available Gemini TTS voices with their characteristics
GEMINI_TTS_VOICES = {
    "Zephyr": "Bright",
    "Puck": "Upbeat",
    "Charon": "Informative",
    "Kore": "Firm",
    "Fenrir": "Excitable",
    "Leda": "Youthful",
    "Orus": "Firm",
    "Aoede": "Breezy",
    "Callirrhoe": "Easy-going",
    "Autonoe": "Bright",
    "Enceladus": "Breathy",
    "Iapetus": "Clear",
    "Umbriel": "Easy-going",
    "Algieba": "Smooth",
    "Despina": "Smooth",
    "Erinome": "Clear",
    "Algenib": "Gravelly",
    "Rasalgethi": "Informative",
    "Laomedeia": "Upbeat",
    "Achernar": "Soft",
    "Alnilam": "Firm",
    "Schedar": "Even",
    "Gacrux": "Mature",
    "Pulcherrima": "Forward",
    "Achird": "Friendly",
    "Zubenelgenubi": "Casual",
    "Vindemiatrix": "Gentle",
    "Sadachbia": "Lively",
    "Sadaltager": "Knowledgeable",
    "Sulafat": "Warm",
}


# =============================================================================
# TTS Providers Enum
# =============================================================================

TTS_PROVIDERS = {
    "inworld": "Inworld AI",
    "gemini": "Gemini TTS",
}


# =============================================================================
# Pricing Cache Manager
# =============================================================================

class PricingCache:
    """Manages cached pricing information for models."""
    
    CACHE_FILE = ".pricing_cache.json"
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.cache_path = os.path.join(self.base_dir, self.CACHE_FILE)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load pricing cache from disk."""
        # Start with defaults
        self._cache = dict(DEFAULT_MODEL_PRICING)
        
        # Overlay with user-saved prices
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    saved = json.load(f)
                    self._cache.update(saved.get("models", {}))
            except Exception:
                pass
    
    def _save_cache(self):
        """Save pricing cache to disk."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "models": {k: v for k, v in self._cache.items() 
                          if k not in DEFAULT_MODEL_PRICING or v != DEFAULT_MODEL_PRICING.get(k)}
            }
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def get_pricing(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get pricing for a model."""
        # Normalize model ID (handle variations)
        normalized = self._normalize_model_id(model_id)
        return self._cache.get(normalized)
    
    def set_pricing(self, model_id: str, pricing: Dict[str, Any]):
        """Set/update pricing for a model."""
        normalized = self._normalize_model_id(model_id)
        self._cache[normalized] = pricing
        self._save_cache()
    
    def _normalize_model_id(self, model_id: str) -> str:
        """Normalize model ID for consistent lookup."""
        # Remove common prefixes/suffixes
        normalized = model_id.lower().strip()
        return normalized
    
    def get_text_model_cost(self, model_id: str, tokens: int) -> Tuple[float, bool]:
        """Get cost for text model usage.
        
        Returns (cost, is_estimated) where is_estimated=True if pricing was guessed.
        """
        pricing = self.get_pricing(model_id)
        
        if pricing and pricing.get("type") == "text":
            # Use input price for estimation (output is variable)
            rate = pricing.get("input", 0.10)
            return (tokens / 1_000_000.0) * rate, False
        
        # Unknown model - estimate based on name patterns
        rate = self._guess_text_rate(model_id)
        return (tokens / 1_000_000.0) * rate, True
    
    def get_tts_model_cost(self, model_id: str, chars: int) -> Tuple[float, bool]:
        """Get cost for TTS model usage.
        
        Returns (cost, is_estimated) where is_estimated=True if pricing was guessed.
        """
        pricing = self.get_pricing(model_id)
        
        if pricing and pricing.get("type") == "tts":
            rate = pricing.get("rate", 0.50)
            return (chars / 1_000_000.0) * rate, False
        
        # Unknown model - estimate based on name patterns
        rate = self._guess_tts_rate(model_id)
        return (chars / 1_000_000.0) * rate, True
    
    def _guess_text_rate(self, model_id: str) -> float:
        """Guess text model rate based on name patterns."""
        model_lower = model_id.lower()
        
        if "pro" in model_lower or "opus" in model_lower:
            return 1.25  # Premium models
        elif "flash" in model_lower or "lite" in model_lower:
            return 0.10  # Fast/cheap models
        elif "gpt-4" in model_lower:
            return 5.00  # GPT-4 class
        elif "gpt-3" in model_lower or "turbo" in model_lower:
            return 0.50  # GPT-3.5 class
        else:
            return 0.50  # Default estimate
    
    def _guess_tts_rate(self, model_id: str) -> float:
        """Guess TTS model rate based on name patterns."""
        model_lower = model_id.lower()
        
        if "hd" in model_lower or "pro" in model_lower or "premium" in model_lower:
            return 1.00  # High quality
        elif "inworld" in model_lower:
            return 0.30  # Inworld models
        elif "gemini" in model_lower:
            return 0.80  # Gemini TTS
        elif "elevenlabs" in model_lower:
            return 1.50  # ElevenLabs (premium)
        else:
            return 0.50  # Default estimate
    
    def lookup_and_cache_pricing(self, model_id: str, model_type: str = "tts") -> Optional[Dict[str, Any]]:
        """Look up pricing for an unknown model and cache it.
        
        This is a placeholder for web lookup - currently just uses heuristics.
        In a full implementation, this could query pricing APIs or scrape docs.
        """
        # Check if we already have it
        existing = self.get_pricing(model_id)
        if existing:
            return existing
        
        # Generate estimated pricing based on heuristics
        if model_type == "tts":
            rate = self._guess_tts_rate(model_id)
            pricing = {
                "type": "tts",
                "rate": rate,
                "unit": "chars",
                "estimated": True,
                "added_date": datetime.now().isoformat(),
            }
        else:
            input_rate = self._guess_text_rate(model_id)
            pricing = {
                "type": "text",
                "input": input_rate,
                "output": input_rate * 4,  # Output typically 4x input
                "unit": "tokens",
                "estimated": True,
                "added_date": datetime.now().isoformat(),
            }
        
        # Cache it
        self.set_pricing(model_id, pricing)
        return pricing


# Global pricing cache instance
_pricing_cache: Optional[PricingCache] = None

def get_pricing_cache() -> PricingCache:
    """Get the global pricing cache instance."""
    global _pricing_cache
    if _pricing_cache is None:
        _pricing_cache = PricingCache()
    return _pricing_cache


# =============================================================================
# Config Manager
# =============================================================================

class ConfigManager:
    def __init__(self, path: str):
        self.path = path
        self._pricing_cache = get_pricing_cache()

    def default_config(self) -> Dict[str, Any]:
        return {
            "gemini_api_key": "",
            # Gemini model for text processing (summarization/cleaning)
            "model_name": "gemini-2.0-flash",
            
            # TTS Provider: "inworld" (default) or "gemini"
            "tts_provider": "inworld",
            # Auto-fallback to Gemini if Inworld quota exceeded
            "tts_fallback_enabled": True,
            
            # Inworld TTS settings
            "inworld_api_key": os.environ.get("INWORLD_API_KEY", ""),
            "inworld_voice_id": "Ashley",
            "inworld_model_id": "inworld-tts-1",
            
            # Gemini TTS settings (fallback)
            "tts_model_name": "gemini-2.5-flash-preview-tts",
            "tts_voice_name": "Kore",
            "tts_style_prompt": "",
            
            # Citation handling
            "citation_style": "Ignore",
            # Conversion mode: 'Summarized' (Gemini-structured) or 'Verbatim' (raw text)
            "conversion_mode": "Summarized",
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
        
        # Inworld API key: use config.json if set, otherwise fall back to env
        # This allows UI to override env variable
        config_inworld_key = data.get("inworld_api_key", "")
        env_inworld_key = os.environ.get("INWORLD_API_KEY", "")
        
        if not config_inworld_key and env_inworld_key:
            # Config is empty, use env as fallback
            data["inworld_api_key"] = env_inworld_key
        
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

    def estimate_gemini_cost(self, tokens: int, model_name: str = None) -> Tuple[float, str]:
        """Estimate Gemini text processing cost.
        
        Returns (cost, model_name) tuple.
        """
        try:
            data = self.load()
            model = model_name or data.get("model_name", "gemini-2.0-flash")
            cost, is_estimated = self._pricing_cache.get_text_model_cost(model, tokens)
            
            # If unknown model, look it up and cache
            if is_estimated:
                self._pricing_cache.lookup_and_cache_pricing(model, "text")
            
            return cost, model
        except Exception:
            return (tokens / 1_000_000.0) * 0.10, "unknown"

    def estimate_tts_cost(self, num_chars: int, provider: str = None, model: str = None) -> Tuple[float, str, str]:
        """Estimate TTS cost based on character count, provider, and model.
        
        Returns (cost, provider, model) tuple.
        """
        try:
            data = self.load()
            tts_provider = provider or data.get("tts_provider", "inworld")
            
            if tts_provider == "inworld":
                tts_model = model or data.get("inworld_model_id", "inworld-tts-1")
            else:
                tts_model = model or data.get("tts_model_name", "gemini-2.5-flash-preview-tts")
            
            cost, is_estimated = self._pricing_cache.get_tts_model_cost(tts_model, num_chars)
            
            # If unknown model, look it up and cache
            if is_estimated:
                self._pricing_cache.lookup_and_cache_pricing(tts_model, "tts")
            
            return cost, tts_provider, tts_model
        except Exception:
            return (max(0, int(num_chars)) / 1_000_000.0) * 0.50, "unknown", "unknown"

    def get_model_pricing(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get pricing info for a specific model."""
        return self._pricing_cache.get_pricing(model_id)

    def set_model_pricing(self, model_id: str, pricing: Dict[str, Any]):
        """Set pricing for a model (persisted to cache)."""
        self._pricing_cache.set_pricing(model_id, pricing)

    @staticmethod
    def get_available_voices(provider: str = "inworld") -> List[Dict[str, str]]:
        """Return list of available TTS voices for the specified provider."""
        if provider == "inworld":
            return [
                {"name": name, "style": style}
                for name, style in INWORLD_TTS_VOICES.items()
            ]
        else:
            return [
                {"name": name, "style": style}
                for name, style in GEMINI_TTS_VOICES.items()
            ]
    
    @staticmethod
    def get_inworld_models() -> List[Dict[str, str]]:
        """Return list of available Inworld TTS models."""
        return [
            {"id": model_id, "description": desc}
            for model_id, desc in INWORLD_TTS_MODELS.items()
        ]
