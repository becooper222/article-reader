import io
import os
import gc
import wave
import tempfile
import subprocess
import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any
from pathlib import Path

import requests
import psutil

# PDF parsing
import pdfplumber

# Optional image handling (best-effort)
from PIL import Image

# Gemini SDK (new google-genai library)
from google import genai
from google.genai import types

# Progress tracking
from tqdm import tqdm

# Token estimation
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # fallback later


# =============================================================================
# Memory Monitoring
# =============================================================================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def check_memory_pressure(threshold_mb: float = 500) -> bool:
    """Check if memory usage exceeds threshold."""
    return get_memory_usage_mb() > threshold_mb


def force_gc():
    """Force garbage collection to free memory."""
    gc.collect()


# =============================================================================
# PDF Cache for Avoiding Double Parsing
# =============================================================================

@dataclass
class ParsedPDFData:
    """Cached data from parsed PDF."""
    text: str
    token_count: int
    page_count: int
    image_locations: List[Dict[str, Any]] = field(default_factory=list)  # Page/bbox info, not actual images
    cache_key: str = ""


class PDFCache:
    """Simple in-memory cache for parsed PDF data to avoid double parsing."""
    
    _cache: Dict[str, ParsedPDFData] = {}
    _lock = threading.Lock()
    
    @classmethod
    def _make_key(cls, pdf_path: str) -> str:
        """Create cache key from path + modification time."""
        stat = os.stat(pdf_path)
        key_str = f"{pdf_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @classmethod
    def get(cls, pdf_path: str) -> Optional[ParsedPDFData]:
        """Get cached data if available."""
        with cls._lock:
            key = cls._make_key(pdf_path)
            return cls._cache.get(key)
    
    @classmethod
    def set(cls, pdf_path: str, data: ParsedPDFData) -> None:
        """Store parsed data in cache."""
        with cls._lock:
            key = cls._make_key(pdf_path)
            data.cache_key = key
            cls._cache[key] = data
            
            # Limit cache size (keep last 5 PDFs)
            if len(cls._cache) > 5:
                oldest = next(iter(cls._cache))
                del cls._cache[oldest]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all cached data."""
        with cls._lock:
            cls._cache.clear()


# =============================================================================
# Progress Callback Wrapper
# =============================================================================

class ProgressTracker:
    """Unified progress tracker that works with both tqdm and GUI callbacks."""
    
    def __init__(self, total: int, desc: str = "", 
                 callback: Optional[Callable[[int, int, str], None]] = None,
                 use_tqdm: bool = True):
        self.total = total
        self.desc = desc
        self.callback = callback
        self.current = 0
        self.use_tqdm = use_tqdm and callback is None
        self._tqdm = None
        
        if self.use_tqdm:
            self._tqdm = tqdm(total=total, desc=desc, unit="item", 
                             leave=False, dynamic_ncols=True)
    
    def update(self, n: int = 1, status: str = ""):
        """Update progress by n steps."""
        self.current += n
        
        if self._tqdm:
            self._tqdm.update(n)
            if status:
                self._tqdm.set_postfix_str(status)
        
        if self.callback:
            self.callback(self.current, self.total, status or self.desc)
    
    def set_description(self, desc: str):
        """Update the description."""
        self.desc = desc
        if self._tqdm:
            self._tqdm.set_description(desc)
    
    def close(self):
        """Close the progress tracker."""
        if self._tqdm:
            self._tqdm.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# Image Extraction (Lazy)
# =============================================================================

@dataclass
class ImageLocation:
    """Metadata about an image's location in the PDF."""
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, top, x1, bottom
    context_text: str = ""  # Surrounding text for context


def extract_image_locations(pdf_path: str, max_pages: int = 30) -> List[ImageLocation]:
    """Extract image location metadata without loading actual image bytes.
    
    This is a lightweight scan that identifies where images are located.
    """
    locations: List[ImageLocation] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages[:max_pages]):
            try:
                page_text = page.extract_text() or ""
                for im in page.images:
                    x0 = im.get("x0")
                    top = im.get("top")
                    x1 = im.get("x1")
                    bottom = im.get("bottom")
                    
                    if None in (x0, top, x1, bottom):
                        continue
                    
                    # Get context: text near the image
                    context = _extract_nearby_text(page, (x0, top, x1, bottom))
                    
                    locations.append(ImageLocation(
                        page_num=page_num,
                        bbox=(x0, top, x1, bottom),
                        context_text=context[:500] if context else ""
                    ))
            except Exception:
                continue
    
    return locations


def _extract_nearby_text(page, bbox: Tuple[float, float, float, float]) -> str:
    """Extract text near an image bounding box for context."""
    x0, top, x1, bottom = bbox
    
    # Expand bbox to capture nearby text (above and below)
    expanded_top = max(0, top - 50)
    expanded_bottom = bottom + 50
    
    try:
        # Extract text from expanded region
        cropped = page.within_bbox((0, expanded_top, page.width, expanded_bottom))
        return cropped.extract_text() or ""
    except Exception:
        return ""


def extract_single_image(pdf_path: str, location: ImageLocation, 
                         resolution: int = 150) -> Optional[Tuple[bytes, str]]:
    """Extract a single image from PDF on-demand (lazy loading).
    
    Args:
        pdf_path: Path to PDF file
        location: ImageLocation metadata
        resolution: DPI for rasterization (lower = less memory)
    
    Returns:
        Tuple of (image_bytes, mime_type) or None if extraction fails
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if location.page_num >= len(pdf.pages):
                return None
            
            page = pdf.pages[location.page_num]
            x0, top, x1, bottom = location.bbox
            
            # Render page at lower resolution to save memory
            pil_page = page.to_image(resolution=resolution).original
            crop = pil_page.crop((
                x0 * resolution / 72,  # Scale bbox to resolution
                top * resolution / 72,
                x1 * resolution / 72,
                bottom * resolution / 72
            ))
            
            # Convert to bytes
            buf = io.BytesIO()
            crop.save(buf, format="PNG", optimize=True)
            return (buf.getvalue(), "image/png")
            
    except Exception:
        return None
    finally:
        force_gc()  # Clean up after image processing


def extract_images_batch(pdf_path: str, locations: List[ImageLocation],
                         max_images: int = 6,
                         progress_callback: Optional[Callable] = None) -> List[Tuple[bytes, str]]:
    """Extract multiple images with progress tracking and memory management.
    
    Limits extraction to max_images to control memory usage.
    """
    images: List[Tuple[bytes, str]] = []
    to_extract = locations[:max_images]
    
    with ProgressTracker(len(to_extract), "Extracting images", 
                        callback=progress_callback) as progress:
        for loc in to_extract:
            if check_memory_pressure(threshold_mb=800):
                force_gc()
                if check_memory_pressure(threshold_mb=800):
                    progress.set_description("Memory limit - stopping image extraction")
                    break
            
            result = extract_single_image(pdf_path, loc)
            if result:
                images.append(result)
            
            progress.update(1, f"Page {loc.page_num + 1}")
    
    return images


# =============================================================================
# PDF Parsing with Caching
# =============================================================================

DEFAULT_MAX_PDF_PAGES = 20  # Default maximum pages allowed for conversion


def get_pdf_page_count(pdf_path: str) -> int:
    """Quickly get the page count of a PDF without full parsing.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Number of pages in the PDF
        
    Raises:
        FileNotFoundError: If PDF doesn't exist
        RuntimeError: If PDF can't be opened
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")


def parse_pdf(pdf_path: str, max_pages: int = DEFAULT_MAX_PDF_PAGES,
              progress_callback: Optional[Callable] = None) -> Tuple[str, List[ImageLocation]]:
    """Parse PDF and cache results. Returns text and image locations (not actual images).
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to process
        progress_callback: Optional callback(current, total, status)
    
    Returns:
        Tuple of (combined_text, list_of_image_locations)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found")
    
    # Check cache first
    cached = PDFCache.get(pdf_path)
    if cached:
        # Return cached text and re-extract image locations (lightweight)
        locations = extract_image_locations(pdf_path, max_pages)
        return cached.text, locations
    
    text_parts: List[str] = []
    image_locations: List[ImageLocation] = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            if num_pages > max_pages:
                raise ValueError(f"PDF exceeds {max_pages}-page limit (has {num_pages} pages)")
            
            with ProgressTracker(num_pages, "Parsing PDF", 
                                callback=progress_callback) as progress:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text with layout awareness
                        txt = page.extract_text(layout=True) or page.extract_text() or ""
                        text_parts.append(txt.strip())
                        
                        # Record image locations (don't extract actual bytes yet)
                        for im in page.images:
                            x0 = im.get("x0")
                            top = im.get("top")
                            x1 = im.get("x1")
                            bottom = im.get("bottom")
                            
                            if None in (x0, top, x1, bottom):
                                continue
                            
                            context = _extract_nearby_text(page, (x0, top, x1, bottom))
                            image_locations.append(ImageLocation(
                                page_num=page_num,
                                bbox=(x0, top, x1, bottom),
                                context_text=context[:500]
                            ))
                    except Exception:
                        text_parts.append("")
                    
                    progress.update(1, f"Page {page_num + 1}/{num_pages}")
                    
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")
    
    combined_text = "\n\n".join([t for t in text_parts if t])
    
    # Estimate tokens
    token_count = _estimate_tokens_from_text(combined_text)
    
    # Cache the results
    PDFCache.set(pdf_path, ParsedPDFData(
        text=combined_text,
        token_count=token_count,
        page_count=len(text_parts),
        image_locations=[{
            "page": loc.page_num,
            "bbox": loc.bbox,
            "context": loc.context_text
        } for loc in image_locations]
    ))
    
    return combined_text, image_locations


def _estimate_tokens_from_text(text: str) -> int:
    """Estimate token count from text."""
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic: ~4 chars/token
    return max(1, int(len(text) / 4))


def estimate_tokens(pdf_path: str, max_pages: int = DEFAULT_MAX_PDF_PAGES) -> int:
    """Estimate tokens using cache if available, otherwise parse.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to consider for estimation
        
    Returns:
        Estimated token count
        
    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If PDF exceeds max_pages limit
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found")
    
    # Check page count first
    page_count = get_pdf_page_count(pdf_path)
    if page_count > max_pages:
        raise ValueError(f"PDF has {page_count} pages, exceeds {max_pages}-page limit")
    
    # Check cache first
    cached = PDFCache.get(pdf_path)
    if cached:
        return cached.token_count
    
    # Parse and cache
    text, _ = parse_pdf(pdf_path, max_pages=max_pages)
    
    # Now it should be in cache
    cached = PDFCache.get(pdf_path)
    if cached:
        return cached.token_count
    
    # Fallback
    return _estimate_tokens_from_text(text)


# =============================================================================
# Gemini Prompts
# =============================================================================

def _build_gemini_prompt(citation_style: str) -> str:
    cite = ("omit citations entirely" if citation_style == "Ignore" else
            "replace in-text citations with brief phrases like 'as a citation notes' without details")
    return (
        "You will receive the contents of an academic paper. Convert it into a natural, single-voice narration script suitable for direct text-to-speech.\n"
        "\n"
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
        "\n"
        "TTS Optimization (important for natural speech synthesis):\n"
        "- Use punctuation strategically: exclamation points (!) make the voice more emphatic; ellipsis (...) or em-dashes (—) create natural pauses.\n"
        "- Use asterisks (*word*) to emphasize key terms or findings that should be stressed when spoken.\n"
        "- Normalize complex values for speech:\n"
        "  - Phone numbers: write out digits separated by commas (e.g., 'one two three, four five six')\n"
        "  - Dates: write as spoken (e.g., 'May sixth, twenty twenty-five' not '5/6/2025')\n"
        "  - Times: write as spoken (e.g., 'twelve fifty-five PM' not '12:55 PM')\n"
        "  - Money: write out (e.g., 'five thousand dollars' not '$5,000')\n"
        "  - Percentages: write out (e.g., 'forty-five percent' not '45%')\n"
        "  - Mathematical expressions: write as spoken (e.g., 'two plus two equals four')\n"
        "- End every sentence with proper punctuation (period, question mark, or exclamation point).\n"
        "- Keep sentences moderate in length for natural breathing pauses.\n"
    )


def _build_gemini_cleanup_prompt(citation_style: str, include_image_descriptions: bool = False) -> str:
    cite = ("omit citations entirely" if citation_style == "Ignore" else
            "replace in-text citations with brief phrases like 'as a citation notes' without details")
    
    image_instruction = ""
    if include_image_descriptions:
        image_instruction = (
            "- IMAGE DESCRIPTIONS: You will also receive descriptions of figures and plots from the paper. "
            "Integrate these descriptions naturally into the text at appropriate locations. "
            "Introduce each with a phrase like 'The figure shows...' or 'As illustrated in the accompanying plot...' "
            "Keep descriptions concise but informative.\n"
        )
    
    return (
        "You will receive raw text extracted from a research PDF. Clean it for direct text-to-speech without changing its meaning.\n"
        "\n"
        "Strict requirements (follow exactly):\n"
        "- DO NOT summarize or reorder content. Preserve the original information and section order.\n"
        "- Output plain text only. No markdown, labels, or stage directions.\n"
        "- Merge broken lines and hyphenations across line breaks; form natural sentences and paragraphs.\n"
        "- Remove page headers/footers, page numbers, and repeated boilerplate.\n"
        "- Omit reference lists, footnotes, and URLs.\n"
        f"- For citations, {cite}.\n"
        f"{image_instruction}"
        "- Expand symbols and formulas into spoken-friendly text when necessary, without altering intent.\n"
        "- Keep the final text conversational and ready for voice synthesis.\n"
        "\n"
        "TTS Optimization (important for natural speech synthesis):\n"
        "- Use punctuation strategically: exclamation points (!) for emphasis; ellipsis (...) or em-dashes (—) for natural pauses.\n"
        "- Use asterisks (*word*) to emphasize key terms that should be stressed when spoken.\n"
        "- Normalize complex values for speech:\n"
        "  - Phone numbers: write out digits (e.g., 'one two three, four five six')\n"
        "  - Dates: write as spoken (e.g., 'May sixth, twenty twenty-five')\n"
        "  - Times: write as spoken (e.g., 'twelve fifty-five PM')\n"
        "  - Money: write out (e.g., 'five thousand dollars')\n"
        "  - Percentages: write out (e.g., 'forty-five percent')\n"
        "  - Mathematical expressions: write as spoken (e.g., 'two plus two equals four')\n"
        "- End every sentence with proper punctuation.\n"
        "- Keep sentences moderate in length for natural breathing pauses.\n"
    )


def _build_image_description_prompt() -> str:
    """Prompt for describing images/figures/plots."""
    return (
        "Describe this figure, chart, or plot from an academic paper. "
        "Provide a clear, spoken-friendly description that captures:\n"
        "1. What type of visualization it is (bar chart, scatter plot, diagram, etc.)\n"
        "2. The key data or relationships shown\n"
        "3. The main takeaway or finding\n"
        "Keep the description concise (2-4 sentences) and suitable for audio narration. "
        "Do not use markdown or bullet points. Write in natural, flowing prose."
    )


# =============================================================================
# Image Description for Verbatim Mode
# =============================================================================

def describe_images_for_verbatim(pdf_path: str, 
                                  image_locations: List[ImageLocation],
                                  api_key: str,
                                  model_name: str,
                                  cancel_event,
                                  log: Callable[[str], None],
                                  progress_callback: Optional[Callable] = None,
                                  max_images: int = 10) -> List[Dict[str, str]]:
    """Generate text descriptions for images to include in verbatim output.
    
    Args:
        pdf_path: Path to PDF
        image_locations: List of image location metadata
        api_key: Gemini API key
        model_name: Model for vision description
        cancel_event: Cancellation event
        log: Logging callback
        progress_callback: Progress callback
        max_images: Maximum images to process
    
    Returns:
        List of dicts with 'page', 'context', 'description' keys
    """
    if not image_locations:
        return []
    
    if not api_key:
        raise ValueError("Gemini API key is missing.")
    
    client = genai.Client(api_key=api_key)
    descriptions: List[Dict[str, str]] = []
    to_process = image_locations[:max_images]
    
    log(f"Generating descriptions for {len(to_process)} images...")
    
    with ProgressTracker(len(to_process), "Describing images",
                        callback=progress_callback) as progress:
        for i, loc in enumerate(to_process):
            if cancel_event.is_set():
                raise InterruptedError("Cancelled")
            
            # Extract this single image (lazy)
            image_data = extract_single_image(pdf_path, loc, resolution=150)
            
            if not image_data:
                progress.update(1, f"Skipped image {i+1}")
                continue
            
            img_bytes, mime_type = image_data
            
            try:
                # Build parts for vision request
                parts = [
                    types.Part.from_text(text=_build_image_description_prompt()),
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                ]
                
                # Add context if available
                if loc.context_text:
                    parts.insert(1, types.Part.from_text(
                        text=f"Context from the paper near this figure: {loc.context_text}"
                    ))
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=parts,
                )
                
                description = response.text.strip() if response.text else ""
                
                if description:
                    descriptions.append({
                        "page": loc.page_num + 1,
                        "context": loc.context_text[:100],
                        "description": description
                    })
                
            except Exception as e:
                log(f"Warning: Failed to describe image on page {loc.page_num + 1}: {e}")
            
            progress.update(1, f"Image {i+1}/{len(to_process)}")
            
            # Memory cleanup after each image
            del img_bytes
            force_gc()
    
    return descriptions


# =============================================================================
# Gemini Processing Functions
# =============================================================================

def structure_with_gemini(text: str,
                          pdf_path: str,
                          image_locations: List[ImageLocation],
                          api_key: str,
                          model_name: str,
                          citation_style: str,
                          cancel_event,
                          log: Callable[[str], None],
                          progress_callback: Optional[Callable] = None) -> str:
    """Use Gemini to structure/summarize PDF text into a narration script.
    
    Images are extracted lazily only when needed.
    """
    if not api_key:
        raise ValueError("Gemini API key is missing. Please set it in Settings.")
    if cancel_event.is_set():
        raise InterruptedError("Cancelled")

    client = genai.Client(api_key=api_key)
    prompt = _build_gemini_prompt(citation_style)

    # Build content parts: prompt + text
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=text[:200000])  # guard against overly large payloads
    ]

    # Lazy extract images (limit to 6 to control cost)
    if image_locations:
        log("Extracting key images for context...")
        images = extract_images_batch(pdf_path, image_locations, 
                                       max_images=6, 
                                       progress_callback=progress_callback)
        
        for img_bytes, mime in images:
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime or "image/png"))
        
        # Clean up image bytes
        del images
        force_gc()

    log("Structuring content with Gemini...")
    if cancel_event.is_set():
        raise InterruptedError("Cancelled")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=parts,
        )
        structured = response.text
        if not structured:
            raise RuntimeError("No content returned from Gemini")
        return structured.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")


def clean_with_gemini(text: str,
                      pdf_path: str,
                      image_locations: List[ImageLocation],
                      api_key: str,
                      model_name: str,
                      citation_style: str,
                      cancel_event,
                      log: Callable[[str], None],
                      progress_callback: Optional[Callable] = None) -> str:
    """Use Gemini to clean PDF text for verbatim TTS.
    
    In verbatim mode, images are described and their descriptions
    are integrated into the text.
    """
    if not api_key:
        raise ValueError("Gemini API key is missing. Please set it in Settings.")
    if cancel_event.is_set():
        raise InterruptedError("Cancelled")

    # Generate image descriptions for verbatim mode
    image_descriptions = []
    if image_locations:
        try:
            image_descriptions = describe_images_for_verbatim(
                pdf_path=pdf_path,
                image_locations=image_locations,
                api_key=api_key,
                model_name=model_name,
                cancel_event=cancel_event,
                log=log,
                progress_callback=progress_callback,
                max_images=10
            )
        except InterruptedError:
            raise
        except Exception as e:
            log(f"Warning: Could not generate image descriptions: {e}")

    client = genai.Client(api_key=api_key)
    prompt = _build_gemini_cleanup_prompt(citation_style, 
                                           include_image_descriptions=bool(image_descriptions))

    # Build the text with image description markers
    enhanced_text = text[:200000]
    if image_descriptions:
        desc_section = "\n\n--- IMAGE DESCRIPTIONS TO INTEGRATE ---\n"
        for desc in image_descriptions:
            desc_section += f"\n[Page {desc['page']}] {desc['description']}\n"
        desc_section += "\n--- END IMAGE DESCRIPTIONS ---\n\n"
        enhanced_text = desc_section + enhanced_text

    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=enhanced_text)
    ]

    log("Cleaning text with Gemini...")
    if cancel_event.is_set():
        raise InterruptedError("Cancelled")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=parts,
        )
        cleaned = response.text
        if not cleaned:
            raise RuntimeError("No content returned from Gemini")
        return cleaned.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")


# =============================================================================
# Streaming Audio Generation
# =============================================================================

def _pcm_to_mp3_streaming(pcm_chunks: List[bytes], output_path: str,
                           sample_rate: int = 24000, 
                           channels: int = 1, 
                           sample_width: int = 2) -> int:
    """Convert PCM audio chunks to MP3, writing incrementally to disk.
    
    Returns the final file size in bytes.
    """
    # Create temporary WAV file
    wav_path = output_path + ".tmp.wav"
    
    try:
        # Write PCM data as WAV incrementally
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            
            for chunk in pcm_chunks:
                wf.writeframes(chunk)
        
        # Convert WAV to MP3 using ffmpeg
        result = subprocess.run(
            [
                'ffmpeg', '-y',
                '-i', wav_path,
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                '-ar', str(sample_rate),
                '-ac', str(channels),
                output_path
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
        
        return os.path.getsize(output_path)
        
    finally:
        # Clean up temp WAV
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def _chunk_text_for_tts(text: str, max_chars: int = 4000) -> List[str]:
    """Split text into chunks for TTS processing.
    
    Gemini TTS has a 32k token context window, but we chunk for reliability.
    Inworld TTS has a strict 2000 character limit per request.
    
    This function guarantees all returned chunks are <= max_chars.
    """
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: List[str] = []
    
    def split_at_boundaries(text_segment: str, limit: int) -> List[str]:
        """Recursively split text at sentence/clause boundaries."""
        if len(text_segment) <= limit:
            return [text_segment] if text_segment.strip() else []
        
        # Try to find a good split point, preferring sentence endings
        separators = ['. ', '? ', '! ', '.\n', '!\n', '?\n', '; ', ', ', ' — ', ' - ', ' ']
        
        best_split = -1
        for sep in separators:
            # Look for separator in the first `limit` characters
            search_area = text_segment[:limit]
            # Find the LAST occurrence of separator in the valid range
            idx = search_area.rfind(sep)
            if idx > 0:  # Found a valid split point
                best_split = idx + len(sep)
                break
        
        if best_split <= 0:
            # No separator found - force split at word boundary
            # Find the last space before limit
            space_idx = text_segment[:limit].rfind(' ')
            if space_idx > 0:
                best_split = space_idx + 1
            else:
                # Absolute last resort: hard cut at limit
                best_split = limit
        
        # Split and recurse
        first_part = text_segment[:best_split].strip()
        remaining = text_segment[best_split:].strip()
        
        result = [first_part] if first_part else []
        if remaining:
            result.extend(split_at_boundaries(remaining, limit))
        
        return result
    
    # Split the normalized text
    raw_chunks = split_at_boundaries(normalized, max_chars)
    
    # Final validation - ensure no chunk exceeds limit (safety check)
    for chunk in raw_chunks:
        if len(chunk) <= max_chars:
            chunks.append(chunk)
        else:
            # This shouldn't happen, but handle it gracefully
            # Force split by words
            words = chunk.split(' ')
            current_chunk = []
            current_len = 0
            for word in words:
                word_len = len(word) + (1 if current_chunk else 0)
                if current_len + word_len > max_chars and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(word)
                current_len += word_len
            if current_chunk:
                chunks.append(' '.join(current_chunk))
    
    return [c for c in chunks if c.strip()]


# =============================================================================
# Text Normalization for TTS (Inworld Best Practices)
# =============================================================================

import re


def normalize_text_for_tts(text: str) -> str:
    """Normalize text for optimal TTS output following Inworld best practices.
    
    Handles:
    - Phone numbers: "(123)456-7891" -> "one two three, four five six, seven eight nine one"
    - Dates: "5/6/2025" -> "May sixth, twenty twenty-five"
    - Times: "12:55 PM" -> "twelve fifty-five PM"
    - Monetary values: "$5,342.29" -> "five thousand three hundred forty-two dollars and twenty-nine cents"
    - Emails: "test@example.com" -> "test at example dot com"
    - Mathematical symbols: "2+2=4" -> "two plus two equals four"
    - Percentages: "45%" -> "forty-five percent"
    - Ordinals: "1st, 2nd, 3rd" -> "first, second, third"
    """
    result = text
    
    # Number word mappings
    ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 
            'seventeen', 'eighteen', 'nineteen']
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    
    def number_to_words(n: int) -> str:
        """Convert a number to words."""
        if n < 0:
            return "negative " + number_to_words(-n)
        if n < 20:
            return ones[n]
        if n < 100:
            if n % 10 == 0:
                return tens[n // 10]
            return tens[n // 10] + "-" + ones[n % 10]
        if n < 1000:
            if n % 100 == 0:
                return ones[n // 100] + " hundred"
            return ones[n // 100] + " hundred " + number_to_words(n % 100)
        if n < 1000000:
            thousands = n // 1000
            remainder = n % 1000
            if remainder == 0:
                return number_to_words(thousands) + " thousand"
            return number_to_words(thousands) + " thousand " + number_to_words(remainder)
        if n < 1000000000:
            millions = n // 1000000
            remainder = n % 1000000
            if remainder == 0:
                return number_to_words(millions) + " million"
            return number_to_words(millions) + " million " + number_to_words(remainder)
        return str(n)  # Fallback for very large numbers
    
    def digit_to_word(d: str) -> str:
        """Convert a single digit to word."""
        return ones[int(d)]
    
    # Email normalization: test@example.com -> "test at example dot com"
    def normalize_email(match):
        email = match.group(0)
        email = email.replace('@', ' at ')
        email = email.replace('.', ' dot ')
        return email
    
    result = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', normalize_email, result)
    
    # Phone numbers: (123)456-7891 or 123-456-7891 -> spoken digits
    def normalize_phone(match):
        digits = re.sub(r'\D', '', match.group(0))
        if len(digits) == 10:
            return (f"{digit_to_word(digits[0])} {digit_to_word(digits[1])} {digit_to_word(digits[2])}, "
                    f"{digit_to_word(digits[3])} {digit_to_word(digits[4])} {digit_to_word(digits[5])}, "
                    f"{digit_to_word(digits[6])} {digit_to_word(digits[7])} {digit_to_word(digits[8])} {digit_to_word(digits[9])}")
        elif len(digits) == 11 and digits[0] == '1':
            return (f"one, {digit_to_word(digits[1])} {digit_to_word(digits[2])} {digit_to_word(digits[3])}, "
                    f"{digit_to_word(digits[4])} {digit_to_word(digits[5])} {digit_to_word(digits[6])}, "
                    f"{digit_to_word(digits[7])} {digit_to_word(digits[8])} {digit_to_word(digits[9])} {digit_to_word(digits[10])}")
        return match.group(0)
    
    result = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', normalize_phone, result)
    
    # Monetary values: $5,342.29 -> "five thousand three hundred forty-two dollars and twenty-nine cents"
    def normalize_money(match):
        amount = match.group(0)
        # Remove $ and commas
        cleaned = amount.replace('$', '').replace(',', '')
        try:
            if '.' in cleaned:
                dollars, cents = cleaned.split('.')
                dollars = int(dollars)
                cents = int(cents[:2].ljust(2, '0'))  # Ensure 2 digits
                if cents == 0:
                    return number_to_words(dollars) + " dollars"
                return number_to_words(dollars) + " dollars and " + number_to_words(cents) + " cents"
            else:
                return number_to_words(int(cleaned)) + " dollars"
        except:
            return amount
    
    result = re.sub(r'\$[\d,]+(?:\.\d{1,2})?', normalize_money, result)
    
    # Percentages: 45% -> "forty-five percent"
    def normalize_percent(match):
        num = match.group(1).replace(',', '')
        try:
            if '.' in num:
                whole, decimal = num.split('.')
                return number_to_words(int(whole)) + " point " + ' '.join([digit_to_word(d) for d in decimal]) + " percent"
            return number_to_words(int(num)) + " percent"
        except:
            return match.group(0)
    
    result = re.sub(r'([\d,]+(?:\.\d+)?)\s*%', normalize_percent, result)
    
    # Times: 12:55 PM -> "twelve fifty-five PM"
    def normalize_time(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3) if match.group(3) else ""
        
        if minute == 0:
            return number_to_words(hour) + " o'clock" + (" " + period if period else "")
        elif minute < 10:
            return number_to_words(hour) + " oh " + number_to_words(minute) + (" " + period if period else "")
        else:
            return number_to_words(hour) + " " + number_to_words(minute) + (" " + period if period else "")
    
    result = re.sub(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', normalize_time, result)
    
    # Dates: 5/6/2025 or 05/06/2025 -> "May sixth, twenty twenty-five"
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    ordinals = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
                'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth',
                'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'twenty-first', 'twenty-second',
                'twenty-third', 'twenty-fourth', 'twenty-fifth', 'twenty-sixth', 'twenty-seventh',
                'twenty-eighth', 'twenty-ninth', 'thirtieth', 'thirty-first']
    
    def normalize_date(match):
        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))
        
        if month < 1 or month > 12 or day < 1 or day > 31:
            return match.group(0)
        
        # Format year
        if year >= 2000 and year < 2010:
            year_words = "two thousand " + (number_to_words(year % 100) if year % 100 else "")
        elif year >= 2010 and year < 2100:
            year_words = "twenty " + number_to_words(year % 100)
        elif year >= 1900 and year < 2000:
            year_words = "nineteen " + number_to_words(year % 100)
        else:
            year_words = number_to_words(year)
        
        return f"{months[month - 1]} {ordinals[day]}, {year_words.strip()}"
    
    result = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', normalize_date, result)
    
    # Mathematical symbols
    result = re.sub(r'\s*\+\s*', ' plus ', result)
    result = re.sub(r'\s*-\s*(?=\d)', ' minus ', result)  # Only before numbers
    result = re.sub(r'\s*×\s*', ' times ', result)
    result = re.sub(r'\s*÷\s*', ' divided by ', result)
    result = re.sub(r'\s*=\s*', ' equals ', result)
    result = re.sub(r'\s*<\s*', ' less than ', result)
    result = re.sub(r'\s*>\s*', ' greater than ', result)
    result = re.sub(r'\s*≤\s*', ' less than or equal to ', result)
    result = re.sub(r'\s*≥\s*', ' greater than or equal to ', result)
    result = re.sub(r'\s*≠\s*', ' not equal to ', result)
    result = re.sub(r'\s*±\s*', ' plus or minus ', result)
    
    # Common abbreviations
    result = re.sub(r'\betc\.\s*', 'et cetera. ', result)
    result = re.sub(r'\be\.g\.\s*', 'for example, ', result)
    result = re.sub(r'\bi\.e\.\s*', 'that is, ', result)
    result = re.sub(r'\bvs\.\s*', 'versus ', result)
    result = re.sub(r'\bDr\.\s*', 'Doctor ', result)
    result = re.sub(r'\bMr\.\s*', 'Mister ', result)
    result = re.sub(r'\bMrs\.\s*', 'Missus ', result)
    result = re.sub(r'\bMs\.\s*', 'Miss ', result)
    result = re.sub(r'\bProf\.\s*', 'Professor ', result)
    
    # Ordinal numbers in text: 1st, 2nd, 3rd, etc.
    def normalize_ordinal(match):
        num = int(match.group(1))
        if num <= 31:
            return ordinals[num]
        # For larger ordinals, approximate
        if num % 100 >= 11 and num % 100 <= 13:
            suffix = "th"
        elif num % 10 == 1:
            suffix = "st"
        elif num % 10 == 2:
            suffix = "nd"
        elif num % 10 == 3:
            suffix = "rd"
        else:
            suffix = "th"
        return number_to_words(num) + suffix.replace('th', 'th').replace('st', 'st')
    
    result = re.sub(r'(\d+)(?:st|nd|rd|th)\b', normalize_ordinal, result)
    
    # Clean up extra whitespace
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()


# =============================================================================
# Inworld AI TTS
# =============================================================================

INWORLD_TTS_API_URL = "https://api.inworld.ai/tts/v1/voice"


class InworldQuotaExceededError(Exception):
    """Raised when Inworld free tier quota is exceeded."""
    pass


class InworldAPIError(Exception):
    """Raised for Inworld API errors that may be recoverable."""
    def __init__(self, message: str, status_code: int = None, can_fallback: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.can_fallback = can_fallback


def _check_inworld_error(response: requests.Response, chunk_idx: int = None) -> None:
    """Check Inworld API response for errors and raise appropriate exceptions.
    
    Common error codes:
    - 401: Invalid API key
    - 402: Payment required (quota exceeded on free tier)
    - 429: Rate limit exceeded
    - 503: Service temporarily unavailable
    """
    if response.ok:
        return
    
    status = response.status_code
    chunk_info = f" (chunk {chunk_idx})" if chunk_idx else ""
    
    try:
        error_data = response.json()
        error_msg = error_data.get('message', error_data.get('error', str(error_data)))
    except Exception:
        error_msg = response.text[:200] if response.text else "Unknown error"
    
    if status == 401:
        raise InworldAPIError(
            f"Invalid Inworld API key. Please check INWORLD_API_KEY in .env.local",
            status_code=status,
            can_fallback=False  # Can't fallback if key is wrong
        )
    
    elif status == 402:
        raise InworldQuotaExceededError(
            f"Inworld free tier quota exceeded{chunk_info}. "
            f"Consider upgrading your Inworld plan or switching to Gemini TTS in Settings."
        )
    
    elif status == 429:
        raise InworldQuotaExceededError(
            f"Inworld rate limit exceeded{chunk_info}. "
            f"You've made too many requests. Will attempt fallback to Gemini TTS."
        )
    
    elif status == 503:
        raise InworldAPIError(
            f"Inworld service temporarily unavailable{chunk_info}. "
            f"The service may be overloaded. Will attempt fallback to Gemini TTS.",
            status_code=status,
            can_fallback=True
        )
    
    else:
        raise InworldAPIError(
            f"Inworld API error{chunk_info} (HTTP {status}): {error_msg}",
            status_code=status,
            can_fallback=status >= 500  # Server errors can fallback
        )


def synthesize_with_inworld_tts(text: str,
                                 api_key: str,
                                 voice_id: str,
                                 model_id: str,
                                 cancel_event,
                                 log: Callable[[str], None],
                                 progress_callback: Optional[Callable] = None) -> Optional[bytes]:
    """Synthesize speech using Inworld AI TTS.
    
    Args:
        text: Text to convert to speech
        api_key: Inworld API key (base64 encoded credentials)
        voice_id: Voice ID (e.g., "Ashley", "Brian")
        model_id: Model ID (e.g., "inworld-tts-1")
        cancel_event: Cancellation event
        log: Logging callback
        progress_callback: Progress callback
    
    Returns:
        MP3 audio bytes or None if cancelled
    """
    import base64
    
    if not api_key:
        raise ValueError("Inworld API key is missing. Please set INWORLD_API_KEY in .env.local")
    
    if cancel_event.is_set():
        return None
    
    # Normalize text for TTS (handle numbers, dates, symbols, etc.)
    normalized_text = normalize_text_for_tts(text)
    
    # Chunk the text for processing
    # Inworld API has a strict 2000 character limit - use 1800 for safety margin
    chunks = _chunk_text_for_tts(normalized_text, max_chars=1800)
    total_chunks = len(chunks)
    all_audio_data: List[bytes] = []
    
    log(f"Split text into {total_chunks} chunks (max 1800 chars each for Inworld API)")
    
    headers = {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json"
    }
    
    # Validate all chunks are under the limit before starting
    INWORLD_CHAR_LIMIT = 2000
    for i, chunk in enumerate(chunks):
        if len(chunk) > INWORLD_CHAR_LIMIT:
            log(f"Warning: Chunk {i+1} exceeds {INWORLD_CHAR_LIMIT} chars ({len(chunk)}), re-splitting...")
            # Re-chunk this specific chunk
            sub_chunks = _chunk_text_for_tts(chunk, max_chars=INWORLD_CHAR_LIMIT - 100)
            chunks = chunks[:i] + sub_chunks + chunks[i+1:]
    
    total_chunks = len(chunks)
    log(f"Final chunk count: {total_chunks} (validated under {INWORLD_CHAR_LIMIT} chars each)")
    
    with ProgressTracker(total_chunks, "Generating audio",
                        callback=progress_callback) as progress:
        
        for idx, chunk in enumerate(chunks, start=1):
            if cancel_event.is_set():
                return None
            
            # Final safety check
            if len(chunk) > INWORLD_CHAR_LIMIT:
                log(f"Error: Chunk {idx} still exceeds limit ({len(chunk)} chars), truncating...")
                chunk = chunk[:INWORLD_CHAR_LIMIT - 10] + "..."
            
            log(f"Generating audio chunk {idx}/{total_chunks} ({len(chunk)} chars) with Inworld TTS ({voice_id})...")
            
            payload = {
                "text": chunk,
                "voiceId": voice_id,
                "modelId": model_id
            }
            
            try:
                response = requests.post(
                    INWORLD_TTS_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=120
                )
                
                # Check for quota/rate limit errors
                _check_inworld_error(response, chunk_idx=idx)
                
                result = response.json()
                
                if 'audioContent' not in result:
                    raise RuntimeError(f"No audioContent in response: {result}")
                
                audio_content = base64.b64decode(result['audioContent'])
                all_audio_data.append(audio_content)
                
            except (InworldQuotaExceededError, InworldAPIError):
                # Re-raise these for fallback handling
                raise
            except requests.exceptions.Timeout:
                raise InworldAPIError(
                    f"Inworld TTS timeout on chunk {idx}/{total_chunks}. Service may be slow.",
                    can_fallback=True
                )
            except requests.exceptions.ConnectionError:
                raise InworldAPIError(
                    f"Cannot connect to Inworld API. Check your internet connection.",
                    can_fallback=True
                )
            except Exception as e:
                raise RuntimeError(f"Inworld TTS error on chunk {idx}/{total_chunks}: {e}")
            
            progress.update(1, f"Chunk {idx}/{total_chunks}")
            
            # Memory pressure check
            if check_memory_pressure(threshold_mb=800):
                log("Warning: High memory usage detected")
                force_gc()
    
    if not all_audio_data:
        raise RuntimeError("No audio data generated")
    
    # Combine all audio chunks (Inworld returns MP3 directly)
    combined_audio = b"".join(all_audio_data)
    
    # Clear individual chunks to free memory
    all_audio_data.clear()
    force_gc()
    
    return combined_audio


def synthesize_with_inworld_tts_streaming(text: str,
                                           output_path: str,
                                           api_key: str,
                                           voice_id: str,
                                           model_id: str,
                                           cancel_event,
                                           log: Callable[[str], None],
                                           progress_callback: Optional[Callable] = None,
                                           memory_threshold_mb: float = 600) -> Optional[int]:
    """Synthesize speech using Inworld AI TTS with streaming to disk.
    
    Writes audio chunks to disk incrementally for memory efficiency.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save MP3 file
        api_key: Inworld API key
        voice_id: Voice ID
        model_id: Model ID
        cancel_event: Cancellation event
        log: Logging callback
        progress_callback: Progress callback
        memory_threshold_mb: Memory threshold for flushing to disk
    
    Returns:
        Final file size in bytes, or None if cancelled
    """
    import base64
    
    if not api_key:
        raise ValueError("Inworld API key is missing. Please set INWORLD_API_KEY in .env.local")
    
    if cancel_event.is_set():
        return None
    
    # Normalize text for TTS (handle numbers, dates, symbols, etc.)
    normalized_text = normalize_text_for_tts(text)
    
    # Chunk the text for processing
    # Inworld API has a strict 2000 character limit - use 1800 for safety margin
    chunks = _chunk_text_for_tts(normalized_text, max_chars=1800)
    
    # Validate all chunks are under the limit before starting
    INWORLD_CHAR_LIMIT = 2000
    for i, chunk in enumerate(chunks):
        if len(chunk) > INWORLD_CHAR_LIMIT:
            log(f"Warning: Chunk {i+1} exceeds {INWORLD_CHAR_LIMIT} chars ({len(chunk)}), re-splitting...")
            # Re-chunk this specific chunk
            sub_chunks = _chunk_text_for_tts(chunk, max_chars=INWORLD_CHAR_LIMIT - 100)
            chunks = chunks[:i] + sub_chunks + chunks[i+1:]
    
    total_chunks = len(chunks)
    log(f"Split text into {total_chunks} chunks (validated under {INWORLD_CHAR_LIMIT} chars each)")
    
    headers = {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json"
    }
    
    # Temporary directory for chunk files
    temp_dir = tempfile.mkdtemp(prefix="inworld_tts_chunks_")
    chunk_files: List[str] = []
    audio_buffer: List[bytes] = []
    buffer_size = 0
    
    try:
        with ProgressTracker(total_chunks, "Generating audio",
                            callback=progress_callback) as progress:
            
            for idx, chunk in enumerate(chunks, start=1):
                if cancel_event.is_set():
                    return None
                
                # Final safety check
                if len(chunk) > INWORLD_CHAR_LIMIT:
                    log(f"Error: Chunk {idx} still exceeds limit ({len(chunk)} chars), truncating...")
                    chunk = chunk[:INWORLD_CHAR_LIMIT - 10] + "..."
                
                progress.set_description(f"Audio chunk {idx}/{total_chunks}")
                log(f"Generating audio chunk {idx}/{total_chunks} ({len(chunk)} chars) with Inworld TTS ({voice_id})...")
                
                payload = {
                    "text": chunk,
                    "voiceId": voice_id,
                    "modelId": model_id
                }
                
                try:
                    response = requests.post(
                        INWORLD_TTS_API_URL,
                        json=payload,
                        headers=headers,
                        timeout=120
                    )
                    
                    # Check for quota/rate limit errors
                    _check_inworld_error(response, chunk_idx=idx)
                    
                    result = response.json()
                    
                    if 'audioContent' not in result:
                        raise RuntimeError(f"No audioContent in response")
                    
                    audio_content = base64.b64decode(result['audioContent'])
                    audio_buffer.append(audio_content)
                    buffer_size += len(audio_content)
                    
                except (InworldQuotaExceededError, InworldAPIError):
                    # Re-raise these for fallback handling
                    raise
                except requests.exceptions.Timeout:
                    raise InworldAPIError(
                        f"Inworld TTS timeout on chunk {idx}/{total_chunks}. Service may be slow.",
                        can_fallback=True
                    )
                except requests.exceptions.ConnectionError:
                    raise InworldAPIError(
                        f"Cannot connect to Inworld API. Check your internet connection.",
                        can_fallback=True
                    )
                except Exception as e:
                    raise RuntimeError(f"Inworld TTS error on chunk {idx}/{total_chunks}: {e}")
                
                # Check memory pressure and flush to disk if needed
                if buffer_size > memory_threshold_mb * 1024 * 1024 or check_memory_pressure(memory_threshold_mb):
                    log(f"Flushing {buffer_size / (1024*1024):.1f} MB audio buffer to disk...")
                    
                    chunk_mp3 = os.path.join(temp_dir, f"chunk_{len(chunk_files):04d}.mp3")
                    with open(chunk_mp3, 'wb') as f:
                        for audio in audio_buffer:
                            f.write(audio)
                    
                    chunk_files.append(chunk_mp3)
                    audio_buffer.clear()
                    buffer_size = 0
                    force_gc()
                
                progress.update(1, f"Chunk {idx}/{total_chunks}")
        
        # Handle remaining buffer
        if audio_buffer:
            if chunk_files:
                # We already flushed some chunks, so write remaining to disk too
                chunk_mp3 = os.path.join(temp_dir, f"chunk_{len(chunk_files):04d}.mp3")
                with open(chunk_mp3, 'wb') as f:
                    for audio in audio_buffer:
                        f.write(audio)
                chunk_files.append(chunk_mp3)
                audio_buffer.clear()
        
        # Combine audio files
        log("Combining audio chunks...")
        
        if chunk_files:
            # Use ffmpeg to concatenate MP3 files
            concat_list = os.path.join(temp_dir, "concat.txt")
            with open(concat_list, 'w') as f:
                for cf in chunk_files:
                    f.write(f"file '{cf}'\n")
            
            result = subprocess.run(
                [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_list,
                    '-acodec', 'copy',  # Just copy, no re-encoding
                    output_path
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat error: {result.stderr}")
        else:
            # All audio is still in buffer, write directly
            with open(output_path, 'wb') as f:
                for audio in audio_buffer:
                    f.write(audio)
        
        return os.path.getsize(output_path)
        
    finally:
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        
        audio_buffer.clear()
        force_gc()


# =============================================================================
# Gemini TTS (Streaming)
# =============================================================================

def synthesize_with_gemini_tts_streaming(text: str,
                                          output_path: str,
                                          api_key: str,
                                          tts_model: str,
                                          voice_name: str,
                                          style_prompt: str,
                                          cancel_event,
                                          log: Callable[[str], None],
                                          progress_callback: Optional[Callable] = None,
                                          memory_threshold_mb: float = 600) -> Optional[int]:
    """Synthesize speech using Gemini TTS with streaming to disk.
    
    Instead of accumulating all audio in memory, writes chunks to disk
    incrementally when memory pressure is detected.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save MP3 file
        api_key: Gemini API key
        tts_model: TTS model name
        voice_name: Voice name
        style_prompt: Optional style instructions
        cancel_event: Cancellation event
        log: Logging callback
        progress_callback: Progress callback
        memory_threshold_mb: Memory threshold for flushing to disk
    
    Returns:
        Final file size in bytes, or None if cancelled
    """
    if not api_key:
        raise ValueError("Gemini API key is missing. Please set it in Settings.")
    
    if cancel_event.is_set():
        return None

    client = genai.Client(api_key=api_key)
    
    # Chunk the text for processing
    chunks = _chunk_text_for_tts(text)
    total_chunks = len(chunks)
    
    # Temporary directory for chunk files
    temp_dir = tempfile.mkdtemp(prefix="tts_chunks_")
    chunk_files: List[str] = []
    pcm_buffer: List[bytes] = []
    buffer_size = 0
    
    try:
        with ProgressTracker(total_chunks, "Generating audio",
                            callback=progress_callback) as progress:
            
            for idx, chunk in enumerate(chunks, start=1):
                if cancel_event.is_set():
                    return None
                
                progress.set_description(f"Audio chunk {idx}/{total_chunks}")
                log(f"Generating audio chunk {idx}/{total_chunks} with Gemini TTS ({voice_name})...")
                
                # Build the prompt with optional style instructions
                if style_prompt and style_prompt.strip():
                    tts_prompt = f"{style_prompt.strip()}\n\n{chunk}"
                else:
                    tts_prompt = chunk
                
                try:
                    response = client.models.generate_content(
                        model=tts_model,
                        contents=tts_prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=voice_name,
                                    )
                                )
                            ),
                        )
                    )
                    
                    # Extract audio data from response
                    if response.candidates and response.candidates[0].content.parts:
                        part = response.candidates[0].content.parts[0]
                        if hasattr(part, 'inline_data') and part.inline_data:
                            pcm_data = part.inline_data.data
                            if isinstance(pcm_data, str):
                                import base64
                                pcm_data = base64.b64decode(pcm_data)
                            
                            pcm_buffer.append(pcm_data)
                            buffer_size += len(pcm_data)
                        else:
                            log(f"Warning: No audio data in chunk {idx}")
                    else:
                        log(f"Warning: Empty response for chunk {idx}")
                        
                except Exception as e:
                    raise RuntimeError(f"Gemini TTS error on chunk {idx}/{total_chunks}: {e}")
                
                # Check memory pressure and flush to disk if needed
                if buffer_size > memory_threshold_mb * 1024 * 1024 or check_memory_pressure(memory_threshold_mb):
                    log(f"Flushing {buffer_size / (1024*1024):.1f} MB audio buffer to disk...")
                    
                    chunk_wav = os.path.join(temp_dir, f"chunk_{len(chunk_files):04d}.wav")
                    with wave.open(chunk_wav, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        for pcm in pcm_buffer:
                            wf.writeframes(pcm)
                    
                    chunk_files.append(chunk_wav)
                    pcm_buffer.clear()
                    buffer_size = 0
                    force_gc()
                
                progress.update(1, f"Chunk {idx}/{total_chunks}")
        
        # Handle remaining buffer
        if pcm_buffer:
            if chunk_files:
                # We already flushed some chunks, so write remaining to disk too
                chunk_wav = os.path.join(temp_dir, f"chunk_{len(chunk_files):04d}.wav")
                with wave.open(chunk_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    for pcm in pcm_buffer:
                        wf.writeframes(pcm)
                chunk_files.append(chunk_wav)
                pcm_buffer.clear()
            
        # Convert to MP3
        log("Converting audio to MP3...")
        
        if chunk_files:
            # Concatenate chunk files and convert to MP3
            concat_list = os.path.join(temp_dir, "concat.txt")
            with open(concat_list, 'w') as f:
                for cf in chunk_files:
                    f.write(f"file '{cf}'\n")
            
            # Use ffmpeg to concatenate and encode
            result = subprocess.run(
                [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_list,
                    '-acodec', 'libmp3lame',
                    '-b:a', '192k',
                    output_path
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat error: {result.stderr}")
        else:
            # All audio is still in buffer, convert directly
            file_size = _pcm_to_mp3_streaming(pcm_buffer, output_path)
            return file_size
        
        return os.path.getsize(output_path)
        
    finally:
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        
        pcm_buffer.clear()
        force_gc()


def synthesize_with_gemini_tts(text: str,
                                api_key: str,
                                tts_model: str,
                                voice_name: str,
                                style_prompt: str,
                                cancel_event,
                                log: Callable[[str], None],
                                progress_callback: Optional[Callable] = None) -> Optional[bytes]:
    """Synthesize speech using Gemini TTS (returns bytes for backward compatibility).
    
    For large texts, consider using synthesize_with_gemini_tts_streaming instead.
    """
    if not api_key:
        raise ValueError("Gemini API key is missing. Please set it in Settings.")
    
    if cancel_event.is_set():
        return None

    client = genai.Client(api_key=api_key)
    
    # Chunk the text for processing
    chunks = _chunk_text_for_tts(text)
    total_chunks = len(chunks)
    all_pcm_data: List[bytes] = []
    
    with ProgressTracker(total_chunks, "Generating audio",
                        callback=progress_callback) as progress:
        
        for idx, chunk in enumerate(chunks, start=1):
            if cancel_event.is_set():
                return None
            
            log(f"Generating audio chunk {idx}/{total_chunks} with Gemini TTS ({voice_name})...")
            
            # Build the prompt with optional style instructions
            if style_prompt and style_prompt.strip():
                tts_prompt = f"{style_prompt.strip()}\n\n{chunk}"
            else:
                tts_prompt = chunk
            
            try:
                response = client.models.generate_content(
                    model=tts_model,
                    contents=tts_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name,
                                )
                            )
                        ),
                    )
                )
                
                # Extract audio data from response
                if response.candidates and response.candidates[0].content.parts:
                    part = response.candidates[0].content.parts[0]
                    if hasattr(part, 'inline_data') and part.inline_data:
                        pcm_data = part.inline_data.data
                        if isinstance(pcm_data, str):
                            import base64
                            pcm_data = base64.b64decode(pcm_data)
                        all_pcm_data.append(pcm_data)
                    else:
                        log(f"Warning: No audio data in chunk {idx}")
                else:
                    log(f"Warning: Empty response for chunk {idx}")
                    
            except Exception as e:
                raise RuntimeError(f"Gemini TTS error on chunk {idx}/{total_chunks}: {e}")
            
            progress.update(1, f"Chunk {idx}/{total_chunks}")
            
            # Memory pressure check
            if check_memory_pressure(threshold_mb=800):
                log("Warning: High memory usage detected")
                force_gc()
    
    if not all_pcm_data:
        raise RuntimeError("No audio data generated")
    
    # Combine all PCM chunks
    combined_pcm = b"".join(all_pcm_data)
    
    # Clear individual chunks to free memory
    all_pcm_data.clear()
    force_gc()
    
    # Convert to MP3 using temp file approach
    log("Converting audio to MP3...")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
        wav_path = wav_file.name
        with wave.open(wav_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(combined_pcm)
    
    # Free combined PCM
    del combined_pcm
    force_gc()
    
    mp3_path = wav_path.replace('.wav', '.mp3')
    
    try:
        result = subprocess.run(
            [
                'ffmpeg', '-y',
                '-i', wav_path,
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                mp3_path
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
        
        with open(mp3_path, 'rb') as f:
            return f.read()
            
    finally:
        for path in [wav_path, mp3_path]:
            try:
                os.unlink(path)
            except Exception:
                pass


# =============================================================================
# Conversion Worker
# =============================================================================

class ConversionWorker:
    @staticmethod
    def run(pdf_path: str,
            citation_style: str,
            conversion_mode: str,
            config_mgr,
            cancel_event,
            done_callback: Callable[[str, Optional[str]], None],
            progress_callback: Optional[Callable] = None) -> None:
        """Run the conversion pipeline."""
        def send(event: str, payload: Optional[str] = None):
            try:
                done_callback(event, payload)
            except Exception:
                pass

        try:
            send("status", "Parsing PDF...")
            text, image_locations = parse_pdf(pdf_path, progress_callback=progress_callback)
            if cancel_event.is_set():
                send("cancelled")
                return

            cfg = config_mgr.load()
            api_key = cfg.get("gemini_api_key", "")
            model_name = cfg.get("model_name", "gemini-2.0-flash")

            if (conversion_mode or 'Summarized') == 'Verbatim':
                script = clean_with_gemini(
                    text=text,
                    pdf_path=pdf_path,
                    image_locations=image_locations,
                    api_key=api_key,
                    model_name=model_name,
                    citation_style=citation_style,
                    cancel_event=cancel_event,
                    log=lambda m: send("status", m),
                    progress_callback=progress_callback,
                )
            else:
                script = structure_with_gemini(
                    text=text,
                    pdf_path=pdf_path,
                    image_locations=image_locations,
                    api_key=api_key,
                    model_name=model_name,
                    citation_style=citation_style,
                    cancel_event=cancel_event,
                    log=lambda m: send("status", m),
                    progress_callback=progress_callback,
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
                            log: Callable[[str], None],
                            progress_callback: Optional[Callable] = None,
                            allow_fallback: bool = True) -> Optional[bytes]:
        """Generate audio using the configured TTS provider (returns bytes).
        
        If Inworld fails with quota/rate limits and allow_fallback=True,
        automatically falls back to Gemini TTS.
        """
        cfg = config_mgr.load()
        provider = cfg.get("tts_provider", "inworld")
        
        if provider == "inworld":
            # Use Inworld AI TTS
            api_key = cfg.get("inworld_api_key", "")
            voice_id = cfg.get("inworld_voice_id", "Ashley")
            model_id = cfg.get("inworld_model_id", "inworld-tts-1")
            
            if not api_key:
                log("⚠️  No Inworld API key found, falling back to Gemini TTS")
                provider = "gemini"
            else:
                log(f"Using Inworld TTS (voice: {voice_id})")
                try:
                    return synthesize_with_inworld_tts(
                        text=text,
                        api_key=api_key,
                        voice_id=voice_id,
                        model_id=model_id,
                        cancel_event=cancel_event,
                        log=log,
                        progress_callback=progress_callback,
                    )
                except InworldQuotaExceededError as e:
                    if allow_fallback:
                        log(f"⚠️  {e}")
                        log("🔄 Falling back to Gemini TTS...")
                        provider = "gemini"
                    else:
                        raise
                except InworldAPIError as e:
                    if allow_fallback and e.can_fallback:
                        log(f"⚠️  {e}")
                        log("🔄 Falling back to Gemini TTS...")
                        provider = "gemini"
                    else:
                        raise
        
        if provider == "gemini":
            # Use Gemini TTS
            api_key = cfg.get("gemini_api_key", "")
            tts_model = cfg.get("tts_model_name", "gemini-2.5-flash-preview-tts")
            voice_name = cfg.get("tts_voice_name", "Kore")
            style_prompt = cfg.get("tts_style_prompt", "")
            
            if not api_key:
                raise ValueError("No Gemini API key found. Please set it in Settings.")
            
            log(f"Using Gemini TTS (voice: {voice_name})")
            return synthesize_with_gemini_tts(
                text=text,
                api_key=api_key,
                tts_model=tts_model,
                voice_name=voice_name,
                style_prompt=style_prompt,
                cancel_event=cancel_event,
                log=log,
                progress_callback=progress_callback,
            )
        
        return None

    @staticmethod
    def generate_audio_streaming(text: str,
                                  output_path: str,
                                  config_mgr,
                                  cancel_event,
                                  log: Callable[[str], None],
                                  progress_callback: Optional[Callable] = None,
                                  allow_fallback: bool = True) -> Optional[int]:
        """Generate audio using the configured TTS provider with streaming to disk.
        
        If Inworld fails with quota/rate limits and allow_fallback=True,
        automatically falls back to Gemini TTS.
        
        Returns file size in bytes, or None if cancelled.
        """
        cfg = config_mgr.load()
        provider = cfg.get("tts_provider", "inworld")
        
        if provider == "inworld":
            # Use Inworld AI TTS
            api_key = cfg.get("inworld_api_key", "")
            voice_id = cfg.get("inworld_voice_id", "Ashley")
            model_id = cfg.get("inworld_model_id", "inworld-tts-1")
            
            if not api_key:
                log("⚠️  No Inworld API key found, falling back to Gemini TTS")
                provider = "gemini"
            else:
                log(f"Using Inworld TTS (voice: {voice_id})")
                try:
                    return synthesize_with_inworld_tts_streaming(
                        text=text,
                        output_path=output_path,
                        api_key=api_key,
                        voice_id=voice_id,
                        model_id=model_id,
                        cancel_event=cancel_event,
                        log=log,
                        progress_callback=progress_callback,
                    )
                except InworldQuotaExceededError as e:
                    if allow_fallback:
                        log(f"⚠️  {e}")
                        log("🔄 Falling back to Gemini TTS...")
                        provider = "gemini"
                    else:
                        raise
                except InworldAPIError as e:
                    if allow_fallback and e.can_fallback:
                        log(f"⚠️  {e}")
                        log("🔄 Falling back to Gemini TTS...")
                        provider = "gemini"
                    else:
                        raise
        
        if provider == "gemini":
            # Use Gemini TTS
            api_key = cfg.get("gemini_api_key", "")
            tts_model = cfg.get("tts_model_name", "gemini-2.5-flash-preview-tts")
            voice_name = cfg.get("tts_voice_name", "Kore")
            style_prompt = cfg.get("tts_style_prompt", "")
            
            if not api_key:
                raise ValueError("No Gemini API key found. Please set it in Settings.")
            
            log(f"Using Gemini TTS (voice: {voice_name})")
            return synthesize_with_gemini_tts_streaming(
                text=text,
                output_path=output_path,
                api_key=api_key,
                tts_model=tts_model,
                voice_name=voice_name,
                style_prompt=style_prompt,
                cancel_event=cancel_event,
                log=log,
                progress_callback=progress_callback,
            )
        
        return None
