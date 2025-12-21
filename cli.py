#!/usr/bin/env python3
"""
Command-line interface for Research Paper Audiobook Converter.

Includes tqdm progress bars for long-running operations.
"""
import argparse
import gzip
import os
import sys
import threading

from tqdm import tqdm

from config import ConfigManager
from processing import (
    estimate_tokens,
    parse_pdf,
    structure_with_gemini,
    clean_with_gemini,
    ConversionWorker,
    PDFCache,
    get_memory_usage_mb,
)


def _normalize_chars_count(text: str) -> int:
    return len(" ".join(text.split()))


class CLIProgressCallback:
    """Progress callback that updates a tqdm bar."""
    
    def __init__(self, desc: str = "Processing"):
        self.pbar = None
        self.desc = desc
    
    def __call__(self, current: int, total: int, status: str = ""):
        if self.pbar is None:
            self.pbar = tqdm(total=total, desc=self.desc, unit="step", 
                            leave=False, dynamic_ncols=True)
        
        # Update to current position
        self.pbar.n = current
        self.pbar.refresh()
        
        if status:
            self.pbar.set_postfix_str(status[:30])
    
    def close(self):
        if self.pbar:
            self.pbar.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert a research PDF to MP3 via two-step process (Gemini -> TTS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py paper.pdf
  python cli.py paper.pdf --mode Verbatim --output audiobook.mp3
  python cli.py paper.pdf --citations "Subtle Mention" --compress-script
        """
    )
    parser.add_argument("article", help="Path to the PDF article to convert")
    parser.add_argument(
        "--mode",
        choices=["Summarized", "Verbatim"],
        help="Conversion mode: Summarized (Gemini-structured) or Verbatim (Gemini-cleaned with image descriptions)",
    )
    parser.add_argument(
        "--citations",
        choices=["Ignore", "Subtle Mention"],
        help="Citation handling in the script",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the output MP3 (default: <article_basename>.mp3)",
    )
    parser.add_argument(
        "--config",
        help="Path to config.json (defaults to project config.json)",
    )
    parser.add_argument(
        "--compress-script",
        action="store_true",
        help="Save script as gzip-compressed .txt.gz instead of plain .txt",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars (show only final output)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress messages",
    )

    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(project_dir, "config.json")
    cfg_mgr = ConfigManager(config_path)
    cfg_mgr.ensure_config()
    cfg = cfg_mgr.load()

    citation_style = args.citations or cfg.get("citation_style", "Ignore")
    conversion_mode = args.mode or cfg.get("conversion_mode", "Summarized")

    pdf_path = os.path.abspath(args.article)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    verbose = args.verbose and not args.quiet
    show_progress = not args.quiet

    def log(msg: str):
        if verbose:
            print(f"  {msg}")

    # =========================================================================
    # Step 0: Estimate Gemini cost (uses cache if available)
    # =========================================================================
    if show_progress:
        print(f"\n📄 Processing: {os.path.basename(pdf_path)}")
        print(f"   Mode: {conversion_mode} | Citations: {citation_style}")
        print()
    
    try:
        with tqdm(total=1, desc="Estimating cost", disable=not show_progress, leave=False) as pbar:
            tokens = estimate_tokens(pdf_path)
            gem_cost, text_model = cfg_mgr.estimate_gemini_cost(tokens)
            pbar.update(1)
        
        if show_progress:
            print(f"💰 Estimated Gemini ({text_model}): {tokens:,} tokens (~${gem_cost:.4f})")
    except Exception as e:
        print(f"Failed to estimate Gemini cost: {e}", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Step 1: Parse PDF (uses cache from estimation)
    # =========================================================================
    cancel_event = threading.Event()
    
    try:
        if show_progress:
            print()
        
        progress_cb = CLIProgressCallback("Parsing PDF") if show_progress else None
        
        text, image_locations = parse_pdf(pdf_path)
        
        if progress_cb:
            progress_cb.close()
        
        if show_progress:
            print(f"📝 Extracted: {len(text):,} characters, {len(image_locations)} figures")
    except Exception as e:
        print(f"PDF parsing error: {e}", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Step 2: Generate script with Gemini
    # =========================================================================
    api_key = cfg.get("gemini_api_key", "")
    model_name = cfg.get("model_name", "gemini-2.0-flash")
    
    if not api_key:
        print("Error: Gemini API key not set. Please configure in config.json", file=sys.stderr)
        sys.exit(1)
    
    try:
        if show_progress:
            print()
            mode_desc = "Summarizing" if conversion_mode == "Summarized" else "Cleaning (with image descriptions)"
            print(f"🤖 {mode_desc} with Gemini...")
        
        progress_cb = CLIProgressCallback("Processing") if show_progress else None
        
        if conversion_mode == "Verbatim":
            script = clean_with_gemini(
                text=text,
                pdf_path=pdf_path,
                image_locations=image_locations,
                api_key=api_key,
                model_name=model_name,
                citation_style=citation_style,
                cancel_event=cancel_event,
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
                citation_style=citation_style,
                cancel_event=cancel_event,
                log=log,
                progress_callback=progress_cb,
            )
        
        if progress_cb:
            progress_cb.close()
        
        if show_progress:
            print(f"📜 Generated script: {len(script):,} characters")
            
    except Exception as e:
        print(f"Conversion error: {e}", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Step 3: Estimate TTS cost
    # =========================================================================
    try:
        num_chars = _normalize_chars_count(script)
        tts_cost, tts_provider, tts_model = cfg_mgr.estimate_tts_cost(num_chars)
        
        # Get voice name based on provider
        if tts_provider == "inworld":
            voice_name = cfg.get("inworld_voice_id", "Ashley")
            provider_display = "Inworld AI"
        else:
            voice_name = cfg.get("tts_voice_name", "Kore")
            provider_display = "Gemini"
        
        if show_progress:
            print(f"💰 Estimated TTS ({tts_model}): {num_chars:,} characters (~${tts_cost:.4f})")
            print(f"🎙️  Provider: {provider_display} | Voice: {voice_name}")
    except Exception as e:
        print(f"Failed to estimate TTS cost: {e}", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Step 4: Determine output paths
    # =========================================================================
    if args.output:
        mp3_path = os.path.abspath(args.output)
    else:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        input_dir = os.path.dirname(pdf_path)
        mp3_path = os.path.join(input_dir, f"{base}.mp3")

    if args.compress_script:
        txt_path = os.path.splitext(mp3_path)[0] + ".txt.gz"
    else:
        txt_path = os.path.splitext(mp3_path)[0] + ".txt"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(mp3_path) or ".", exist_ok=True)

    # =========================================================================
    # Step 5: Generate audio with streaming
    # =========================================================================
    try:
        if show_progress:
            print()
            print("🔊 Generating audio (streaming to disk)...")
            mem_before = get_memory_usage_mb()
        
        progress_cb = CLIProgressCallback("Generating audio") if show_progress else None
        
        # Use streaming for memory efficiency
        # Allow fallback to Gemini if Inworld quota exceeded
        allow_fallback = cfg.get('tts_fallback_enabled', True)
        
        file_size = ConversionWorker.generate_audio_streaming(
            script,
            mp3_path,
            cfg_mgr,
            cancel_event,
            log=log,
            progress_callback=progress_cb,
            allow_fallback=allow_fallback,
        )
        
        if progress_cb:
            progress_cb.close()
        
        if file_size is None:
            print("Operation cancelled.", file=sys.stderr)
            sys.exit(1)
        
        if show_progress:
            mem_after = get_memory_usage_mb()
            print(f"✅ Audio saved: {mp3_path}")
            print(f"   Size: {file_size / (1024*1024):.2f} MB")
            if verbose:
                print(f"   Memory: {mem_before:.0f} MB -> {mem_after:.0f} MB")
                
    except Exception as e:
        print(f"Audio synthesis error: {e}", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Step 6: Save script
    # =========================================================================
    try:
        if args.compress_script:
            with gzip.open(txt_path, "wt", encoding="utf-8") as tf:
                tf.write(script)
        else:
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(script)
        
        if show_progress:
            print(f"✅ Script saved: {txt_path}")
    except Exception as e:
        print(f"Failed to save script: {e}", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Summary
    # =========================================================================
    if show_progress:
        total_cost = gem_cost + tts_cost
        print()
        print(f"{'='*50}")
        print(f"✨ Conversion complete!")
        print(f"   Total estimated cost: ~${total_cost:.4f}")
        print(f"{'='*50}")
    
    # Clear cache to free memory
    PDFCache.clear()


if __name__ == "__main__":
    main()
