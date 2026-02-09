#!/usr/bin/env python3
"""
Research Paper Audiobook Converter - Streamlined Single-Click UI
"""
import gzip
import threading
import queue
import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, Dict, List

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from config import ConfigManager, GEMINI_TTS_VOICES, INWORLD_TTS_VOICES, INWORLD_TTS_MODELS, TTS_PROVIDERS, fetch_inworld_voices
from processing import (
    estimate_tokens,
    parse_pdf,
    structure_with_gemini,
    clean_with_gemini,
    ConversionWorker,
    get_pdf_page_count,
    DEFAULT_MAX_PDF_PAGES,
)

# Voice sample directory
VOICE_SAMPLES_DIR = Path(__file__).parent / "voice_samples"

# Fun example phrases for voice previews
VOICE_PREVIEW_PHRASES = [
    "Did you know that octopuses have three hearts and blue blood? Science is wonderfully weird!",
    "Welcome to the future of research. Where papers become podcasts and knowledge flows like music.",
    "Ah, the humble PDF. Destroyer of trees, consumer of ink, and now... narrator of tales!",
    "In a world full of equations, be the one who explains them with style and a hint of drama.",
    "They said I couldn't turn a research paper into audio gold. They were magnificently wrong.",
]

APP_TITLE = "Research Paper Audiobook Converter"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

# Pipeline step definitions
PIPELINE_STEPS = [
    ("parsing", "Parsing PDF"),
    ("structuring", "Processing with Gemini"),
    ("tts", "Generating Audio"),
    ("saving", "Saving Files"),
]


class ToolTip:
    """Simple tooltip for CustomTkinter widgets."""
    
    def __init__(self, widget, text_callback):
        self.widget = widget
        self.text_callback = text_callback
        self.tip_window = None
        
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
    
    def show(self, event=None):
        text = self.text_callback()
        if not text:
            return
        
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        
        label = ctk.CTkLabel(
            tw,
            text=text,
            font=ctk.CTkFont(size=11, family="monospace"),
            fg_color=("gray85", "gray25"),
            corner_radius=4,
            padx=8,
            pady=4,
        )
        label.pack()
    
    def hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class CostBadge(ctk.CTkFrame):
    """Unobtrusive cost estimation badge with loading state."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color="transparent")
        
        self.label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray60"),
        )
        self.label.pack(side="left", padx=(0, 5))
        
        self.loading = False
        self.loading_dots = 0
        self._animate_id = None
        
    def set_loading(self, loading: bool = True):
        """Show loading animation."""
        self.loading = loading
        if loading:
            self.label.configure(text="Estimating cost...")
            self._animate()
        else:
            if self._animate_id:
                self.after_cancel(self._animate_id)
                self._animate_id = None
    
    def _animate(self):
        """Animate loading dots."""
        if not self.loading:
            return
        dots = "." * (self.loading_dots % 4)
        self.label.configure(text=f"Estimating cost{dots}")
        self.loading_dots += 1
        self._animate_id = self.after(400, self._animate)
    
    def set_cost(self, summarized_cost: float, verbatim_cost: float):
        """Display estimated costs for both modes."""
        self.loading = False
        if self._animate_id:
            self.after_cancel(self._animate_id)
        self.label.configure(
            text=f"Est. cost: Summarized ~${summarized_cost:.4f} | Verbatim ~${verbatim_cost:.4f}"
        )
    
    def clear(self):
        """Clear the cost display."""
        self.loading = False
        if self._animate_id:
            self.after_cancel(self._animate_id)
        self.label.configure(text="")


class PipelineStepCard(ctk.CTkFrame):
    """Individual pipeline step indicator."""
    
    def __init__(self, parent, step_id: str, step_label: str, **kwargs):
        super().__init__(parent, **kwargs)
        self.step_id = step_id
        self.configure(corner_radius=8, fg_color=("gray90", "gray17"))
        
        # Status icon area
        self.status_label = ctk.CTkLabel(
            self,
            text="○",
            font=ctk.CTkFont(size=18),
            width=30,
            text_color=("gray50", "gray60"),
        )
        self.status_label.pack(side="left", padx=(15, 10), pady=15)
        
        # Step info
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, pady=15)
        
        self.title_label = ctk.CTkLabel(
            info_frame,
            text=step_label,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        self.title_label.pack(fill="x")
        
        self.detail_label = ctk.CTkLabel(
            info_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray60"),
            anchor="w",
        )
        self.detail_label.pack(fill="x")
        
        self.state = "pending"
        self._spinner_idx = 0
        self._spinner_id = None
    
    def set_state(self, state: str, detail: str = ""):
        """Set step state: pending, running, complete, error."""
        self.state = state
        self.detail_label.configure(text=detail)
        
        if self._spinner_id:
            self.after_cancel(self._spinner_id)
            self._spinner_id = None
        
        if state == "pending":
            self.status_label.configure(text="○", text_color=("gray50", "gray60"))
            self.configure(fg_color=("gray90", "gray17"))
        elif state == "running":
            self.configure(fg_color=("gray85", "gray20"))
            self._animate_spinner()
        elif state == "complete":
            self.status_label.configure(text="✓", text_color=("green", "#4ade80"))
            self.configure(fg_color=("gray90", "gray17"))
        elif state == "error":
            self.status_label.configure(text="✗", text_color=("red", "#f87171"))
            self.configure(fg_color=("#fee2e2", "#450a0a"))
    
    def _animate_spinner(self):
        """Animate spinner for running state."""
        if self.state != "running":
            return
        spinner_chars = ["◐", "◓", "◑", "◒"]
        self.status_label.configure(
            text=spinner_chars[self._spinner_idx % 4],
            text_color=("#3b82f6", "#60a5fa")
        )
        self._spinner_idx += 1
        self._spinner_id = self.after(150, self._animate_spinner)


class PipelineProgressView(ctk.CTkFrame):
    """Pipeline progress panel showing all steps."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color="transparent")
        
        self.step_cards: Dict[str, PipelineStepCard] = {}
        
        # Header with progress indicator
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 15))
        
        header = ctk.CTkLabel(
            header_frame,
            text="Conversion Pipeline",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        header.pack(side="left")
        
        # Progress badge for multi-file processing
        self.progress_badge = ctk.CTkLabel(
            header_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#3b82f6", "#2563eb"),
            text_color="white",
            corner_radius=4,
            padx=10,
            pady=3,
        )
        # Don't pack initially - only shown during multi-file processing
        
        # Frame for step cards (so we can hide/show them together)
        self.steps_container = ctk.CTkFrame(self, fg_color="transparent")
        self.steps_container.pack(fill="x")
        
        # Create step cards inside the container
        for step_id, step_label in PIPELINE_STEPS:
            card = PipelineStepCard(self.steps_container, step_id, step_label)
            card.pack(fill="x", pady=5)
            self.step_cards[step_id] = card
        
        # Status/success message at bottom
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.pack(fill="x", pady=(20, 0))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Select PDF files and click Run to begin",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray60"),
        )
        self.status_label.pack()
        
        # Success banner (hidden initially)
        self.success_banner = ctk.CTkFrame(
            self,
            fg_color=("#dcfce7", "#14532d"),
            corner_radius=10,
        )
        # Contains icon and message
        self.success_inner = ctk.CTkFrame(self.success_banner, fg_color="transparent")
        self.success_inner.pack(fill="x", padx=15, pady=12)
        
        self.success_icon = ctk.CTkLabel(
            self.success_inner,
            text="✓",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=("#16a34a", "#4ade80"),
        )
        self.success_icon.pack(side="left", padx=(0, 10))
        
        self.success_text_frame = ctk.CTkFrame(self.success_inner, fg_color="transparent")
        self.success_text_frame.pack(side="left", fill="x", expand=True)
        
        self.success_title = ctk.CTkLabel(
            self.success_text_frame,
            text="Conversion Complete!",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#15803d", "#86efac"),
            anchor="w",
        )
        self.success_title.pack(fill="x")
        
        self.success_message = ctk.CTkLabel(
            self.success_text_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("#166534", "#a7f3d0"),
            anchor="w",
        )
        self.success_message.pack(fill="x")
        
        # Open folder button
        self.open_folder_btn = ctk.CTkButton(
            self.success_inner,
            text="📂 Open Folder",
            width=110,
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=("#16a34a", "#22c55e"),
            hover_color=("#15803d", "#16a34a"),
            text_color="white",
            command=self._open_output_folder,
        )
        self.open_folder_btn.pack(side="right", padx=(10, 0))
        
        # Store output path for opening folder
        self._output_path = ""
    
    def _open_output_folder(self):
        """Open the output folder in the system file explorer."""
        import subprocess
        import platform
        
        if not self._output_path or not os.path.exists(self._output_path):
            return
        
        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                subprocess.run(["open", self._output_path])
            elif system == "Windows":
                subprocess.run(["explorer", self._output_path])
            else:  # Linux
                subprocess.run(["xdg-open", self._output_path])
        except Exception as e:
            print(f"[Error] Could not open folder: {e}")
    
    def reset(self):
        """Reset all steps to pending and show step cards."""
        for card in self.step_cards.values():
            card.set_state("pending")
        self.status_label.configure(text="Select PDF files and click Run to begin")
        self.progress_badge.pack_forget()
        self.success_banner.pack_forget()
        self.steps_container.pack(fill="x", before=self.status_frame)
        self._output_path = ""
    
    def reset_steps_only(self):
        """Reset step cards to pending without clearing progress badge."""
        for card in self.step_cards.values():
            card.set_state("pending")
    
    def set_progress(self, current: int, total: int, filename: str = ""):
        """Show progress for multi-file processing."""
        if total > 1:
            self.progress_badge.configure(text=f"File {current}/{total}")
            self.progress_badge.pack(side="right")
        else:
            self.progress_badge.pack_forget()
    
    def set_step_state(self, step_id: str, state: str, detail: str = ""):
        """Update a specific step's state."""
        if step_id in self.step_cards:
            self.step_cards[step_id].set_state(state, detail)
    
    def set_status(self, message: str):
        """Set the overall status message."""
        self.status_label.configure(text=message)
    
    def show_success(self, message: str, output_path: str = ""):
        """Show an inline success banner and hide step cards for a cleaner view."""
        self._output_path = output_path
        self.success_message.configure(text=message)
        # Hide step cards and progress badge
        self.steps_container.pack_forget()
        self.progress_badge.pack_forget()
        # Show success banner
        self.success_banner.pack(fill="x", pady=(15, 0))
        self.status_label.configure(text="Select new PDFs to start another conversion")
    
    def hide_success(self):
        """Hide the success banner and restore step cards."""
        self.success_banner.pack_forget()
        self.steps_container.pack(fill="x", before=self.status_frame)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title(APP_TITLE)
        self.geometry("800x650")
        self.minsize(600, 500)
        
        # Set appearance
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        
        self.config_mgr = ConfigManager(CONFIG_PATH)
        try:
            self.config_mgr.ensure_config()
            self.config_data = self.config_mgr.load()
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to load config: {e}")
            self.config_data = {}
        
        self.selected_pdfs: List[str] = []
        self.output_dir: Optional[str] = None
        self.is_processing = False
        self.cancel_event = threading.Event()
        self.ui_queue = queue.Queue()
        
        self._build_ui()
        self.after(100, self._process_ui_queue)
    
    def _build_ui(self):
        # Main container with padding
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=30, pady=20)
        
        # ===== Header Section =====
        header = ctk.CTkFrame(container, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header,
            text=APP_TITLE,
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.pack(side="left")
        
        # Settings button
        settings_btn = ctk.CTkButton(
            header,
            text="⚙ Settings",
            width=100,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            command=self.open_settings,
        )
        settings_btn.pack(side="right")
        
        # ===== File Selection Section =====
        file_section = ctk.CTkFrame(container, fg_color="transparent")
        file_section.pack(fill="x", pady=(0, 15))
        
        file_row = ctk.CTkFrame(file_section, fg_color="transparent")
        file_row.pack(fill="x")
        
        select_btn = ctk.CTkButton(
            file_row,
            text="Select PDFs",
            width=120,
            command=self.select_pdfs,
        )
        select_btn.pack(side="left")
        
        self.files_label = ctk.CTkLabel(
            file_row,
            text="No files selected",
            font=ctk.CTkFont(size=13),
            text_color=("gray50", "gray60"),
        )
        self.files_label.pack(side="left", padx=15)
        
        # Cost badge on the right
        self.cost_badge = CostBadge(file_row)
        self.cost_badge.pack(side="right")
        
        # ===== Output Location Section =====
        output_section = ctk.CTkFrame(container, fg_color="transparent")
        output_section.pack(fill="x", pady=(0, 15))
        
        output_label = ctk.CTkLabel(
            output_section,
            text="Output:",
            font=ctk.CTkFont(size=13),
        )
        output_label.pack(side="left")
        
        # Output path in a styled frame that looks like a read-only field
        self.output_path_frame = ctk.CTkFrame(
            output_section,
            fg_color=("gray90", "gray20"),
            corner_radius=6,
        )
        self.output_path_frame.pack(side="left", fill="x", expand=True, padx=10)
        
        self.output_path_label = ctk.CTkLabel(
            self.output_path_frame,
            text="← Select PDFs to set output location",
            font=ctk.CTkFont(size=12, family="monospace"),
            text_color=("gray40", "gray70"),
            anchor="w",
        )
        self.output_path_label.pack(fill="x", padx=10, pady=6)
        
        # Store full path for tooltip
        self._full_output_path = ""
        
        # Add tooltip to show full path on hover
        ToolTip(self.output_path_frame, lambda: self._full_output_path)
        
        change_output_btn = ctk.CTkButton(
            output_section,
            text="Change",
            width=70,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            hover_color=("gray80", "gray30"),
            command=self.change_output_dir,
        )
        change_output_btn.pack(side="left")
        
        # ===== Options Row =====
        options_section = ctk.CTkFrame(container, fg_color="transparent")
        options_section.pack(fill="x", pady=(0, 20))
        
        # Citation style
        ctk.CTkLabel(options_section, text="Citations:", font=ctk.CTkFont(size=13)).pack(side="left")
        self.citation_var = ctk.StringVar(value=self.config_data.get('citation_style', 'Ignore'))
        citation_menu = ctk.CTkOptionMenu(
            options_section,
            variable=self.citation_var,
            values=["Ignore", "Subtle Mention"],
            width=130,
        )
        citation_menu.pack(side="left", padx=(5, 20))
        
        # Conversion mode
        ctk.CTkLabel(options_section, text="Mode:", font=ctk.CTkFont(size=13)).pack(side="left")
        self.mode_var = ctk.StringVar(value=self.config_data.get('conversion_mode', 'Summarized'))
        mode_menu = ctk.CTkOptionMenu(
            options_section,
            variable=self.mode_var,
            values=["Summarized", "Verbatim"],
            width=120,
        )
        mode_menu.pack(side="left", padx=5)
        
        # ===== Action Buttons (pack first to ensure visibility) =====
        action_bar = ctk.CTkFrame(container, fg_color="transparent")
        action_bar.pack(side="bottom", fill="x", pady=(20, 0))
        
        self.cancel_btn = ctk.CTkButton(
            action_bar,
            text="Cancel",
            width=100,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            hover_color=("gray80", "gray30"),
            command=self.on_cancel,
            state="disabled",
        )
        self.cancel_btn.pack(side="left")
        
        self.run_btn = ctk.CTkButton(
            action_bar,
            text="▶ Run",
            width=150,
            height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.on_run,
        )
        self.run_btn.pack(side="right")
        
        # ===== Pipeline Progress Section (fills remaining space) =====
        self.pipeline_view = PipelineProgressView(container)
        self.pipeline_view.pack(fill="both", expand=True)
    
    def _format_path_display(self, path: str, max_len: int = 50) -> str:
        """Format a path for display, showing the most relevant parts."""
        if len(path) <= max_len:
            return path
        
        # Split into parts
        parts = path.split(os.sep)
        
        # Always show last 2-3 parts (most relevant)
        if len(parts) <= 3:
            return path
        
        # Show: .../parent/folder
        end_parts = os.sep.join(parts[-2:])
        if len(end_parts) + 4 <= max_len:
            return f"...{os.sep}{end_parts}"
        else:
            # Just show the last folder
            return f"...{os.sep}{parts[-1]}"
    
    def _update_output_display(self, path: str):
        """Update the output path display with formatted path."""
        self._full_output_path = path
        display_path = self._format_path_display(path)
        self.output_path_label.configure(
            text=display_path,
            text_color=("gray20", "gray80"),
        )
    
    def select_pdfs(self):
        """Open file dialog to select PDF files."""
        paths = filedialog.askopenfilenames(
            title="Select PDF Files",
            filetypes=[("PDF files", "*.pdf")]
        )
        if paths:
            self.selected_pdfs = list(paths)
            self.cost_badge.clear()
            
            # Check page counts and warn about PDFs over the limit
            self._check_pdf_page_limits()
            
            # Update display
            if len(self.selected_pdfs) == 1:
                self.files_label.configure(text=os.path.basename(self.selected_pdfs[0]))
            else:
                self.files_label.configure(text=f"{len(self.selected_pdfs)} PDFs selected")
            
            # Auto-set output directory
            first_pdf_dir = os.path.dirname(self.selected_pdfs[0])
            self.output_dir = os.path.join(first_pdf_dir, "audio_papers")
            self._update_output_display(self.output_dir)
            
            # Reset pipeline view
            self.pipeline_view.reset()
            
            # Estimate cost in background
            self._estimate_cost_async()
    
    def _check_pdf_page_limits(self):
        """Check page counts of selected PDFs and warn about any over the limit."""
        if not self.selected_pdfs:
            return
        
        # Get configured page limit
        max_pages = self.config_data.get('max_pdf_pages', DEFAULT_MAX_PDF_PAGES)
        
        over_limit = []
        for pdf_path in self.selected_pdfs:
            try:
                page_count = get_pdf_page_count(pdf_path)
                if page_count > max_pages:
                    filename = os.path.basename(pdf_path)
                    over_limit.append((filename, page_count))
            except Exception as e:
                print(f"[Warning] Could not check page count for {pdf_path}: {e}")
        
        if over_limit:
            # Build warning message
            if len(over_limit) == 1:
                filename, pages = over_limit[0]
                msg = (
                    f'"{filename}" has {pages} pages, which exceeds the {max_pages}-page limit.\n\n'
                    f"This file will fail during conversion.\n\n"
                    f"💡 You can adjust the page limit in Settings."
                )
            else:
                msg = f"The following {len(over_limit)} files exceed the {max_pages}-page limit:\n\n"
                for filename, pages in over_limit:
                    msg += f"• {filename} ({pages} pages)\n"
                msg += f"\nThese files will fail during conversion.\n\n💡 You can adjust the page limit in Settings."
            
            messagebox.showwarning("Page Limit Exceeded", msg)
    
    def _estimate_cost_async(self):
        """Estimate cost in background thread for both modes."""
        if not self.selected_pdfs:
            return
        
        self.cost_badge.set_loading(True)
        
        # Get configured max pages
        max_pages = self.config_data.get('max_pdf_pages', DEFAULT_MAX_PDF_PAGES)
        
        def estimate():
            try:
                total_tokens = 0
                total_raw_chars = 0
                valid_count = 0
                skipped_count = 0
                
                for pdf_path in self.selected_pdfs:
                    filename = os.path.basename(pdf_path)
                    try:
                        print(f"[Cost] Estimating tokens for: {filename}")
                        tokens = estimate_tokens(pdf_path, max_pages=max_pages)
                        total_tokens += tokens
                        # Raw chars ~ tokens * 4 (heuristic from tokenization)
                        total_raw_chars += tokens * 4
                        valid_count += 1
                    except ValueError as e:
                        # PDF exceeds page limit, skip it
                        print(f"[Cost] Skipping {filename}: {e}")
                        skipped_count += 1
                        continue
                
                if valid_count == 0:
                    print(f"[Cost] No valid PDFs to estimate (all {skipped_count} exceed page limit)")
                    self.ui_queue.put(("cost_error", "All PDFs exceed page limit"))
                    return
                
                # Gemini processing cost (same for both modes)
                gemini_cost, text_model = self.config_mgr.estimate_gemini_cost(total_tokens)
                
                # Summarized mode: output is ~30% of input (condensed)
                summarized_chars = int(total_raw_chars * 0.3)
                summarized_tts_cost, tts_provider, tts_model = self.config_mgr.estimate_tts_cost(summarized_chars)
                summarized_total = gemini_cost + summarized_tts_cost
                
                # Verbatim mode: output is ~90% of input (cleaned but full text)
                verbatim_chars = int(total_raw_chars * 0.9)
                verbatim_tts_cost, _, _ = self.config_mgr.estimate_tts_cost(verbatim_chars)
                verbatim_total = gemini_cost + verbatim_tts_cost
                
                print(f"[Cost] Text model: {text_model} | TTS: {tts_provider} ({tts_model})")
                print(f"[Cost] Input: {total_tokens} tokens, {total_raw_chars} raw chars")
                print(f"[Cost] Summarized: ~{summarized_chars} chars, total ~${summarized_total:.4f}")
                print(f"[Cost] Verbatim: ~{verbatim_chars} chars, total ~${verbatim_total:.4f}")
                if skipped_count > 0:
                    print(f"[Cost] Note: {skipped_count} PDF(s) skipped (exceed page limit)")
                
                self.ui_queue.put(("cost_estimated", (summarized_total, verbatim_total)))
            except Exception as e:
                print(f"[Cost] Estimation error: {e}")
                self.ui_queue.put(("cost_error", str(e)))
        
        threading.Thread(target=estimate, daemon=True).start()
    
    def change_output_dir(self):
        """Let user choose a different output directory."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir = path
            self._update_output_display(path)
    
    def open_settings(self):
        """Open settings dialog."""
        SettingsWindow(self, self.config_mgr, self._on_settings_saved)
    
    def _on_settings_saved(self, data):
        """Callback when settings are saved."""
        self.config_data = data
        self.citation_var.set(data.get('citation_style', 'Ignore'))
        self.mode_var.set(data.get('conversion_mode', 'Summarized'))
        
        # Re-check page limits and re-estimate cost if PDFs are selected
        if self.selected_pdfs:
            self._check_pdf_page_limits()
            self._estimate_cost_async()
    
    def on_run(self):
        """Start the full conversion pipeline."""
        if self.is_processing:
            return
        
        if not self.selected_pdfs:
            messagebox.showwarning("No Files", "Please select one or more PDF files first.")
            return
        
        # Validate API key
        cfg = self.config_mgr.load()
        if not cfg.get("gemini_api_key"):
            messagebox.showerror("API Key Missing", "Please set your Gemini API key in Settings.")
            return
        
        # Ensure output directory
        if not self.output_dir:
            first_pdf_dir = os.path.dirname(self.selected_pdfs[0])
            self.output_dir = os.path.join(first_pdf_dir, "audio_papers")
        
        # Create output dir if needed
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[Pipeline] Output directory: {self.output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create output directory: {e}")
            return
        
        # Start processing
        self.is_processing = True
        self.cancel_event.clear()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.pipeline_view.reset()
        self.pipeline_view.hide_success()
        
        # Run in background thread
        threading.Thread(
            target=self._run_pipeline,
            args=(self.selected_pdfs.copy(), self.output_dir),
            daemon=True,
        ).start()
    
    def _run_pipeline(self, pdf_paths: List[str], output_dir: str):
        """Execute the full conversion pipeline for all PDFs."""
        total = len(pdf_paths)
        
        def send(event: str, payload=None):
            self.ui_queue.put((event, payload))
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] Starting conversion of {total} PDF(s)")
        print(f"{'='*60}\n")
        
        cfg = self.config_mgr.load()
        api_key = cfg.get("gemini_api_key", "")
        model_name = cfg.get("model_name", "gemini-2.0-flash")
        citation_style = self.citation_var.get()
        conversion_mode = self.mode_var.get()
        max_pdf_pages = cfg.get("max_pdf_pages", DEFAULT_MAX_PDF_PAGES)
        
        success_count = 0
        
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            if self.cancel_event.is_set():
                send("cancelled")
                return
            
            filename = os.path.basename(pdf_path)
            base_name = os.path.splitext(filename)[0]
            
            print(f"\n[Pipeline] Processing {idx}/{total}: {filename}")
            print("-" * 40)
            
            # Reset steps and show progress for multi-file processing
            send("new_file", (idx, total, filename))
            send("status", f"Processing: {filename}")
            
            try:
                # Step 1: Parse PDF (now returns image locations, not actual images)
                send("step_update", ("parsing", "running", f"Reading {filename}..."))
                print(f"[Parsing] Extracting text and image locations from PDF...")
                
                text, image_locations = parse_pdf(pdf_path, max_pages=max_pdf_pages)
                
                print(f"[Parsing] Extracted {len(text)} characters, {len(image_locations)} image locations")
                send("step_update", ("parsing", "complete", f"{len(text):,} chars, {len(image_locations)} figures"))
                
                if self.cancel_event.is_set():
                    send("cancelled")
                    return
                
                # Step 2: Process with Gemini (images extracted lazily as needed)
                send("step_update", ("structuring", "running", f"{'Summarizing' if conversion_mode == 'Summarized' else 'Cleaning'} with Gemini..."))
                
                if conversion_mode == "Verbatim":
                    print(f"[Gemini] Cleaning text with image descriptions (verbatim mode)...")
                    script = clean_with_gemini(
                        text=text,
                        pdf_path=pdf_path,
                        image_locations=image_locations,
                        api_key=api_key,
                        model_name=model_name,
                        citation_style=citation_style,
                        cancel_event=self.cancel_event,
                        log=lambda m: print(f"[Gemini] {m}"),
                    )
                else:
                    print(f"[Gemini] Structuring content (summarized mode)...")
                    script = structure_with_gemini(
                        text=text,
                        pdf_path=pdf_path,
                        image_locations=image_locations,
                        api_key=api_key,
                        model_name=model_name,
                        citation_style=citation_style,
                        cancel_event=self.cancel_event,
                        log=lambda m: print(f"[Gemini] {m}"),
                    )
                
                print(f"[Gemini] Generated script: {len(script)} characters")
                send("step_update", ("structuring", "complete", f"Script: {len(script):,} chars"))
                
                if self.cancel_event.is_set():
                    send("cancelled")
                    return
                
                # Step 3: Generate Audio (streaming to disk for memory efficiency)
                send("step_update", ("tts", "running", "Synthesizing speech..."))
                print(f"[TTS] Starting audio generation (streaming to disk)...")
                
                mp3_path = os.path.join(output_dir, f"{base_name}.mp3")
                
                # Use streaming audio generation for better memory efficiency
                # Allow fallback to Gemini if Inworld quota exceeded
                allow_fallback = cfg.get('tts_fallback_enabled', True)
                
                # Custom log callback that extracts progress percentage
                import re
                def tts_log(m):
                    print(f"[TTS] {m}")
                    # Parse "chunk X/Y" pattern to show percentage
                    match = re.search(r'chunk\s+(\d+)/(\d+)', m, re.IGNORECASE)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        percent = int((current / total) * 100)
                        send("step_update", ("tts", "running", f"Generating audio... {percent}% ({current}/{total} chunks)"))
                
                file_size = ConversionWorker.generate_audio_streaming(
                    script,
                    mp3_path,
                    self.config_mgr,
                    self.cancel_event,
                    log=tts_log,
                    allow_fallback=allow_fallback,
                )
                
                if file_size is None:
                    if self.cancel_event.is_set():
                        send("cancelled")
                        return
                    raise RuntimeError("Audio generation returned no data")
                
                audio_size_mb = file_size / (1024 * 1024)
                print(f"[TTS] Generated audio: {audio_size_mb:.2f} MB")
                send("step_update", ("tts", "complete", f"Audio: {audio_size_mb:.2f} MB"))
                
                if self.cancel_event.is_set():
                    send("cancelled")
                    return
                
                # Step 4: Save script (MP3 already saved by streaming)
                send("step_update", ("saving", "running", "Saving script..."))
                
                txt_gz_path = os.path.join(output_dir, f"{base_name}.txt.gz")
                
                # Save gzip-compressed text
                print(f"[Save] Writing compressed text: {txt_gz_path}")
                with gzip.open(txt_gz_path, 'wt', encoding='utf-8') as f:
                    f.write(script)
                
                send("step_update", ("saving", "complete", f"Saved to {output_dir}"))
                print(f"[Save] Files saved successfully")
                
                success_count += 1
                
            except InterruptedError:
                send("cancelled")
                return
            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")
                import traceback
                traceback.print_exc()
                send("step_error", str(e))
                send("error", f"Error processing {filename}: {e}")
                return
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] Completed: {success_count}/{total} files processed")
        print(f"[Pipeline] Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        # Send completion with message and output path
        send("complete", (f"Successfully converted {success_count} file(s)", output_dir))
    
    def on_cancel(self):
        """Cancel the current operation."""
        if self.is_processing:
            print("[Pipeline] Cancellation requested by user")
            self.cancel_event.set()
            self.pipeline_view.set_status("Cancelling...")
    
    def _process_ui_queue(self):
        """Process events from background threads."""
        try:
            while True:
                event, payload = self.ui_queue.get_nowait()
                
                if event == "status":
                    self.pipeline_view.set_status(payload)
                
                elif event == "new_file":
                    idx, total, filename = payload
                    # Reset all step cards for the new file
                    self.pipeline_view.reset_steps_only()
                    # Update progress indicator
                    self.pipeline_view.set_progress(idx, total, filename)
                    # Hide any previous success banner
                    self.pipeline_view.hide_success()
                
                elif event == "step_update":
                    step_id, state, detail = payload
                    self.pipeline_view.set_step_state(step_id, state, detail)
                
                elif event == "step_error":
                    # Mark current step as error
                    for step_id, card in self.pipeline_view.step_cards.items():
                        if card.state == "running":
                            card.set_state("error", payload)
                            break
                
                elif event == "cost_estimated":
                    summarized_cost, verbatim_cost = payload
                    self.cost_badge.set_cost(summarized_cost, verbatim_cost)
                
                elif event == "cost_error":
                    self.cost_badge.clear()
                
                elif event == "error":
                    self.is_processing = False
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    messagebox.showerror("Error", str(payload))
                
                elif event == "cancelled":
                    self.is_processing = False
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.pipeline_view.set_status("Operation cancelled")
                
                elif event == "complete":
                    self.is_processing = False
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    
                    # Unpack message and output path
                    message, output_path = payload
                    
                    # Reset file selection for next run
                    self.selected_pdfs = []
                    self.files_label.configure(text="No files selected")
                    self.cost_badge.clear()
                    
                    # Show success with output path (for "Open Folder" button)
                    self.pipeline_view.show_success(
                        f"{message}\n📁 Saved to: {output_path}",
                        output_path=output_path
                    )
        
        except queue.Empty:
            pass
        
        self.after(100, self._process_ui_queue)


class SettingsWindow(ctk.CTkToplevel):
    """Settings dialog window with clear API key status and provider selection."""
    
    def __init__(self, parent, config_mgr: ConfigManager, on_save):
        super().__init__(parent)
        
        self.title("Settings")
        self.geometry("580x720")
        self.minsize(500, 600)
        self.resizable(True, True)
        
        self.config_mgr = config_mgr
        self.on_save = on_save
        self._audio_process = None  # Track currently playing audio
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        data = {}
        try:
            data = self.config_mgr.load()
        except Exception:
            pass
        
        self.data = data
        
        # Use CTkScrollableFrame with explicit height to enable scrolling
        self.scrollable = ctk.CTkScrollableFrame(
            self, 
            fg_color="transparent",
            scrollbar_button_color=("gray70", "gray30"),
            scrollbar_button_hover_color=("gray60", "gray40"),
        )
        self.scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Bind mouse wheel events for better scroll behavior
        self._bind_scroll_events(self.scrollable)
        
        container = self.scrollable
        
        # =====================================================================
        # API KEYS SECTION - Always visible at top
        # =====================================================================
        keys_frame = ctk.CTkFrame(container, fg_color=("gray92", "gray18"), corner_radius=10)
        keys_frame.pack(fill="x", pady=(0, 15))
        
        keys_inner = ctk.CTkFrame(keys_frame, fg_color="transparent")
        keys_inner.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(
            keys_inner,
            text="🔑 API Keys",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            keys_inner,
            text="Both keys may be needed depending on your TTS provider choice",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray60"),
        ).pack(anchor="w", pady=(2, 12))
        
        # --- Gemini API Key ---
        gemini_key_frame = ctk.CTkFrame(keys_inner, fg_color="transparent")
        gemini_key_frame.pack(fill="x", pady=5)
        
        gemini_key_header = ctk.CTkFrame(gemini_key_frame, fg_color="transparent")
        gemini_key_header.pack(fill="x")
        
        ctk.CTkLabel(
            gemini_key_header, 
            text="Gemini API Key",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=120,
            anchor="w"
        ).pack(side="left")
        
        # Status indicator for Gemini
        existing_gemini_key = data.get('gemini_api_key', '')
        env_gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if existing_gemini_key:
            if env_gemini_key and existing_gemini_key == env_gemini_key:
                gemini_status = "✓ From .env.local"
            else:
                gemini_status = "✓ Configured"
            gemini_color = ("#22c55e", "#4ade80")
        else:
            gemini_status = "✗ Required"
            gemini_color = ("#ef4444", "#f87171")
        self.gemini_status_label = ctk.CTkLabel(
            gemini_key_header,
            text=gemini_status,
            font=ctk.CTkFont(size=11),
            text_color=gemini_color,
        )
        self.gemini_status_label.pack(side="left", padx=10)
        
        ctk.CTkLabel(
            gemini_key_header,
            text="(always needed for text processing)",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        ).pack(side="left")
        
        self.api_key_entry = ctk.CTkEntry(
            gemini_key_frame,
            placeholder_text="Enter Gemini API key (or set GEMINI_API_KEY in .env.local)",
            show="•",
        )
        self.api_key_entry.pack(fill="x", pady=(5, 0))
        if existing_gemini_key:
            self.api_key_entry.insert(0, existing_gemini_key)
        self.api_key_entry.bind("<KeyRelease>", self._update_key_status)
        
        # --- Inworld API Key ---
        inworld_key_frame = ctk.CTkFrame(keys_inner, fg_color="transparent")
        inworld_key_frame.pack(fill="x", pady=(12, 5))
        
        inworld_key_header = ctk.CTkFrame(inworld_key_frame, fg_color="transparent")
        inworld_key_header.pack(fill="x")
        
        ctk.CTkLabel(
            inworld_key_header, 
            text="Inworld API Key",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=120,
            anchor="w"
        ).pack(side="left")
        
        # Status indicator for Inworld
        existing_inworld_key = data.get('inworld_api_key', '')
        env_inworld_key = os.environ.get("INWORLD_API_KEY", "")
        
        if existing_inworld_key:
            if env_inworld_key and existing_inworld_key == env_inworld_key:
                inworld_status = "✓ From .env.local"
            else:
                inworld_status = "✓ Configured"
            inworld_color = ("#22c55e", "#4ade80")
        else:
            inworld_status = "○ Optional"
            inworld_color = ("gray50", "gray60")
        
        self.inworld_status_label = ctk.CTkLabel(
            inworld_key_header,
            text=inworld_status,
            font=ctk.CTkFont(size=11),
            text_color=inworld_color,
        )
        self.inworld_status_label.pack(side="left", padx=10)
        
        self.inworld_needed_label = ctk.CTkLabel(
            inworld_key_header,
            text="(needed if using Inworld TTS)",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        )
        self.inworld_needed_label.pack(side="left")
        
        self.inworld_api_key_entry = ctk.CTkEntry(
            inworld_key_frame,
            placeholder_text="Enter Inworld API key (or set INWORLD_API_KEY in .env.local)",
            show="•",
        )
        self.inworld_api_key_entry.pack(fill="x", pady=(5, 0))
        if existing_inworld_key:
            self.inworld_api_key_entry.insert(0, existing_inworld_key)
        self.inworld_api_key_entry.bind("<KeyRelease>", self._update_key_status)
        
        # =====================================================================
        # TTS PROVIDER SECTION
        # =====================================================================
        tts_frame = ctk.CTkFrame(container, fg_color=("gray92", "gray18"), corner_radius=10)
        tts_frame.pack(fill="x", pady=(0, 15))
        
        tts_inner = ctk.CTkFrame(tts_frame, fg_color="transparent")
        tts_inner.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(
            tts_inner,
            text="🔊 Text-to-Speech",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w")
        
        # Provider selection with radio-style buttons
        provider_select_frame = ctk.CTkFrame(tts_inner, fg_color="transparent")
        provider_select_frame.pack(fill="x", pady=(10, 5))
        
        self.provider_var = ctk.StringVar(value=data.get('tts_provider', 'inworld'))
        
        # Inworld option
        self.inworld_radio = ctk.CTkRadioButton(
            provider_select_frame,
            text="Inworld AI",
            variable=self.provider_var,
            value="inworld",
            font=ctk.CTkFont(size=13),
            command=self._on_provider_change,
        )
        self.inworld_radio.pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(
            provider_select_frame,
            text="(free tier available)",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        ).pack(side="left", padx=(0, 30))
        
        # Gemini option
        self.gemini_radio = ctk.CTkRadioButton(
            provider_select_frame,
            text="Gemini TTS",
            variable=self.provider_var,
            value="gemini",
            font=ctk.CTkFont(size=13),
            command=self._on_provider_change,
        )
        self.gemini_radio.pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(
            provider_select_frame,
            text="(uses Gemini key)",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        ).pack(side="left")
        
        # Fallback option
        fallback_frame = ctk.CTkFrame(tts_inner, fg_color="transparent")
        fallback_frame.pack(fill="x", pady=(8, 0))
        
        self.fallback_var = ctk.BooleanVar(value=data.get('tts_fallback_enabled', True))
        self.fallback_checkbox = ctk.CTkCheckBox(
            fallback_frame,
            text="Auto-fallback to Gemini if Inworld quota exceeded",
            variable=self.fallback_var,
            font=ctk.CTkFont(size=11),
        )
        self.fallback_checkbox.pack(side="left")
        
        ctk.CTkLabel(
            fallback_frame,
            text="(recommended)",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        ).pack(side="left", padx=5)
        
        # =====================================================================
        # VOICE SETTINGS - Tabbed view
        # =====================================================================
        voice_frame = ctk.CTkFrame(container, fg_color=("gray92", "gray18"), corner_radius=10)
        voice_frame.pack(fill="x", pady=(0, 15))
        
        voice_inner = ctk.CTkFrame(voice_frame, fg_color="transparent")
        voice_inner.pack(fill="x", padx=15, pady=15)
        
        # Header shows which provider is active
        voice_header = ctk.CTkFrame(voice_inner, fg_color="transparent")
        voice_header.pack(fill="x")
        
        ctk.CTkLabel(
            voice_header,
            text="🎙️ Voice Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left")
        
        self.active_provider_badge = ctk.CTkLabel(
            voice_header,
            text=f"Using: {TTS_PROVIDERS.get(self.provider_var.get(), '')}",
            font=ctk.CTkFont(size=11),
            fg_color=("#3b82f6", "#2563eb"),
            text_color="white",
            corner_radius=4,
            padx=8,
            pady=2,
        )
        self.active_provider_badge.pack(side="left", padx=10)
        
        # Preview text section
        preview_text_frame = ctk.CTkFrame(voice_inner, fg_color="transparent")
        preview_text_frame.pack(fill="x", pady=(10, 5))
        
        preview_header = ctk.CTkFrame(preview_text_frame, fg_color="transparent")
        preview_header.pack(fill="x")
        
        ctk.CTkLabel(
            preview_header,
            text="Preview Text:",
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).pack(side="left")
        
        # Custom indicator (hidden initially)
        self.custom_indicator = ctk.CTkLabel(
            preview_header,
            text="✎ Modified",
            font=ctk.CTkFont(size=10),
            text_color=("#f59e0b", "#fbbf24"),
        )
        
        # Built-in presets
        self.builtin_presets = {
            "🎭 Dramatic": "In a world full of equations, be the one who explains them with style and a hint of drama.",
            "🤓 Nerdy": "Did you know that octopuses have three hearts and blue blood? Science is wonderfully weird!",
            "😎 Cool": "They said I couldn't turn a research paper into audio gold. They were magnificently wrong.",
        }
        
        # Load custom presets from config
        self.custom_presets = data.get('custom_voice_presets', {})
        
        # Get saved preview text
        import hashlib as _hashlib
        default_phrase_idx = int(_hashlib.md5(b"default").hexdigest(), 16) % len(VOICE_PREVIEW_PHRASES)
        saved_preview_text = data.get('voice_preview_text', VOICE_PREVIEW_PHRASES[default_phrase_idx])
        
        self.preview_text_entry = ctk.CTkTextbox(
            preview_text_frame,
            height=50,
            font=ctk.CTkFont(size=11),
        )
        self.preview_text_entry.pack(fill="x", pady=(3, 0))
        self.preview_text_entry.insert("1.0", saved_preview_text)
        
        # Bind text changes to detect modifications
        self.preview_text_entry.bind("<KeyRelease>", self._on_preview_text_change)
        
        # Preset buttons frame
        presets_frame = ctk.CTkFrame(preview_text_frame, fg_color="transparent")
        presets_frame.pack(fill="x", pady=(5, 0))
        
        # Built-in presets row
        builtin_row = ctk.CTkFrame(presets_frame, fg_color="transparent")
        builtin_row.pack(fill="x")
        
        ctk.CTkLabel(
            builtin_row,
            text="Presets:",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
            width=50,
        ).pack(side="left")
        
        self.preset_buttons = {}
        for label, phrase in self.builtin_presets.items():
            btn = ctk.CTkButton(
                builtin_row,
                text=label,
                width=75,
                height=22,
                font=ctk.CTkFont(size=10),
                fg_color="transparent",
                border_width=1,
                text_color=("gray30", "gray70"),
                hover_color=("gray85", "gray25"),
                command=lambda p=phrase, l=label: self._set_preview_text(p, l),
            )
            btn.pack(side="left", padx=2)
            self.preset_buttons[label] = btn
        
        # Custom presets row (shown only if custom presets exist)
        self.custom_row = ctk.CTkFrame(presets_frame, fg_color="transparent")
        
        ctk.CTkLabel(
            self.custom_row,
            text="Custom:",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
            width=50,
        ).pack(side="left")
        
        self.custom_buttons_frame = ctk.CTkFrame(self.custom_row, fg_color="transparent")
        self.custom_buttons_frame.pack(side="left", fill="x", expand=True)
        
        # Populate custom preset buttons
        self._rebuild_custom_preset_buttons()
        
        # Save as custom button (shown when text is modified)
        self.save_custom_frame = ctk.CTkFrame(presets_frame, fg_color="transparent")
        
        self.custom_name_entry = ctk.CTkEntry(
            self.save_custom_frame,
            placeholder_text="Name your preset...",
            width=150,
            height=24,
            font=ctk.CTkFont(size=10),
        )
        self.custom_name_entry.pack(side="left", padx=(50, 5))
        
        self.save_custom_btn = ctk.CTkButton(
            self.save_custom_frame,
            text="💾 Save Custom",
            width=90,
            height=24,
            font=ctk.CTkFont(size=10),
            fg_color=("#10b981", "#059669"),
            hover_color=("#059669", "#047857"),
            command=self._save_custom_preset,
        )
        self.save_custom_btn.pack(side="left")
        
        # Track current preset
        self.current_preset = None
        self._check_if_preset_matches()
        
        # Inworld voice settings
        self.inworld_voice_frame = ctk.CTkFrame(voice_inner, fg_color="transparent")
        
        inworld_voice_row = ctk.CTkFrame(self.inworld_voice_frame, fg_color="transparent")
        inworld_voice_row.pack(fill="x", pady=5)
        
        ctk.CTkLabel(inworld_voice_row, text="Voice:", width=70, anchor="w").pack(side="left")
        
        # Try to fetch voices from API, fall back to static list
        self.inworld_voices = INWORLD_TTS_VOICES.copy()
        existing_inworld_api_key = data.get('inworld_api_key', '')
        if existing_inworld_api_key:
            try:
                fetched = fetch_inworld_voices(existing_inworld_api_key)
                if fetched:
                    self.inworld_voices = fetched
            except Exception as e:
                print(f"[Settings] Could not fetch Inworld voices: {e}")
        
        inworld_voice_options = [f"{name} ({style})" for name, style in self.inworld_voices.items()]
        current_inworld_voice = data.get('inworld_voice_id', list(self.inworld_voices.keys())[0] if self.inworld_voices else 'Cove')
        current_inworld_display = next(
            (v for v in inworld_voice_options if v.startswith(current_inworld_voice + " ")),
            inworld_voice_options[0] if inworld_voice_options else "No voices available"
        )
        
        self.inworld_voice_var = ctk.StringVar(value=current_inworld_display)
        self.inworld_voice_menu = ctk.CTkOptionMenu(
            inworld_voice_row,
            variable=self.inworld_voice_var,
            values=inworld_voice_options if inworld_voice_options else ["No voices available"],
            width=200,
        )
        self.inworld_voice_menu.pack(side="left")
        
        # Buttons row for Refresh and Preview (separate line to prevent hiding)
        inworld_buttons_row = ctk.CTkFrame(self.inworld_voice_frame, fg_color="transparent")
        inworld_buttons_row.pack(fill="x", pady=(5, 0))
        
        # Spacer to align with voice dropdown
        ctk.CTkLabel(inworld_buttons_row, text="", width=70).pack(side="left")
        
        # Refresh button to reload voices from API
        self.refresh_voices_btn = ctk.CTkButton(
            inworld_buttons_row,
            text="🔄 Refresh Voices",
            width=110,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            border_width=1,
            text_color=("gray30", "gray70"),
            hover_color=("gray85", "gray25"),
            command=self._refresh_inworld_voices,
        )
        self.refresh_voices_btn.pack(side="left")
        
        # Preview button for Inworld
        self.inworld_preview_btn = ctk.CTkButton(
            inworld_buttons_row,
            text="▶ Preview",
            width=90,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color=("#6366f1", "#4f46e5"),
            hover_color=("#4f46e5", "#4338ca"),
            command=lambda: self._preview_voice("inworld"),
        )
        self.inworld_preview_btn.pack(side="left", padx=(10, 0))
        
        self.inworld_preview_status = ctk.CTkLabel(
            inworld_buttons_row,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        )
        self.inworld_preview_status.pack(side="left", padx=5)
        
        inworld_model_row = ctk.CTkFrame(self.inworld_voice_frame, fg_color="transparent")
        inworld_model_row.pack(fill="x", pady=5)
        
        ctk.CTkLabel(inworld_model_row, text="Quality:", width=70, anchor="w").pack(side="left")
        
        self.inworld_model_var = ctk.StringVar(value=data.get('inworld_model_id', 'inworld-tts-1'))
        ctk.CTkOptionMenu(
            inworld_model_row,
            variable=self.inworld_model_var,
            values=list(INWORLD_TTS_MODELS.keys()),
            width=180,
        ).pack(side="left")
        
        # Gemini voice settings
        self.gemini_voice_frame = ctk.CTkFrame(voice_inner, fg_color="transparent")
        
        gemini_voice_row = ctk.CTkFrame(self.gemini_voice_frame, fg_color="transparent")
        gemini_voice_row.pack(fill="x", pady=5)
        
        ctk.CTkLabel(gemini_voice_row, text="Voice:", width=70, anchor="w").pack(side="left")
        
        gemini_voice_options = [f"{name} ({style})" for name, style in GEMINI_TTS_VOICES.items()]
        current_gemini_voice = data.get('tts_voice_name', 'Kore')
        current_gemini_display = next(
            (v for v in gemini_voice_options if v.startswith(current_gemini_voice + " ")),
            gemini_voice_options[0]
        )
        
        self.gemini_voice_var = ctk.StringVar(value=current_gemini_display)
        self.gemini_voice_menu = ctk.CTkOptionMenu(
            gemini_voice_row,
            variable=self.gemini_voice_var,
            values=gemini_voice_options,
            width=220,
        )
        self.gemini_voice_menu.pack(side="left")
        
        # Preview button for Gemini
        self.gemini_preview_btn = ctk.CTkButton(
            gemini_voice_row,
            text="▶ Preview",
            width=80,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color=("#6366f1", "#4f46e5"),
            hover_color=("#4f46e5", "#4338ca"),
            command=lambda: self._preview_voice("gemini"),
        )
        self.gemini_preview_btn.pack(side="left", padx=(10, 0))
        
        self.gemini_preview_status = ctk.CTkLabel(
            gemini_voice_row,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        )
        self.gemini_preview_status.pack(side="left", padx=5)
        
        gemini_style_label = ctk.CTkLabel(
            self.gemini_voice_frame,
            text="Style prompt (optional):",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        gemini_style_label.pack(anchor="w", pady=(10, 3))
        
        self.style_textbox = ctk.CTkTextbox(self.gemini_voice_frame, height=50)
        self.style_textbox.pack(fill="x")
        self.style_textbox.insert("1.0", data.get('tts_style_prompt', ''))
        
        ctk.CTkLabel(
            self.gemini_voice_frame,
            text="e.g., 'Speak warmly and conversationally'",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        ).pack(anchor="w", pady=(2, 0))
        
        # Show appropriate voice frame
        self._update_voice_frame()
        
        # =====================================================================
        # CONVERSION PREFERENCES
        # =====================================================================
        prefs_frame = ctk.CTkFrame(container, fg_color=("gray92", "gray18"), corner_radius=10)
        prefs_frame.pack(fill="x", pady=(0, 15))
        
        prefs_inner = ctk.CTkFrame(prefs_frame, fg_color="transparent")
        prefs_inner.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(
            prefs_inner,
            text="⚙️ Conversion Defaults",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w")
        
        prefs_row = ctk.CTkFrame(prefs_inner, fg_color="transparent")
        prefs_row.pack(fill="x", pady=(10, 0))
        
        ctk.CTkLabel(prefs_row, text="Citation Style:", width=100, anchor="w").pack(side="left")
        self.citation_var = ctk.StringVar(value=self.data.get('citation_style', 'Ignore'))
        ctk.CTkOptionMenu(
            prefs_row,
            variable=self.citation_var,
            values=["Ignore", "Subtle Mention"],
            width=140,
        ).pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(prefs_row, text="Mode:", anchor="w").pack(side="left")
        self.mode_var = ctk.StringVar(value=self.data.get('conversion_mode', 'Summarized'))
        ctk.CTkOptionMenu(
            prefs_row,
            variable=self.mode_var,
            values=["Summarized", "Verbatim"],
            width=120,
        ).pack(side="left", padx=5)
        
        # Page limit row
        page_limit_row = ctk.CTkFrame(prefs_inner, fg_color="transparent")
        page_limit_row.pack(fill="x", pady=(10, 0))
        
        ctk.CTkLabel(page_limit_row, text="Max PDF Pages:", width=100, anchor="w").pack(side="left")
        
        # Get current value from config
        current_max_pages = self.data.get('max_pdf_pages', DEFAULT_MAX_PDF_PAGES)
        self.max_pages_var = ctk.IntVar(value=current_max_pages)
        
        self.max_pages_entry = ctk.CTkEntry(
            page_limit_row,
            textvariable=self.max_pages_var,
            width=60,
            justify="center",
        )
        self.max_pages_entry.pack(side="left")
        
        ctk.CTkLabel(
            page_limit_row, 
            text=f"pages  (default: {DEFAULT_MAX_PDF_PAGES})",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray60"),
        ).pack(side="left", padx=(5, 0))
        
        # =====================================================================
        # BUTTONS
        # =====================================================================
        button_row = ctk.CTkFrame(container, fg_color="transparent")
        button_row.pack(fill="x", pady=(10, 5))
        
        ctk.CTkButton(
            button_row,
            text="Cancel",
            width=100,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            command=self.destroy,
        ).pack(side="right", padx=(10, 0))
        
        ctk.CTkButton(
            button_row,
            text="Save Settings",
            width=120,
            command=self.save,
        ).pack(side="right")
        
        # Initial UI state
        self._update_voice_frame()
        self._update_key_status()
    
    def _bind_scroll_events(self, scrollable_frame):
        """Bind mouse wheel events to all children for better scroll behavior."""
        def on_mousewheel(event):
            # Get the canvas from the scrollable frame
            canvas = scrollable_frame._parent_canvas
            if platform.system() == 'Darwin':  # macOS
                canvas.yview_scroll(-1 * event.delta, "units")
            elif event.num == 4:  # Linux scroll up
                canvas.yview_scroll(-3, "units")
            elif event.num == 5:  # Linux scroll down
                canvas.yview_scroll(3, "units")
            else:  # Windows
                canvas.yview_scroll(-1 * (event.delta // 120), "units")
        
        # Bind to the scrollable frame and its internal canvas
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Button-4>", on_mousewheel)
        scrollable_frame.bind("<Button-5>", on_mousewheel)
        
        # Also bind to the parent window to catch events anywhere
        self.bind("<MouseWheel>", on_mousewheel)
        self.bind("<Button-4>", on_mousewheel)
        self.bind("<Button-5>", on_mousewheel)
    
    def _refresh_inworld_voices(self):
        """Refresh the Inworld voice list from the API."""
        api_key = self.inworld_api_key_entry.get().strip()
        if not api_key:
            # Show error briefly
            self.refresh_voices_btn.configure(text="❌ No API Key")
            self.after(2000, lambda: self.refresh_voices_btn.configure(text="🔄 Refresh Voices"))
            return
        
        # Show loading state
        self.refresh_voices_btn.configure(text="⏳ Loading...", state="disabled")
        
        def fetch():
            try:
                fetched = fetch_inworld_voices(api_key)
                if fetched:
                    self.inworld_voices = fetched
                    voice_options = [f"{name} ({style})" for name, style in fetched.items()]
                    
                    # Update menu on main thread
                    def update_menu():
                        current = self.inworld_voice_var.get()
                        self.inworld_voice_menu.configure(values=voice_options)
                        # Keep current selection if still valid
                        if current not in voice_options:
                            self.inworld_voice_var.set(voice_options[0])
                        self.refresh_voices_btn.configure(text="✓ Updated!", state="normal")
                        self.after(2000, lambda: self.refresh_voices_btn.configure(text="🔄 Refresh Voices"))
                    
                    self.after(0, update_menu)
                else:
                    self.after(0, lambda: self.refresh_voices_btn.configure(text="❌ Failed", state="normal"))
                    self.after(2000, lambda: self.refresh_voices_btn.configure(text="🔄 Refresh Voices"))
            except Exception as e:
                print(f"[Settings] Error refreshing voices: {e}")
                self.after(0, lambda: self.refresh_voices_btn.configure(text="❌ Error", state="normal"))
                self.after(2000, lambda: self.refresh_voices_btn.configure(text="🔄 Refresh Voices"))
        
        threading.Thread(target=fetch, daemon=True).start()
    
    def _on_provider_change(self, *args):
        """Update UI when TTS provider changes."""
        self._update_voice_frame()
        self._update_key_status()
    
    def _update_voice_frame(self):
        """Show the voice settings for the active provider."""
        provider = self.provider_var.get()
        
        # Update badge
        self.active_provider_badge.configure(text=f"Using: {TTS_PROVIDERS.get(provider, '')}")
        
        # Show appropriate voice frame
        if provider == "inworld":
            self.gemini_voice_frame.pack_forget()
            self.inworld_voice_frame.pack(fill="x", pady=(10, 0))
        else:
            self.inworld_voice_frame.pack_forget()
            self.gemini_voice_frame.pack(fill="x", pady=(10, 0))
    
    def _update_key_status(self, event=None):
        """Update key status indicators based on current values."""
        # Gemini key status
        gemini_key = self.api_key_entry.get().strip()
        if gemini_key:
            self.gemini_status_label.configure(
                text="✓ Configured",
                text_color=("#22c55e", "#4ade80")
            )
        else:
            self.gemini_status_label.configure(
                text="✗ Required",
                text_color=("#ef4444", "#f87171")
            )
        
        # Inworld key status
        inworld_key = self.inworld_api_key_entry.get().strip()
        env_key = os.environ.get("INWORLD_API_KEY", "")
        provider = self.provider_var.get()
        
        if inworld_key:
            if env_key and inworld_key == env_key:
                status = "✓ From .env.local"
            else:
                status = "✓ Configured"
            color = ("#22c55e", "#4ade80")
        elif provider == "inworld":
            status = "✗ Required"
            color = ("#ef4444", "#f87171")
        else:
            status = "○ Not needed"
            color = ("gray50", "gray60")
        
        self.inworld_status_label.configure(text=status, text_color=color)
        
        # Update "needed" label
        if provider == "inworld":
            self.inworld_needed_label.configure(text="(required for Inworld TTS)")
        else:
            self.inworld_needed_label.configure(text="(not needed if using Gemini)")
    
    def _set_preview_text(self, text: str, preset_label: str = None):
        """Set the preview text to a preset phrase."""
        self.preview_text_entry.delete("1.0", "end")
        self.preview_text_entry.insert("1.0", text)
        self.current_preset = preset_label
        self._update_preset_ui()
    
    def _on_preview_text_change(self, event=None):
        """Called when preview text is modified."""
        self._check_if_preset_matches()
    
    def _check_if_preset_matches(self):
        """Check if current text matches any preset."""
        current_text = self.preview_text_entry.get("1.0", "end").strip()
        
        # Check built-in presets
        for label, phrase in self.builtin_presets.items():
            if current_text == phrase:
                self.current_preset = label
                self._update_preset_ui()
                return
        
        # Check custom presets
        for label, phrase in self.custom_presets.items():
            if current_text == phrase:
                self.current_preset = label
                self._update_preset_ui()
                return
        
        # No match - it's custom/modified
        self.current_preset = None
        self._update_preset_ui()
    
    def _update_preset_ui(self):
        """Update UI to reflect current preset state."""
        # Update button highlighting
        for label, btn in self.preset_buttons.items():
            if label == self.current_preset:
                btn.configure(
                    fg_color=("#6366f1", "#4f46e5"),
                    text_color="white",
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=("gray30", "gray70"),
                )
        
        # Update custom buttons highlighting
        for widget in self.custom_buttons_frame.winfo_children():
            if hasattr(widget, '_preset_label'):
                if widget._preset_label == self.current_preset:
                    widget.configure(
                        fg_color=("#6366f1", "#4f46e5"),
                        text_color="white",
                    )
                else:
                    widget.configure(
                        fg_color="transparent",
                        text_color=("gray30", "gray70"),
                    )
        
        # Show/hide custom indicator and save button
        if self.current_preset is None:
            self.custom_indicator.pack(side="left", padx=10)
            self.save_custom_frame.pack(fill="x", pady=(5, 0))
        else:
            self.custom_indicator.pack_forget()
            self.save_custom_frame.pack_forget()
        
        # Show custom row if there are custom presets
        if self.custom_presets:
            self.custom_row.pack(fill="x", pady=(3, 0))
        else:
            self.custom_row.pack_forget()
    
    def _rebuild_custom_preset_buttons(self):
        """Rebuild the custom preset buttons."""
        # Clear existing buttons
        for widget in self.custom_buttons_frame.winfo_children():
            widget.destroy()
        
        # Create buttons for each custom preset
        for label, phrase in self.custom_presets.items():
            btn_frame = ctk.CTkFrame(self.custom_buttons_frame, fg_color="transparent")
            btn_frame.pack(side="left", padx=2)
            
            btn = ctk.CTkButton(
                btn_frame,
                text=f"★ {label}",
                width=80,
                height=22,
                font=ctk.CTkFont(size=10),
                fg_color="transparent",
                border_width=1,
                text_color=("gray30", "gray70"),
                hover_color=("gray85", "gray25"),
                command=lambda p=phrase, l=label: self._set_preview_text(p, l),
            )
            btn._preset_label = label
            btn.pack(side="left")
            
            # Delete button
            del_btn = ctk.CTkButton(
                btn_frame,
                text="×",
                width=18,
                height=22,
                font=ctk.CTkFont(size=10),
                fg_color="transparent",
                text_color=("#ef4444", "#f87171"),
                hover_color=("gray85", "gray25"),
                command=lambda l=label: self._delete_custom_preset(l),
            )
            del_btn.pack(side="left")
    
    def _save_custom_preset(self):
        """Save the current text as a custom preset."""
        name = self.custom_name_entry.get().strip()
        if not name:
            # Generate a default name
            name = f"Custom {len(self.custom_presets) + 1}"
        
        text = self.preview_text_entry.get("1.0", "end").strip()
        if not text:
            return
        
        # Add to custom presets
        self.custom_presets[name] = text
        self.current_preset = name
        
        # Clear the name entry
        self.custom_name_entry.delete(0, "end")
        
        # Rebuild UI
        self._rebuild_custom_preset_buttons()
        self._update_preset_ui()
    
    def _delete_custom_preset(self, label: str):
        """Delete a custom preset."""
        if label in self.custom_presets:
            del self.custom_presets[label]
            
            # If we were using this preset, mark as modified
            if self.current_preset == label:
                self.current_preset = None
            
            self._rebuild_custom_preset_buttons()
            self._update_preset_ui()
    
    def _preview_voice(self, provider: str):
        """Play a preview of the selected voice."""
        import hashlib
        
        # Get the preview text
        preview_text = self.preview_text_entry.get("1.0", "end").strip()
        if not preview_text:
            preview_text = VOICE_PREVIEW_PHRASES[0]
            self._set_preview_text(preview_text)
        
        # Create a short hash of the text for the filename
        text_hash = hashlib.md5(preview_text.encode()).hexdigest()[:8]
        
        # Get voice info
        if provider == "inworld":
            voice_display = self.inworld_voice_var.get()
            voice_id = voice_display.split(" (")[0] if " (" in voice_display else voice_display
            status_label = self.inworld_preview_status
            btn = self.inworld_preview_btn
            api_key = self.inworld_api_key_entry.get().strip()
            if not api_key:
                status_label.configure(text="No API key!", text_color=("#ef4444", "#f87171"))
                self.after(2000, lambda: status_label.configure(text=""))
                return
        else:  # gemini
            voice_display = self.gemini_voice_var.get()
            voice_id = voice_display.split(" (")[0] if " (" in voice_display else voice_display
            status_label = self.gemini_preview_status
            btn = self.gemini_preview_btn
            api_key = self.api_key_entry.get().strip()
            if not api_key:
                status_label.configure(text="No API key!", text_color=("#ef4444", "#f87171"))
                self.after(2000, lambda: status_label.configure(text=""))
                return
        
        # Sample file path includes text hash for dynamic caching
        VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
        voice_safe = voice_id.lower().replace(' ', '_')
        sample_file = VOICE_SAMPLES_DIR / f"{provider}_{voice_safe}_{text_hash}.mp3"
        
        # If sample exists, play it
        if sample_file.exists():
            self._play_audio(sample_file, status_label, btn)
            return
        
        # Generate sample in background
        status_label.configure(text="Generating...", text_color=("#f59e0b", "#fbbf24"))
        btn.configure(state="disabled")
        
        # Use the custom preview text
        phrase = preview_text
        
        def generate():
            try:
                if provider == "inworld":
                    self._generate_inworld_sample(voice_id, phrase, sample_file, api_key)
                else:
                    self._generate_gemini_sample(voice_id, phrase, sample_file, api_key)
                
                self.after(0, lambda: self._play_audio(sample_file, status_label, btn))
            except Exception as e:
                # Log full error to terminal
                import traceback
                print(f"\n[Voice Preview] ERROR generating {provider} sample:")
                print(f"[Voice Preview] Voice: {voice_id}")
                print(f"[Voice Preview] Error: {e}")
                traceback.print_exc()
                print()
                
                error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
                self.after(0, lambda: status_label.configure(
                    text=f"Error: {error_msg}", 
                    text_color=("#ef4444", "#f87171")
                ))
                self.after(0, lambda: btn.configure(state="normal"))
                self.after(3000, lambda: status_label.configure(text=""))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def _generate_inworld_sample(self, voice_id: str, text: str, output_path: Path, api_key: str):
        """Generate a voice sample using Inworld TTS."""
        import requests
        import base64
        
        print(f"[Voice Preview] Generating Inworld sample for voice: {voice_id}")
        
        url = "https://api.inworld.ai/tts/v1/voice"
        
        # Inworld uses Basic auth with base64-encoded credentials
        # The API key should already be the base64-encoded credential
        headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "text": text,
            "voiceId": voice_id,
            "modelId": "inworld-tts-1",
        }
        
        print(f"[Voice Preview] Sending request to Inworld API...")
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            
            # Log response status
            print(f"[Voice Preview] Response status: {resp.status_code}")
            
            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('message', str(error_data))
                except:
                    error_msg = resp.text[:500] if resp.text else "No error message"
                print(f"[Voice Preview] ERROR: {error_msg}")
                raise RuntimeError(f"Inworld API error ({resp.status_code}): {error_msg}")
            
            # Response is JSON with base64-encoded audio
            result = resp.json()
            
            if 'audioContent' not in result:
                print(f"[Voice Preview] ERROR: No audioContent in response")
                print(f"[Voice Preview] Response keys: {list(result.keys())}")
                raise RuntimeError(f"No audioContent in Inworld response")
            
            audio_bytes = base64.b64decode(result['audioContent'])
            print(f"[Voice Preview] Success! Generated {len(audio_bytes)} bytes of audio")
            output_path.write_bytes(audio_bytes)
            
        except requests.exceptions.Timeout:
            print(f"[Voice Preview] ERROR: Request timed out")
            raise RuntimeError("Inworld API request timed out")
        except requests.exceptions.ConnectionError as e:
            print(f"[Voice Preview] ERROR: Connection error - {e}")
            raise RuntimeError(f"Cannot connect to Inworld API: {e}")
    
    def _generate_gemini_sample(self, voice_id: str, text: str, output_path: Path, api_key: str):
        """Generate a voice sample using Gemini TTS."""
        from google import genai
        import wave
        import tempfile
        
        print(f"[Voice Preview] Generating Gemini sample for voice: {voice_id}")
        
        client = genai.Client(api_key=api_key)
        
        print(f"[Voice Preview] Sending request to Gemini TTS API...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=genai.types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai.types.SpeechConfig(
                    voice_config=genai.types.VoiceConfig(
                        prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(
                            voice_name=voice_id,
                        )
                    )
                ),
            ),
        )
        
        print(f"[Voice Preview] Got response from Gemini, extracting audio...")
        
        # Convert PCM to MP3
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        sample_rate = 24000
        
        print(f"[Voice Preview] Got {len(audio_data)} bytes of PCM audio, converting to MP3...")
        
        # Write WAV first
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
            with wave.open(wav_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
        
        # Convert to MP3 with ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", "128k", str(output_path)],
            capture_output=True,
        )
        os.unlink(wav_path)
        
        if result.returncode != 0:
            print(f"[Voice Preview] ffmpeg error: {result.stderr.decode()}")
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode()[:200]}")
        
        file_size = output_path.stat().st_size
        print(f"[Voice Preview] Success! Generated {file_size} bytes of audio")
    
    def _play_audio(self, file_path: Path, status_label, btn):
        """Play an audio file using system tools."""
        # Stop any currently playing audio
        if self._audio_process and self._audio_process.poll() is None:
            self._audio_process.terminate()
        
        status_label.configure(text="▶ Playing...", text_color=("#22c55e", "#4ade80"))
        btn.configure(state="normal", text="■ Stop")
        
        def play():
            try:
                if platform.system() == "Darwin":  # macOS
                    self._audio_process = subprocess.Popen(
                        ["afplay", str(file_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif platform.system() == "Windows":
                    # Use PowerShell on Windows
                    self._audio_process = subprocess.Popen(
                        ["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync()"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:  # Linux
                    self._audio_process = subprocess.Popen(
                        ["mpg123", "-q", str(file_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                
                self._audio_process.wait()
                self.after(0, lambda: status_label.configure(text=""))
                self.after(0, lambda: btn.configure(text="▶ Preview"))
            except Exception as e:
                self.after(0, lambda: status_label.configure(
                    text="Playback error", 
                    text_color=("#ef4444", "#f87171")
                ))
                self.after(0, lambda: btn.configure(text="▶ Preview"))
        
        threading.Thread(target=play, daemon=True).start()
    
    def save(self):
        """Save settings and close."""
        # Extract voice names from display
        inworld_voice_display = self.inworld_voice_var.get()
        inworld_voice_id = inworld_voice_display.split(" (")[0] if " (" in inworld_voice_display else inworld_voice_display
        
        gemini_voice_display = self.gemini_voice_var.get()
        gemini_voice_name = gemini_voice_display.split(" (")[0] if " (" in gemini_voice_display else gemini_voice_display
        
        # Validate and get max pages value
        try:
            max_pages = self.max_pages_var.get()
            if max_pages < 1:
                max_pages = DEFAULT_MAX_PDF_PAGES
        except (ValueError, tk.TclError):
            max_pages = DEFAULT_MAX_PDF_PAGES
        
        data = {
            'gemini_api_key': self.api_key_entry.get().strip(),
            'tts_provider': self.provider_var.get(),
            'tts_fallback_enabled': self.fallback_var.get(),
            'inworld_api_key': self.inworld_api_key_entry.get().strip(),
            'inworld_voice_id': inworld_voice_id,
            'inworld_model_id': self.inworld_model_var.get(),
            'tts_voice_name': gemini_voice_name,
            'tts_style_prompt': self.style_textbox.get("1.0", "end-1c").strip(),
            'voice_preview_text': self.preview_text_entry.get("1.0", "end-1c").strip(),
            'custom_voice_presets': self.custom_presets,
            'citation_style': self.citation_var.get(),
            'conversion_mode': self.mode_var.get(),
            'max_pdf_pages': max_pages,
        }
        
        try:
            self.config_mgr.save(data)
            self.on_save(data)
            self.destroy()
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
