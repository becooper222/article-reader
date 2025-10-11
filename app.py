#!/usr/bin/env python3
import threading
import queue
import json
import os
import tkinter as tk
from tkinter import Tk, Toplevel, filedialog, messagebox, StringVar, DoubleVar, BooleanVar
from tkinter import ttk

from config import ConfigManager
from processing import ConversionWorker, estimate_tokens

APP_TITLE = "Research Paper Audiobook Converter"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

class App:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry('1100x800')

        self.config_mgr = ConfigManager(CONFIG_PATH)
        try:
            self.config_mgr.ensure_config()
            self.config_data = self.config_mgr.load()
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to load config: {e}")
            self.config_data = {}

        self.selected_pdf = StringVar(value="")
        self.citation_style = StringVar(value=self.config_data.get('citation_style', 'Ignore'))
        self.conversion_mode = StringVar(value=self.config_data.get('conversion_mode', 'Summarized'))
        self.status_text = StringVar(value="Ready")
        self.is_processing = BooleanVar(value=False)

        self.preflight_text = None
        self.worker_thread = None
        self.cancel_event = threading.Event()
        self.ui_queue = queue.Queue()

        self._build_menu()
        self._build_main_ui()
        self._build_status_bar()

        self.root.after(100, self._process_ui_queue)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Select PDF", command=self.select_pdf)
        file_menu.add_command(label="Load Script...", command=self.load_script_from_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Settings", command=self.open_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        self.root.config(menu=menubar)

    def _build_main_ui(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill='both', expand=True)

        # Top controls
        top = ttk.Frame(container)
        top.pack(fill='x', pady=(0,10))

        select_btn = ttk.Button(top, text="Select PDF", command=self.select_pdf)
        select_btn.pack(side='left')

        # Expandable, read-only path display to avoid pushing buttons off-screen
        path_entry = ttk.Entry(top, textvariable=self.selected_pdf, state='readonly')
        path_entry.pack(side='left', padx=10, fill='x', expand=True)

        ttk.Label(top, text="Citation Style:").pack(side='left', padx=(20,5))
        citation_combo = ttk.Combobox(top, textvariable=self.citation_style, values=["Ignore","Subtle Mention"], state='readonly', width=18)
        citation_combo.pack(side='left')

        ttk.Label(top, text="Mode:").pack(side='left', padx=(20,5))
        mode_combo = ttk.Combobox(top, textvariable=self.conversion_mode, values=["Summarized","Verbatim"], state='readonly', width=14)
        mode_combo.pack(side='left')

        # Settings button within the main UI
        settings_btn = ttk.Button(top, text="Settings", command=self.open_settings)
        settings_btn.pack(side='right', padx=(0,10))

        convert_btn = ttk.Button(top, text="Convert", command=self.on_convert)
        convert_btn.pack(side='right')

        cancel_btn = ttk.Button(top, text="Cancel", command=self.on_cancel)
        cancel_btn.pack(side='right', padx=(0,10))

        # Pre-flight editor
        editor_frame = ttk.LabelFrame(container, text="Pre-flight Review")
        editor_frame.pack(fill='both', expand=True)
        self.preflight_text = ttk.Frame(editor_frame)
        self.preflight_text.pack(fill='both', expand=True)

        # Text with scrollbar inside the frame
        text_widget = ttk.Frame(self.preflight_text)
        text_widget.pack(fill='both', expand=True)
        self.script_text = tk.Text(text_widget, wrap='word')
        self.script_text.pack(side='left', fill='both', expand=True)
        scroll = tk.Scrollbar(text_widget, command=self.script_text.yview)
        scroll.pack(side='right', fill='y')
        self.script_text.config(yscrollcommand=scroll.set)
        self.script_text.insert('1.0', "Structured script will appear here after Gemini processing...")

        # Generate Audio button
        action_bar = ttk.Frame(container)
        # Pin action bar to bottom so its buttons remain visible on small windows
        action_bar.pack(side='bottom', fill='x', pady=(10,0))
        load_btn = ttk.Button(action_bar, text="Load Script", command=self.load_script_from_file)
        load_btn.pack(side='left')
        gen_btn = ttk.Button(action_bar, text="Generate Audio", command=self.on_generate_audio)
        gen_btn.pack(side='right')

    def _build_status_bar(self):
        status = ttk.Frame(self.root)
        status.pack(fill='x', side='bottom')
        ttk.Label(status, textvariable=self.status_text, anchor='w').pack(fill='x')

    def log(self, message: str):
        self.ui_queue.put(("status", message))

    def select_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files","*.pdf")])
        if path:
            self.selected_pdf.set(path)

    def open_settings(self):
        SettingsWindow(self.root, self.config_mgr, self._on_settings_saved)

    def _on_settings_saved(self, data):
        self.config_data = data
        self.citation_style.set(self.config_data.get('citation_style', 'Ignore'))
        self.conversion_mode.set(self.config_data.get('conversion_mode', 'Summarized'))

    def on_convert(self):
        if self.is_processing.get():
            messagebox.showinfo("In Progress", "A conversion is already running.")
            return
        pdf_path = self.selected_pdf.get()
        if not pdf_path:
            messagebox.showwarning("No File", "Please select a PDF file first.")
            return
        try:
            # Always estimate Gemini cost since Verbatim is cleaned with LLM now
            est_tokens = estimate_tokens(pdf_path)
            est_cost = self.config_mgr.estimate_gemini_cost(est_tokens)
            proceed = messagebox.askyesno("Confirm Cost", f"Estimated {est_tokens} tokens (~${est_cost:.4f}). Proceed?")
            if not proceed:
                return
        except Exception as e:
            messagebox.showerror("Estimation Error", str(e))
            return

        self.cancel_event.clear()
        self.is_processing.set(True)
        self.status_text.set("Starting conversion...")

        def done_callback(event_type, payload=None):
            self.ui_queue.put((event_type, payload))

        self.worker_thread = threading.Thread(
            target=ConversionWorker.run,
            args=(pdf_path, self.citation_style.get(), self.conversion_mode.get(), self.config_mgr, self.cancel_event, done_callback),
            daemon=True,
        )
        self.worker_thread.start()

    def on_cancel(self):
        if not self.is_processing.get():
            return
        self.cancel_event.set()
        self.log("Cancellation requested...")

    def on_generate_audio(self):
        text = self.script_text.get('1.0', 'end-1c')
        if not text.strip():
            messagebox.showwarning("No Script", "No script to synthesize. Run Convert first.")
            return
        # Estimate audio cost and confirm
        try:
            cfg = self.config_mgr.load()
            provider = (cfg.get('audio_provider') or 'google_tts').strip().lower()
            num_chars = len(" ".join(text.split()))
            est_cost = self.config_mgr.estimate_tts_cost(provider, num_chars)
            proceed = messagebox.askyesno(
                "Confirm Audio Cost",
                f"Audio provider: {provider}\nCharacters: {num_chars}\nEstimated cost: ${est_cost:.4f}. Proceed?"
            )
            if not proceed:
                return
        except Exception as e:
            messagebox.showerror("Estimation Error", str(e))
            return
        try:
            audio_bytes = ConversionWorker.generate_audio_only(text, self.config_mgr, self.cancel_event, self.log)
            if audio_bytes is None:
                return
        except Exception as e:
            messagebox.showerror("Audio Error", str(e))
            return
        # Save As dialog
        default_name = os.path.splitext(os.path.basename(self.selected_pdf.get() or 'output'))[0] + '.mp3'
        save_path = filedialog.asksaveasfilename(defaultextension=".mp3", initialfile=default_name, filetypes=[("MP3","*.mp3")])
        if save_path:
            try:
                with open(save_path, 'wb') as f:
                    f.write(audio_bytes)
                # Also save the exact text used for synthesis next to the MP3
                base, _ = os.path.splitext(save_path)
                text_path = base + '.txt'
                with open(text_path, 'w', encoding='utf-8') as tf:
                    tf.write(text)
                messagebox.showinfo("Success", "Audio saved successfully.")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

    def load_script_from_file(self):
        path = filedialog.askopenfilename(title="Load Script", filetypes=[
            ("Text files","*.txt"),
            ("Markdown","*.md"),
            ("All files","*.*")
        ])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.script_text.delete('1.0', 'end')
            self.script_text.insert('1.0', content)
            self.status_text.set(f"Loaded script from {os.path.basename(path)}. Ready to Generate Audio.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _process_ui_queue(self):
        try:
            while True:
                event_type, payload = self.ui_queue.get_nowait()
                if event_type == "status":
                    self.status_text.set(payload)
                elif event_type == "error":
                    self.is_processing.set(False)
                    messagebox.showerror("Error", str(payload))
                elif event_type == "script":
                    self.script_text.delete('1.0', 'end')
                    self.script_text.insert('1.0', payload)
                    self.status_text.set("Script ready. Review and click Generate Audio.")
                elif event_type == "done":
                    self.is_processing.set(False)
                    self.status_text.set("Conversion complete. Review script or generate audio.")
                elif event_type == "cancelled":
                    self.is_processing.set(False)
                    self.status_text.set("Operation cancelled.")
        except queue.Empty:
            pass
        self.root.after(100, self._process_ui_queue)

class SettingsWindow:
    def __init__(self, master, config_mgr: ConfigManager, on_save):
        self.top = Toplevel(master)
        self.top.title("Settings")
        self.top.geometry('600x400')
        self.config_mgr = config_mgr
        self.on_save = on_save

        data = {}
        try:
            data = self.config_mgr.load()
        except Exception:
            pass

        self.gemini_key = StringVar(value=data.get('gemini_api_key',''))
        self.eleven_key = StringVar(value=data.get('eleven_api_key',''))
        self.google_tts_key = StringVar(value=data.get('google_tts_api_key',''))
        self.voice_id = StringVar(value=data.get('voice_id',''))
        self.voice_display_map = {}  # display name -> id
        self.stability = DoubleVar(value=float(data.get('stability', 0.5)))
        self.clarity = DoubleVar(value=float(data.get('clarity', 0.5)))
        self.citation_style = StringVar(value=data.get('citation_style','Ignore'))
        self.audio_provider = StringVar(value=data.get('audio_provider','elevenlabs'))
        # Google TTS params
        self.gtts_voice_name = StringVar(value=data.get('gtts_voice_name','en-US-Studio-O'))
        self.gtts_language_code = StringVar(value=data.get('gtts_language_code','en-US'))
        self.gtts_speaking_rate = DoubleVar(value=float(data.get('gtts_speaking_rate',1.0)))
        self.gtts_pitch = DoubleVar(value=float(data.get('gtts_pitch',0.0)))

        frm = ttk.Frame(self.top, padding=10)
        frm.pack(fill='both', expand=True)

        row = 0
        ttk.Label(frm, text="Gemini API Key").grid(row=row, column=0, sticky='w'); row+=1
        ttk.Entry(frm, textvariable=self.gemini_key, show='*').grid(row=row, column=0, sticky='ew', columnspan=2); row+=1
        ttk.Label(frm, text="Eleven Labs API Key").grid(row=row, column=0, sticky='w'); row+=1
        ttk.Entry(frm, textvariable=self.eleven_key, show='*').grid(row=row, column=0, sticky='ew', columnspan=2); row+=1
        ttk.Label(frm, text="Google TTS API Key").grid(row=row, column=0, sticky='w'); row+=1
        ttk.Entry(frm, textvariable=self.google_tts_key, show='*').grid(row=row, column=0, sticky='ew', columnspan=2); row+=1

        ttk.Label(frm, text="Audio Provider").grid(row=row, column=0, sticky='w')
        ttk.Combobox(frm, textvariable=self.audio_provider, values=["elevenlabs","google_tts"], state='readonly').grid(row=row, column=1, sticky='ew'); row+=1
        ttk.Label(frm, text="Voice").grid(row=row, column=0, sticky='w')
        self.voice_combo = ttk.Combobox(frm, values=[], state='readonly')
        self.voice_combo.grid(row=row, column=1, sticky='ew'); row+=1

        ttk.Label(frm, text="Stability").grid(row=row, column=0, sticky='w')
        ttk.Scale(frm, variable=self.stability, from_=0.0, to=1.0, orient='horizontal').grid(row=row, column=1, sticky='ew'); row+=1
        ttk.Label(frm, text="Clarity").grid(row=row, column=0, sticky='w')
        ttk.Scale(frm, variable=self.clarity, from_=0.0, to=1.0, orient='horizontal').grid(row=row, column=1, sticky='ew'); row+=1

        gtts_frame = ttk.LabelFrame(frm, text="Google TTS (Studio)")
        gtts_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(10,0)); row+=1
        ttk.Label(gtts_frame, text="Voice Name").grid(row=0, column=0, sticky='w')
        ttk.Entry(gtts_frame, textvariable=self.gtts_voice_name).grid(row=0, column=1, sticky='ew')
        ttk.Label(gtts_frame, text="Language Code").grid(row=1, column=0, sticky='w')
        ttk.Entry(gtts_frame, textvariable=self.gtts_language_code).grid(row=1, column=1, sticky='ew')
        ttk.Label(gtts_frame, text="Speaking Rate").grid(row=2, column=0, sticky='w')
        ttk.Scale(gtts_frame, variable=self.gtts_speaking_rate, from_=0.25, to=4.0, orient='horizontal').grid(row=2, column=1, sticky='ew')
        ttk.Label(gtts_frame, text="Pitch").grid(row=3, column=0, sticky='w')
        ttk.Scale(gtts_frame, variable=self.gtts_pitch, from_=-20.0, to=20.0, orient='horizontal').grid(row=3, column=1, sticky='ew')

        ttk.Label(frm, text="Citation Style").grid(row=row, column=0, sticky='w')
        ttk.Combobox(frm, textvariable=self.citation_style, values=["Ignore","Subtle Mention"], state='readonly').grid(row=row, column=1, sticky='ew'); row+=1

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=2, pady=(10,0), sticky='e')
        ttk.Button(btns, text="Fetch Voices", command=self.fetch_voices).pack(side='left')
        ttk.Button(btns, text="Save", command=self.save).pack(side='left', padx=10)

        frm.columnconfigure(0, weight=0)
        frm.columnconfigure(1, weight=1)

    def fetch_voices(self):
        try:
            voices = self.config_mgr.fetch_elevenlabs_voices(self.eleven_key.get())
            # Build display list as "Name (id)" and map back to id
            display_values = []
            self.voice_display_map = {}
            for v in voices:
                vid = v.get('voice_id') or ''
                name = v.get('name') or vid
                display = f"{name} ({vid})" if name and vid else name or vid
                display_values.append(display)
                self.voice_display_map[display] = vid
            self.voice_combo['values'] = display_values
            # Try to select existing voice id by finding its display
            current_id = (self.voice_id.get() or '').strip()
            if current_id:
                for disp, vid in self.voice_display_map.items():
                    if vid == current_id:
                        self.voice_combo.set(disp)
                        break
            elif display_values:
                self.voice_combo.set(display_values[0])
            messagebox.showinfo("Voices", f"Fetched {len(voices)} voices.")
        except Exception as e:
            messagebox.showerror("Voice Fetch Error", str(e))

    def save(self):
        # Resolve voice id from displayed selection
        selected_display = self.voice_combo.get()
        resolved_voice_id = self.voice_display_map.get(selected_display, self.voice_id.get().strip())
        data = {
            'gemini_api_key': self.gemini_key.get().strip(),
            'eleven_api_key': self.eleven_key.get().strip(),
            'google_tts_api_key': self.google_tts_key.get().strip(),
            'voice_id': resolved_voice_id,
            'stability': float(self.stability.get()),
            'clarity': float(self.clarity.get()),
            'citation_style': self.citation_style.get(),
            'audio_provider': self.audio_provider.get(),
            'gtts_voice_name': self.gtts_voice_name.get().strip(),
            'gtts_language_code': self.gtts_language_code.get().strip(),
            'gtts_speaking_rate': float(self.gtts_speaking_rate.get()),
            'gtts_pitch': float(self.gtts_pitch.get()),
        }
        try:
            self.config_mgr.save(data)
            self.on_save(data)
            messagebox.showinfo("Saved", "Settings saved.")
            self.top.destroy()
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


def main():
    root = Tk()
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    App(root)
    root.mainloop()

if __name__ == '__main__':
    main()
