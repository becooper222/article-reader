#!/usr/bin/env python3
"""
Batch processing for Research Paper Audiobook Converter.

Supports:
- Background execution (survives terminal close, computer sleep)
- Checkpoint/resume after interruption
- Job status monitoring
- Parallel processing (optional)

Usage:
  # Start a batch job
  python batch.py start /path/to/pdfs/*.pdf --output ./audio_papers/
  
  # Check status
  python batch.py status
  
  # Resume interrupted job
  python batch.py resume
  
  # Run in background (survives terminal close)
  nohup python batch.py start /path/to/pdfs/*.pdf --output ./audio/ &
  
  # Or use the built-in daemon mode
  python batch.py start /path/to/pdfs/*.pdf --daemon
"""
import argparse
import gzip
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
import traceback

# Ensure we can import from project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from processing import (
    parse_pdf,
    structure_with_gemini,
    clean_with_gemini,
    ConversionWorker,
    PDFCache,
    estimate_tokens,
    get_memory_usage_mb,
    force_gc,
)


# =============================================================================
# Job State Management
# =============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PDFJob:
    """State for a single PDF conversion job."""
    pdf_path: str
    status: str = JobStatus.PENDING.value
    output_mp3: str = ""
    output_script: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    script_chars: int = 0
    audio_size_bytes: int = 0


@dataclass
class BatchState:
    """Persistent state for a batch conversion job."""
    job_id: str
    created_at: str
    output_dir: str
    conversion_mode: str
    citation_style: str
    compress_scripts: bool
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    current_index: int = 0
    is_running: bool = False
    last_updated: str = ""
    completed_count: int = 0
    failed_count: int = 0
    total_audio_bytes: int = 0
    pid: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchState":
        return cls(**data)


class BatchStateManager:
    """Manages persistent batch job state."""
    
    STATE_FILE = ".batch_state.json"
    LOCK_FILE = ".batch_lock"
    LOG_FILE = "batch.log"
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.state_path = os.path.join(self.base_dir, self.STATE_FILE)
        self.lock_path = os.path.join(self.base_dir, self.LOCK_FILE)
        self.log_path = os.path.join(self.base_dir, self.LOG_FILE)
    
    def acquire_lock(self) -> bool:
        """Try to acquire the batch lock. Returns False if another process has it."""
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path, 'r') as f:
                    data = json.load(f)
                    pid = data.get('pid', 0)
                    
                # Check if process is still running
                if pid and self._is_process_running(pid):
                    return False
                    
                # Stale lock, remove it
                os.unlink(self.lock_path)
            except Exception:
                pass
        
        # Create lock
        with open(self.lock_path, 'w') as f:
            json.dump({'pid': os.getpid(), 'time': datetime.now().isoformat()}, f)
        return True
    
    def release_lock(self):
        """Release the batch lock."""
        try:
            if os.path.exists(self.lock_path):
                os.unlink(self.lock_path)
        except Exception:
            pass
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    
    def load_state(self) -> Optional[BatchState]:
        """Load batch state from disk."""
        if not os.path.exists(self.state_path):
            return None
        try:
            with open(self.state_path, 'r') as f:
                data = json.load(f)
            return BatchState.from_dict(data)
        except Exception as e:
            self.log(f"Error loading state: {e}")
            return None
    
    def save_state(self, state: BatchState):
        """Save batch state to disk (atomic write)."""
        state.last_updated = datetime.now().isoformat()
        state.pid = os.getpid()
        
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp_path, self.state_path)
    
    def clear_state(self):
        """Remove batch state file."""
        try:
            if os.path.exists(self.state_path):
                os.unlink(self.state_path)
        except Exception:
            pass
    
    def log(self, message: str, also_print: bool = True):
        """Log a message to the batch log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.log_path, 'a') as f:
                f.write(log_line)
        except Exception:
            pass
        
        if also_print:
            print(message)
    
    def get_log_tail(self, lines: int = 20) -> List[str]:
        """Get the last N lines from the log file."""
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path, 'r') as f:
                all_lines = f.readlines()
            return all_lines[-lines:]
        except Exception:
            return []


# =============================================================================
# Batch Processor
# =============================================================================

class BatchProcessor:
    """Handles batch PDF conversion with checkpoint/resume."""
    
    def __init__(self, state_manager: BatchStateManager, config_mgr: ConfigManager):
        self.state_manager = state_manager
        self.config_mgr = config_mgr
        self.cancel_event = threading.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Handle graceful shutdown on SIGINT/SIGTERM."""
        def handler(signum, frame):
            self.state_manager.log("Received shutdown signal, saving state...")
            self.cancel_event.set()
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def create_batch(self, pdf_paths: List[str], output_dir: str,
                     mode: str = "Summarized", citations: str = "Ignore",
                     compress: bool = True) -> BatchState:
        """Create a new batch job."""
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        jobs = []
        for pdf_path in pdf_paths:
            abs_path = os.path.abspath(pdf_path)
            if not os.path.exists(abs_path):
                self.state_manager.log(f"Warning: PDF not found, skipping: {pdf_path}")
                continue
            
            base_name = os.path.splitext(os.path.basename(abs_path))[0]
            mp3_path = os.path.join(output_dir, f"{base_name}.mp3")
            script_ext = ".txt.gz" if compress else ".txt"
            script_path = os.path.join(output_dir, f"{base_name}{script_ext}")
            
            jobs.append(PDFJob(
                pdf_path=abs_path,
                output_mp3=mp3_path,
                output_script=script_path,
            ).__dict__)
        
        state = BatchState(
            job_id=job_id,
            created_at=datetime.now().isoformat(),
            output_dir=os.path.abspath(output_dir),
            conversion_mode=mode,
            citation_style=citations,
            compress_scripts=compress,
            jobs=jobs,
        )
        
        return state
    
    def run(self, state: BatchState):
        """Run the batch processing job."""
        if not self.state_manager.acquire_lock():
            self.state_manager.log("ERROR: Another batch job is already running!")
            self.state_manager.log("Use 'python batch.py status' to check progress")
            return False
        
        try:
            state.is_running = True
            self.state_manager.save_state(state)
            
            # Ensure output directory exists
            os.makedirs(state.output_dir, exist_ok=True)
            
            self.state_manager.log(f"")
            self.state_manager.log(f"{'='*60}")
            self.state_manager.log(f"BATCH JOB STARTED: {state.job_id}")
            self.state_manager.log(f"{'='*60}")
            self.state_manager.log(f"Total PDFs: {len(state.jobs)}")
            self.state_manager.log(f"Output: {state.output_dir}")
            self.state_manager.log(f"Mode: {state.conversion_mode}")
            self.state_manager.log(f"PID: {os.getpid()}")
            self.state_manager.log(f"")
            
            cfg = self.config_mgr.load()
            api_key = cfg.get("gemini_api_key", "")
            model_name = cfg.get("model_name", "gemini-2.0-flash")
            
            if not api_key:
                self.state_manager.log("ERROR: Gemini API key not configured!")
                return False
            
            # Process from current index (for resume)
            while state.current_index < len(state.jobs):
                if self.cancel_event.is_set():
                    self.state_manager.log("Batch cancelled, progress saved.")
                    break
                
                job_dict = state.jobs[state.current_index]
                job = PDFJob(**job_dict)
                
                # Skip already completed jobs
                if job.status == JobStatus.COMPLETED.value:
                    state.current_index += 1
                    continue
                
                # Process this PDF
                success = self._process_single(
                    job=job,
                    api_key=api_key,
                    model_name=model_name,
                    mode=state.conversion_mode,
                    citations=state.citation_style,
                    compress=state.compress_scripts,
                )
                
                # Update job in state
                state.jobs[state.current_index] = job.__dict__
                
                if success:
                    state.completed_count += 1
                    state.total_audio_bytes += job.audio_size_bytes
                else:
                    state.failed_count += 1
                
                state.current_index += 1
                
                # Save checkpoint after each PDF
                self.state_manager.save_state(state)
                
                # Memory cleanup
                force_gc()
            
            # Final status
            state.is_running = False
            self.state_manager.save_state(state)
            
            self._print_summary(state)
            return True
            
        finally:
            state.is_running = False
            self.state_manager.save_state(state)
            self.state_manager.release_lock()
    
    def _process_single(self, job: PDFJob, api_key: str, model_name: str,
                        mode: str, citations: str, compress: bool) -> bool:
        """Process a single PDF. Returns True on success."""
        filename = os.path.basename(job.pdf_path)
        idx = f"[{self.state_manager.load_state().current_index + 1}/{len(self.state_manager.load_state().jobs)}]"
        
        self.state_manager.log(f"")
        self.state_manager.log(f"{idx} Processing: {filename}")
        self.state_manager.log(f"    Memory: {get_memory_usage_mb():.0f} MB")
        
        job.status = JobStatus.PROCESSING.value
        job.started_at = datetime.now().isoformat()
        
        try:
            # Step 1: Parse PDF
            self.state_manager.log(f"    Parsing PDF...", also_print=False)
            text, image_locations = parse_pdf(job.pdf_path)
            self.state_manager.log(f"    Extracted {len(text):,} chars, {len(image_locations)} figures")
            
            if self.cancel_event.is_set():
                job.status = JobStatus.CANCELLED.value
                return False
            
            # Step 2: Process with Gemini
            self.state_manager.log(f"    Processing with Gemini ({mode})...")
            
            if mode == "Verbatim":
                script = clean_with_gemini(
                    text=text,
                    pdf_path=job.pdf_path,
                    image_locations=image_locations,
                    api_key=api_key,
                    model_name=model_name,
                    citation_style=citations,
                    cancel_event=self.cancel_event,
                    log=lambda m: self.state_manager.log(f"      {m}", also_print=False),
                )
            else:
                script = structure_with_gemini(
                    text=text,
                    pdf_path=job.pdf_path,
                    image_locations=image_locations,
                    api_key=api_key,
                    model_name=model_name,
                    citation_style=citations,
                    cancel_event=self.cancel_event,
                    log=lambda m: self.state_manager.log(f"      {m}", also_print=False),
                )
            
            job.script_chars = len(script)
            self.state_manager.log(f"    Generated script: {len(script):,} chars")
            
            if self.cancel_event.is_set():
                job.status = JobStatus.CANCELLED.value
                return False
            
            # Step 3: Generate audio (streaming)
            self.state_manager.log(f"    Generating audio...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(job.output_mp3), exist_ok=True)
            
            # Allow fallback to Gemini if Inworld quota exceeded
            cfg = self.config_mgr.load()
            allow_fallback = cfg.get('tts_fallback_enabled', True)
            
            file_size = ConversionWorker.generate_audio_streaming(
                script,
                job.output_mp3,
                self.config_mgr,
                self.cancel_event,
                log=lambda m: self.state_manager.log(f"      {m}", also_print=False),
                allow_fallback=allow_fallback,
            )
            
            if file_size is None:
                if self.cancel_event.is_set():
                    job.status = JobStatus.CANCELLED.value
                    return False
                raise RuntimeError("Audio generation failed")
            
            job.audio_size_bytes = file_size
            self.state_manager.log(f"    Audio: {file_size / (1024*1024):.2f} MB")
            
            # Step 4: Save script
            if compress:
                with gzip.open(job.output_script, 'wt', encoding='utf-8') as f:
                    f.write(script)
            else:
                with open(job.output_script, 'w', encoding='utf-8') as f:
                    f.write(script)
            
            job.status = JobStatus.COMPLETED.value
            job.completed_at = datetime.now().isoformat()
            self.state_manager.log(f"    ✅ Completed!")
            
            return True
            
        except Exception as e:
            job.status = JobStatus.FAILED.value
            job.error = str(e)
            job.completed_at = datetime.now().isoformat()
            self.state_manager.log(f"    ❌ Failed: {e}")
            self.state_manager.log(f"    {traceback.format_exc()}", also_print=False)
            return False
    
    def _print_summary(self, state: BatchState):
        """Print batch completion summary."""
        total = len(state.jobs)
        completed = state.completed_count
        failed = state.failed_count
        pending = total - completed - failed
        
        total_mb = state.total_audio_bytes / (1024 * 1024)
        
        self.state_manager.log(f"")
        self.state_manager.log(f"{'='*60}")
        self.state_manager.log(f"BATCH COMPLETE: {state.job_id}")
        self.state_manager.log(f"{'='*60}")
        self.state_manager.log(f"")
        self.state_manager.log(f"  ✅ Completed: {completed}/{total}")
        self.state_manager.log(f"  ❌ Failed:    {failed}/{total}")
        if pending > 0:
            self.state_manager.log(f"  ⏳ Pending:   {pending}/{total}")
        self.state_manager.log(f"")
        self.state_manager.log(f"  Total audio: {total_mb:.2f} MB")
        self.state_manager.log(f"  Output dir:  {state.output_dir}")
        self.state_manager.log(f"")
        
        if failed > 0:
            self.state_manager.log(f"Failed jobs:")
            for job_dict in state.jobs:
                job = PDFJob(**job_dict)
                if job.status == JobStatus.FAILED.value:
                    self.state_manager.log(f"  - {os.path.basename(job.pdf_path)}: {job.error}")


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_start(args, state_manager: BatchStateManager, config_mgr: ConfigManager):
    """Start a new batch job."""
    # Check for existing job
    existing = state_manager.load_state()
    if existing and existing.is_running:
        print("ERROR: A batch job is already running!")
        print("Use 'python batch.py status' to check progress")
        print("Use 'python batch.py resume' to resume an interrupted job")
        return 1
    
    if not args.pdfs:
        print("ERROR: No PDF files specified")
        return 1
    
    # Expand globs and validate
    pdf_paths = []
    for pattern in args.pdfs:
        if '*' in pattern or '?' in pattern:
            import glob
            matches = glob.glob(pattern)
            pdf_paths.extend(matches)
        else:
            pdf_paths.append(pattern)
    
    pdf_paths = [p for p in pdf_paths if p.lower().endswith('.pdf')]
    
    if not pdf_paths:
        print("ERROR: No valid PDF files found")
        return 1
    
    print(f"Found {len(pdf_paths)} PDF(s) to process")
    
    # Create batch
    processor = BatchProcessor(state_manager, config_mgr)
    state = processor.create_batch(
        pdf_paths=pdf_paths,
        output_dir=args.output,
        mode=args.mode,
        citations=args.citations,
        compress=not args.no_compress,
    )
    
    if not state.jobs:
        print("ERROR: No valid PDFs to process")
        return 1
    
    # Daemon mode
    if args.daemon:
        import subprocess
        
        # Build command without --daemon
        cmd = [sys.executable, __file__, "start"] + args.pdfs
        cmd.extend(["--output", args.output])
        cmd.extend(["--mode", args.mode])
        cmd.extend(["--citations", args.citations])
        if args.no_compress:
            cmd.append("--no-compress")
        
        # Start in background
        log_file = open(state_manager.log_path, 'a')
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        
        print(f"Batch job started in background (PID: {proc.pid})")
        print(f"Log file: {state_manager.log_path}")
        print(f"Use 'python batch.py status' to check progress")
        return 0
    
    # Run in foreground
    processor.run(state)
    return 0


def cmd_status(args, state_manager: BatchStateManager, config_mgr: ConfigManager):
    """Check status of current/last batch job."""
    state = state_manager.load_state()
    
    if not state:
        print("No batch job found.")
        print("Use 'python batch.py start <pdfs>' to start a new job")
        return 0
    
    total = len(state.jobs)
    completed = sum(1 for j in state.jobs if j['status'] == JobStatus.COMPLETED.value)
    failed = sum(1 for j in state.jobs if j['status'] == JobStatus.FAILED.value)
    pending = total - completed - failed
    
    # Check if process is still running
    is_actually_running = False
    if state.is_running and state.pid:
        try:
            os.kill(state.pid, 0)
            is_actually_running = True
        except (OSError, ProcessLookupError):
            pass
    
    print(f"")
    print(f"{'='*50}")
    print(f"BATCH JOB: {state.job_id}")
    print(f"{'='*50}")
    print(f"")
    print(f"Status: {'🟢 RUNNING' if is_actually_running else '⏸️  STOPPED' if pending > 0 else '✅ COMPLETE'}")
    if is_actually_running:
        print(f"PID: {state.pid}")
    print(f"")
    print(f"Progress: {completed}/{total} completed")
    
    # Progress bar
    bar_width = 40
    filled = int(bar_width * completed / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    percent = (completed / total * 100) if total > 0 else 0
    print(f"[{bar}] {percent:.1f}%")
    print(f"")
    print(f"  ✅ Completed: {completed}")
    print(f"  ❌ Failed:    {failed}")
    print(f"  ⏳ Pending:   {pending}")
    print(f"")
    print(f"Mode: {state.conversion_mode}")
    print(f"Output: {state.output_dir}")
    print(f"")
    
    if args.verbose:
        print("Jobs:")
        for i, job_dict in enumerate(state.jobs):
            job = PDFJob(**job_dict)
            status_icon = {
                JobStatus.COMPLETED.value: "✅",
                JobStatus.FAILED.value: "❌",
                JobStatus.PROCESSING.value: "🔄",
                JobStatus.PENDING.value: "⏳",
                JobStatus.CANCELLED.value: "⏸️",
            }.get(job.status, "?")
            print(f"  {status_icon} {os.path.basename(job.pdf_path)}")
            if job.status == JobStatus.FAILED.value and job.error:
                print(f"      Error: {job.error[:60]}")
    
    if args.log:
        print(f"\nRecent log entries:")
        print("-" * 50)
        for line in state_manager.get_log_tail(10):
            print(line.rstrip())
    
    return 0


def cmd_resume(args, state_manager: BatchStateManager, config_mgr: ConfigManager):
    """Resume an interrupted batch job."""
    state = state_manager.load_state()
    
    if not state:
        print("No batch job to resume.")
        print("Use 'python batch.py start <pdfs>' to start a new job")
        return 1
    
    pending = sum(1 for j in state.jobs if j['status'] in [JobStatus.PENDING.value, JobStatus.PROCESSING.value])
    
    if pending == 0:
        print("Batch job already complete. Nothing to resume.")
        return 0
    
    print(f"Resuming batch job: {state.job_id}")
    print(f"Remaining: {pending} PDFs")
    
    processor = BatchProcessor(state_manager, config_mgr)
    processor.run(state)
    return 0


def cmd_cancel(args, state_manager: BatchStateManager, config_mgr: ConfigManager):
    """Cancel the running batch job."""
    state = state_manager.load_state()
    
    if not state:
        print("No batch job found.")
        return 0
    
    if state.is_running and state.pid:
        try:
            os.kill(state.pid, signal.SIGTERM)
            print(f"Sent termination signal to process {state.pid}")
            print("Job will save progress and stop gracefully.")
            print("Use 'python batch.py status' to check when it's stopped.")
        except (OSError, ProcessLookupError):
            print("Process not found. Clearing lock...")
            state_manager.release_lock()
    else:
        print("No running batch job to cancel.")
    
    return 0


def cmd_clear(args, state_manager: BatchStateManager, config_mgr: ConfigManager):
    """Clear batch state (after completion or to start fresh)."""
    state = state_manager.load_state()
    
    if state and state.is_running:
        print("Cannot clear while a batch job is running.")
        print("Use 'python batch.py cancel' first.")
        return 1
    
    state_manager.clear_state()
    state_manager.release_lock()
    print("Batch state cleared.")
    return 0


def cmd_retry_failed(args, state_manager: BatchStateManager, config_mgr: ConfigManager):
    """Retry only the failed jobs from the last batch."""
    state = state_manager.load_state()
    
    if not state:
        print("No batch job found.")
        return 1
    
    # Reset failed jobs to pending
    failed_count = 0
    for job in state.jobs:
        if job['status'] == JobStatus.FAILED.value:
            job['status'] = JobStatus.PENDING.value
            job['error'] = ""
            failed_count += 1
    
    if failed_count == 0:
        print("No failed jobs to retry.")
        return 0
    
    # Find first pending job
    state.current_index = 0
    for i, job in enumerate(state.jobs):
        if job['status'] == JobStatus.PENDING.value:
            state.current_index = i
            break
    
    state.failed_count -= failed_count
    state_manager.save_state(state)
    
    print(f"Reset {failed_count} failed jobs. Resuming...")
    
    processor = BatchProcessor(state_manager, config_mgr)
    processor.run(state)
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch processing for Research Paper Audiobook Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # start command
    start_parser = subparsers.add_parser("start", help="Start a new batch job")
    start_parser.add_argument("pdfs", nargs="+", help="PDF files to process (supports glob patterns)")
    start_parser.add_argument("--output", "-o", default="./audio_papers", 
                              help="Output directory (default: ./audio_papers)")
    start_parser.add_argument("--mode", choices=["Summarized", "Verbatim"], 
                              default="Summarized", help="Conversion mode")
    start_parser.add_argument("--citations", choices=["Ignore", "Subtle Mention"], 
                              default="Ignore", help="Citation handling")
    start_parser.add_argument("--no-compress", action="store_true",
                              help="Save scripts as plain .txt instead of .txt.gz")
    start_parser.add_argument("--daemon", "-d", action="store_true",
                              help="Run in background (survives terminal close)")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Check batch job status")
    status_parser.add_argument("--verbose", "-v", action="store_true",
                               help="Show individual job status")
    status_parser.add_argument("--log", "-l", action="store_true",
                               help="Show recent log entries")
    
    # resume command
    subparsers.add_parser("resume", help="Resume an interrupted batch job")
    
    # cancel command
    subparsers.add_parser("cancel", help="Cancel the running batch job")
    
    # clear command
    subparsers.add_parser("clear", help="Clear batch state")
    
    # retry-failed command
    subparsers.add_parser("retry-failed", help="Retry failed jobs from last batch")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize managers
    project_dir = os.path.dirname(os.path.abspath(__file__))
    state_manager = BatchStateManager(project_dir)
    config_mgr = ConfigManager(os.path.join(project_dir, "config.json"))
    config_mgr.ensure_config()
    
    # Dispatch command
    commands = {
        "start": cmd_start,
        "status": cmd_status,
        "resume": cmd_resume,
        "cancel": cmd_cancel,
        "clear": cmd_clear,
        "retry-failed": cmd_retry_failed,
    }
    
    return commands[args.command](args, state_manager, config_mgr)


if __name__ == "__main__":
    sys.exit(main())

