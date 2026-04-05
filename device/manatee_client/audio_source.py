"""Pluggable audio source interface and file-watcher implementation.

AudioSource ABC: override for hydrophone, USB mic, or file-watcher.
FileWatcherSource: watches a directory for audio files (for testing).
"""

import logging
import queue
import shutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


@dataclass
class AudioChunk:
    audio_path: Path
    timestamp: datetime
    duration: float | None  # None if unknown


class AudioSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def get_next_audio(self, timeout: float = 5.0) -> Optional[AudioChunk]: ...


class FileWatcherSource(AudioSource):
    """Watches a directory for new audio files and yields them as AudioChunks.

    Files are moved to a 'processing/' subdirectory after pickup to avoid
    re-processing.
    """

    def __init__(self, watch_dir: str, poll_interval: float = 2.0):
        self.watch_dir = Path(watch_dir)
        self.processing_dir = self.watch_dir / "processing"
        self.poll_interval = poll_interval
        self._queue: queue.Queue[AudioChunk] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("FileWatcherSource started (watching %s)", self.watch_dir)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("FileWatcherSource stopped")

    def get_next_audio(self, timeout: float = 5.0) -> Optional[AudioChunk]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._scan_once()
            except Exception as e:
                logger.error("FileWatcher scan error: %s", e)
            self._stop_event.wait(self.poll_interval)

    def _scan_once(self) -> None:
        if not self.watch_dir.exists():
            return

        files = sorted(
            (f for f in self.watch_dir.iterdir()
             if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS),
            key=lambda f: f.stat().st_mtime,
        )

        for f in files:
            dest = self.processing_dir / f.name
            try:
                shutil.move(str(f), str(dest))
            except OSError as e:
                logger.warning("Failed to move %s: %s", f.name, e)
                continue

            chunk = AudioChunk(
                audio_path=dest,
                timestamp=datetime.now(timezone.utc),
                duration=None,  # determined at inference time
            )
            self._queue.put(chunk)
            logger.debug("Queued audio: %s", f.name)
