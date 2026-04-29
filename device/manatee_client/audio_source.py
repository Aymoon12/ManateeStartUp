"""Pluggable audio source interface with hydrophone and file-watcher implementations.

AudioSource ABC: override for hydrophone, USB mic, or file-watcher.
HydrophoneSource: captures real-time audio from a USB hydrophone via sounddevice.
FileWatcherSource: watches a directory for audio files (for testing / manual ingestion).
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

import numpy as np
import soundfile as sf

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


class HydrophoneSource(AudioSource):
    """Captures real-time audio from a USB hydrophone via sounddevice.

    Records continuously in fixed-length segments (default 30 s), writes each
    segment to a WAV file, and yields it as an AudioChunk.  The Recorder picks
    these up identically to FileWatcherSource output.

    30 seconds is the natural segment length: EdgeDetector.MAX_CLIPS = 60
    clips at 0.5 s stride = 30 s.  Longer segments waste audio (clips beyond
    60 are dropped); shorter segments weaken the 50 % detection rule.
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 44100,
        segment_duration: float = 30.0,
        output_dir: str = "/var/lib/manatee/incoming",
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.output_dir = Path(output_dir)
        self._queue: queue.Queue[AudioChunk] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "HydrophoneSource started (device=%s, rate=%d, segment=%.0fs)",
            self.device if self.device is not None else "default",
            self.sample_rate,
            self.segment_duration,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
        logger.info("HydrophoneSource stopped")

    def get_next_audio(self, timeout: float = 5.0) -> Optional[AudioChunk]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        import sounddevice as sd

        segment_samples = int(self.sample_rate * self.segment_duration)
        block_size = int(self.sample_rate)  # read in 1-second blocks for responsive shutdown

        while not self._stop_event.is_set():
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    blocksize=block_size,
                ) as stream:
                    logger.info("Audio stream opened on device %s", self.device or "default")

                    while not self._stop_event.is_set():
                        timestamp = datetime.now(timezone.utc)
                        buffer: list[np.ndarray] = []
                        collected = 0

                        # Collect one full segment in 1-second reads
                        while collected < segment_samples and not self._stop_event.is_set():
                            to_read = min(block_size, segment_samples - collected)
                            data, overflowed = stream.read(to_read)
                            if overflowed:
                                logger.warning("Audio input overflow — samples may have been dropped")
                            buffer.append(data.copy())
                            collected += len(data)

                        if self._stop_event.is_set():
                            break

                        # Write segment to WAV
                        audio = np.concatenate(buffer, axis=0)
                        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
                        filepath = self.output_dir / f"capture_{ts_str}.wav"
                        sf.write(str(filepath), audio, self.sample_rate)

                        chunk = AudioChunk(
                            audio_path=filepath,
                            timestamp=timestamp,
                            duration=self.segment_duration,
                        )
                        self._queue.put(chunk)
                        logger.debug("Captured segment: %s (%.1fs)", filepath.name, self.segment_duration)

            except Exception as e:
                logger.error("Audio device error (retry in 10s): %s", e)
                self._stop_event.wait(10.0)
