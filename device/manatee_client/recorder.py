"""Recorder: audio source -> inference -> outbox + detection log.

Runs in its own thread, controlled by a stop_event.
"""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import soundfile as sf

from manatee_client.audio_source import AudioSource
from manatee_client.detection_log import DetectionLog, DetectionLogEntry
from manatee_client.inference import EdgeDetector

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(
        self,
        audio_source: AudioSource,
        detector: EdgeDetector,
        detection_log: DetectionLog,
        outbox_dir: str,
        archive_dir: str,
    ):
        self.audio_source = audio_source
        self.detector = detector
        self.detection_log = detection_log
        self.outbox_dir = Path(outbox_dir)
        self.archive_dir = Path(archive_dir)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def run(self, stop_event: Event) -> None:
        """Main recorder loop. Call from a daemon thread."""
        logger.info("Recorder started")
        self.audio_source.start()

        try:
            while not stop_event.is_set():
                chunk = self.audio_source.get_next_audio(timeout=5.0)
                if chunk is None:
                    continue

                try:
                    self._process_chunk(chunk)
                except Exception as e:
                    logger.error("Failed to process %s: %s", chunk.audio_path.name, e)
        finally:
            self.audio_source.stop()
            logger.info("Recorder stopped")

    def _process_chunk(self, chunk) -> None:
        start = time.monotonic()
        result = self.detector.predict_file(str(chunk.audio_path))
        inference_time = round(time.monotonic() - start, 2)

        # Determine audio duration from file
        try:
            info = sf.info(str(chunk.audio_path))
            audio_duration = round(info.duration, 2)
        except Exception:
            audio_duration = None

        # Build filesystem-safe timestamp: colons -> hyphens
        ts = chunk.timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
        ts_iso = chunk.timestamp.isoformat()

        # Name outbox file to match outbox.py regex patterns
        ext = chunk.audio_path.suffix.lower() or ".wav"
        if result.is_manatee:
            outbox_name = f"detection_{ts}_{result.confidence:.2f}{ext}"
        else:
            outbox_name = f"background_{ts}{ext}"

        outbox_path = self.outbox_dir / outbox_name

        # Copy audio to outbox for upload
        shutil.copy2(str(chunk.audio_path), str(outbox_path))

        # Write .meta sidecar
        meta = {
            "audio_duration": audio_duration,
            "extra_metadata": {
                "clips_analyzed": result.clips_analyzed,
                "clips_positive": result.clips_positive,
                "max_confidence": result.max_confidence,
                "avg_positive_confidence": result.avg_positive_confidence,
                "inference_time_s": inference_time,
                "edge_inference": True,
            },
        }
        meta_path = outbox_path.with_suffix(outbox_path.suffix + ".meta")
        meta_path.write_text(json.dumps(meta))

        # Copy audio to archive for SD card retrieval
        archive_path = self.archive_dir / outbox_name
        shutil.copy2(str(chunk.audio_path), str(archive_path))

        # Append to detection log
        entry = DetectionLogEntry(
            timestamp=ts_iso,
            confidence=result.confidence,
            is_manatee=result.is_manatee,
            audio_file=str(archive_path),
            outbox_file=str(outbox_path),
            clips_analyzed=result.clips_analyzed,
            clips_positive=result.clips_positive,
            max_confidence=result.max_confidence,
        )
        self.detection_log.append(entry)

        # Clean up the processing file
        try:
            chunk.audio_path.unlink()
        except OSError:
            pass

        label = "DETECTION" if result.is_manatee else "background"
        logger.info(
            "%s: conf=%.2f clips=%d/%d (%.1fs inference) -> %s",
            label, result.confidence, result.clips_positive,
            result.clips_analyzed, inference_time, outbox_name,
        )
