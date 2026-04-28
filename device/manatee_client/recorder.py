"""Recorder: audio source -> inference -> outbox + detection log.

Runs in its own thread, controlled by a stop_event. The audit trail
(detection_log) records every segment, but only positive detections and a
periodic baseline background sample are copied into the outbox for upload.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from threading import Event

import soundfile as sf

from manatee_client.audio_source import AudioSource
from manatee_client.detection_log import DetectionLog, DetectionLogEntry
from manatee_client.inference import EdgeDetector
from manatee_client.spectrogram import render_spectrogram_png

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(
        self,
        audio_source: AudioSource,
        detector: EdgeDetector,
        detection_log: DetectionLog,
        outbox_dir: str,
        archive_dir: str,
        baseline_interval_sec: float = 900.0,
    ):
        self.audio_source = audio_source
        self.detector = detector
        self.detection_log = detection_log
        self.outbox_dir = Path(outbox_dir)
        self.archive_dir = Path(archive_dir)
        self.baseline_interval_sec = baseline_interval_sec
        # Initialized to 0 so the first non-detection segment after startup
        # uploads as a baseline — proves the upload pipe works on boot.
        self._last_baseline_at = 0.0
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
        ext = chunk.audio_path.suffix.lower() or ".wav"

        # Decide whether this segment goes to the outbox.
        now = time.monotonic()
        if result.is_manatee:
            outbox_name = f"detection_{ts}_{result.confidence:.2f}{ext}"
            should_upload = True
            kind = "DETECTION"
        elif now - self._last_baseline_at >= self.baseline_interval_sec:
            outbox_name = f"background_{ts}{ext}"
            should_upload = True
            self._last_baseline_at = now
            kind = "baseline"
        else:
            outbox_name = None
            should_upload = False
            kind = "skipped"

        # Always archive (SD card retention). Always log (audit trail).
        archive_path = self.archive_dir / f"{ts}_{('detection' if result.is_manatee else 'background')}{ext}"
        shutil.copy2(str(chunk.audio_path), str(archive_path))

        outbox_path = None
        if should_upload:
            outbox_path = self.outbox_dir / outbox_name
            shutil.copy2(str(chunk.audio_path), str(outbox_path))

            # Render spectrogram alongside the audio. If rendering fails the
            # upload still proceeds — the server tolerates a missing PNG.
            spec_path = outbox_path.with_suffix(outbox_path.suffix + ".spec.png")
            try:
                render_spectrogram_png(
                    str(chunk.audio_path),
                    str(spec_path),
                    title=f"{kind} {ts_iso}",
                )
            except Exception as e:
                logger.warning("Spectrogram render failed for %s: %s", outbox_name, e)
                if spec_path.exists():
                    spec_path.unlink(missing_ok=True)

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

        # Append to detection log (records every segment regardless)
        entry = DetectionLogEntry(
            timestamp=ts_iso,
            confidence=result.confidence,
            is_manatee=result.is_manatee,
            audio_file=str(archive_path),
            outbox_file=str(outbox_path) if outbox_path else "",
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

        if should_upload:
            logger.info(
                "%s: conf=%.2f clips=%d/%d (%.1fs inference) -> %s",
                kind, result.confidence, result.clips_positive,
                result.clips_analyzed, inference_time, outbox_name,
            )
        else:
            logger.info(
                "skipped: conf=%.2f clips=%d/%d (next baseline in %.0fs)",
                result.confidence, result.clips_positive, result.clips_analyzed,
                self.baseline_interval_sec - (now - self._last_baseline_at),
            )
