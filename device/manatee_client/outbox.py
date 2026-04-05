"""Outbox processor — scans a directory for audio files and uploads them."""

import json
import logging
import os
import re
import shutil
from pathlib import Path

from manatee_client.api_client import ManateeAPIClient

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

# Filename patterns:
#   detection_2026-04-04T12-30-00Z_0.87.wav
#   background_2026-04-04T12-35-00Z.wav
DETECTION_RE = re.compile(
    r"^detection_(.+?)_(\d+\.\d+)\.(" + "|".join(e.lstrip(".") for e in AUDIO_EXTENSIONS) + r")$"
)
BACKGROUND_RE = re.compile(
    r"^background_(.+?)\.(" + "|".join(e.lstrip(".") for e in AUDIO_EXTENSIONS) + r")$"
)


def _parse_timestamp(raw: str) -> str:
    """Convert filesystem-safe timestamp back to ISO 8601 (hyphens in time → colons)."""
    # Input:  2026-04-04T12-30-00Z
    # Output: 2026-04-04T12:30:00Z
    # Only replace hyphens after the T (time portion)
    t_pos = raw.find("T")
    if t_pos == -1:
        return raw
    date_part = raw[:t_pos]
    time_part = raw[t_pos + 1:]
    time_part = time_part.replace("-", ":")
    return f"{date_part}T{time_part}"


def _read_retry_count(audio_path: Path) -> int:
    """Read the retry counter from a .retry sidecar file."""
    retry_path = audio_path.with_suffix(audio_path.suffix + ".retry")
    if retry_path.exists():
        try:
            return int(retry_path.read_text().strip())
        except (ValueError, OSError):
            return 0
    return 0


def _write_retry_count(audio_path: Path, count: int) -> None:
    """Write the retry counter to a .retry sidecar file."""
    retry_path = audio_path.with_suffix(audio_path.suffix + ".retry")
    retry_path.write_text(str(count))


def _read_meta(audio_path: Path) -> dict:
    """Read optional .meta JSON sidecar. Returns empty dict if absent/invalid."""
    meta_path = audio_path.with_suffix(audio_path.suffix + ".meta")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", meta_path.name, e)
    return {}


def _cleanup(audio_path: Path) -> None:
    """Remove the audio file and all sidecar files."""
    for suffix in ["", ".meta", ".retry"]:
        p = audio_path.with_suffix(audio_path.suffix + suffix) if suffix else audio_path
        if p.exists():
            p.unlink()


def _move_to_failed(audio_path: Path, failed_dir: Path) -> None:
    """Move the audio file and sidecars to the failed directory."""
    failed_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["", ".meta", ".retry"]:
        p = audio_path.with_suffix(audio_path.suffix + suffix) if suffix else audio_path
        if p.exists():
            shutil.move(str(p), str(failed_dir / p.name))


class OutboxProcessor:
    def __init__(
        self,
        api_client: ManateeAPIClient,
        outbox_dir: str,
        failed_dir: str,
        retry_max: int = 5,
        detection_log=None,
    ):
        self.api_client = api_client
        self.outbox_dir = Path(outbox_dir)
        self.failed_dir = Path(failed_dir)
        self.retry_max = retry_max
        self.detection_log = detection_log

    def scan(self) -> list[Path]:
        """List audio files in the outbox, sorted oldest-first."""
        if not self.outbox_dir.exists():
            return []
        files = [
            f for f in self.outbox_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ]
        files.sort(key=lambda f: f.stat().st_mtime)
        return files

    def process_file(self, audio_path: Path) -> bool:
        """Process a single outbox file. Returns True on success."""
        name = audio_path.name
        meta = _read_meta(audio_path)

        # Try detection pattern first
        det_match = DETECTION_RE.match(name)
        if det_match:
            timestamp = _parse_timestamp(det_match.group(1))
            confidence = float(det_match.group(2))
            try:
                self.api_client.upload_detection(
                    timestamp=timestamp,
                    confidence=confidence,
                    audio_path=str(audio_path),
                    audio_duration=meta.get("audio_duration"),
                    extra_metadata=meta.get("extra_metadata"),
                )
                logger.info("Uploaded detection: %s", name)
                if self.detection_log:
                    self.detection_log.mark_uploaded(timestamp, name)
                _cleanup(audio_path)
                return True
            except Exception as e:
                logger.warning("Failed to upload %s: %s", name, e)
                self._handle_failure(audio_path)
                return False

        # Try background pattern
        bg_match = BACKGROUND_RE.match(name)
        if bg_match:
            timestamp = _parse_timestamp(bg_match.group(1))
            try:
                self.api_client.upload_background(
                    timestamp=timestamp,
                    audio_path=str(audio_path),
                    audio_duration=meta.get("audio_duration"),
                    extra_metadata=meta.get("extra_metadata"),
                )
                logger.info("Uploaded background: %s", name)
                if self.detection_log:
                    self.detection_log.mark_uploaded(timestamp, name)
                _cleanup(audio_path)
                return True
            except Exception as e:
                logger.warning("Failed to upload %s: %s", name, e)
                self._handle_failure(audio_path)
                return False

        logger.warning("Skipping unrecognized filename: %s", name)
        return False

    def _handle_failure(self, audio_path: Path) -> None:
        """Increment retry counter; move to failed dir if max exceeded."""
        count = _read_retry_count(audio_path) + 1
        if count >= self.retry_max:
            logger.error(
                "Max retries (%d) reached for %s — moving to failed",
                self.retry_max,
                audio_path.name,
            )
            _move_to_failed(audio_path, self.failed_dir)
        else:
            _write_retry_count(audio_path, count)
            logger.info("Retry %d/%d for %s", count, self.retry_max, audio_path.name)

    def process_all(self) -> tuple[int, int]:
        """Process all files in the outbox. Returns (successes, failures)."""
        files = self.scan()
        if not files:
            return 0, 0

        successes = 0
        failures = 0
        for f in files:
            if self.process_file(f):
                successes += 1
            else:
                failures += 1

        if successes or failures:
            logger.info("Outbox: %d uploaded, %d failed", successes, failures)
        return successes, failures