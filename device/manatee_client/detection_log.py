"""Append-only JSONL detection log for offline SD card retrieval."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectionLogEntry:
    timestamp: str
    confidence: float
    is_manatee: bool
    audio_file: str
    outbox_file: str
    clips_analyzed: int
    clips_positive: int
    max_confidence: float
    uploaded: bool = False
    uploaded_at: Optional[str] = None


class DetectionLog:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: DetectionLogEntry) -> None:
        """Append a detection entry as a single JSON line."""
        line = json.dumps(asdict(entry), default=str)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def mark_uploaded(self, timestamp: str, audio_filename: str) -> None:
        """Mark a detection as uploaded by rewriting its line.

        Reads the full file, updates matching entry, writes back.
        Safe for small-to-medium log files on Pi.
        """
        if not self.log_path.exists():
            return

        lines = self.log_path.read_text().splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                new_lines.append(line)
                continue

            if (entry.get("timestamp") == timestamp
                    and Path(entry.get("audio_file", "")).name == audio_filename
                    and not entry.get("uploaded")):
                entry["uploaded"] = True
                entry["uploaded_at"] = datetime.now(timezone.utc).isoformat()
                updated = True

            new_lines.append(json.dumps(entry, default=str))

        if updated:
            self.log_path.write_text("\n".join(new_lines) + "\n")
            logger.debug("Marked uploaded: %s / %s", timestamp, audio_filename)

    def read_all(self) -> list[dict]:
        """Read all entries (for SD card export / diagnostics)."""
        if not self.log_path.exists():
            return []
        entries = []
        for line in self.log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
