"""Configuration loading for the Manatee device client."""

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = "/etc/manatee/config.env"


def load_config_file(path: str) -> dict[str, str]:
    """Read KEY=VALUE lines from a config file. Skips comments and blank lines."""
    values = {}
    config_path = Path(path)
    if not config_path.exists():
        return values
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _get(file_values: dict[str, str], key: str, default: str | None = None) -> str | None:
    """Get a config value: env var overrides file value, then default."""
    return os.environ.get(key, file_values.get(key, default))


@dataclass
class Config:
    api_key: str
    server_url: str
    device_id: int | None = None
    heartbeat_interval: int = 300
    outbox_dir: str = "/var/lib/manatee/outbox"
    failed_dir: str = "/var/lib/manatee/failed"
    log_level: str = "INFO"
    retry_max: int = 5
    model_path: str = "/opt/manatee/manatee_model.pt"
    incoming_dir: str = "/var/lib/manatee/incoming"
    archive_dir: str = "/var/lib/manatee/archive"
    detection_log_path: str = "/var/lib/manatee/detections.jsonl"
    inference_enabled: bool = True
    audio_source: str = "hydrophone"  # "hydrophone" or "filewatcher"
    audio_device: str | None = None   # sounddevice device index/name; None = system default
    segment_duration: float = 30.0    # seconds per recording segment
    clip_threshold: float = 0.5              # per-clip sigmoid threshold for "positive"
    min_positive_clips: int = 10              # >= N positive clips → segment is a detection
    high_confidence_threshold: float = 0.90  # OR a single clip this confident → detection
    baseline_interval_sec: float = 900.0     # upload one background sample every N seconds
    config_path: str = field(default=DEFAULT_CONFIG_PATH, repr=False)

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> "Config":
        file_values = load_config_file(path)

        api_key = _get(file_values, "MANATEE_API_KEY")
        server_url = _get(file_values, "MANATEE_SERVER_URL")

        if not api_key:
            raise ValueError("MANATEE_API_KEY is required but not set")
        if not server_url:
            raise ValueError("MANATEE_SERVER_URL is required but not set")

        # Strip trailing slash from server URL
        server_url = server_url.rstrip("/")

        device_id_str = _get(file_values, "MANATEE_DEVICE_ID")
        device_id = int(device_id_str) if device_id_str else None

        return cls(
            api_key=api_key,
            server_url=server_url,
            device_id=device_id,
            heartbeat_interval=int(_get(file_values, "MANATEE_HEARTBEAT_INTERVAL", "300")),
            outbox_dir=_get(file_values, "MANATEE_OUTBOX_DIR", "/var/lib/manatee/outbox"),
            failed_dir=_get(file_values, "MANATEE_FAILED_DIR", "/var/lib/manatee/failed"),
            log_level=_get(file_values, "MANATEE_LOG_LEVEL", "INFO"),
            retry_max=int(_get(file_values, "MANATEE_RETRY_MAX", "5")),
            model_path=_get(file_values, "MANATEE_MODEL_PATH", "/opt/manatee/manatee_model.pt"),
            incoming_dir=_get(file_values, "MANATEE_INCOMING_DIR", "/var/lib/manatee/incoming"),
            archive_dir=_get(file_values, "MANATEE_ARCHIVE_DIR", "/var/lib/manatee/archive"),
            detection_log_path=_get(file_values, "MANATEE_DETECTION_LOG", "/var/lib/manatee/detections.jsonl"),
            inference_enabled=_get(file_values, "MANATEE_INFERENCE_ENABLED", "true").lower() in ("true", "1", "yes"),
            audio_source=_get(file_values, "MANATEE_AUDIO_SOURCE", "hydrophone"),
            audio_device=_get(file_values, "MANATEE_AUDIO_DEVICE") or None,
            segment_duration=float(_get(file_values, "MANATEE_SEGMENT_DURATION", "30")),
            clip_threshold=float(_get(file_values, "MANATEE_CLIP_THRESHOLD", "0.5")),
            min_positive_clips=int(_get(file_values, "MANATEE_MIN_POSITIVE_CLIPS", "10")),
            high_confidence_threshold=float(
                _get(file_values, "MANATEE_HIGH_CONFIDENCE_THRESHOLD", "0.90")
            ),
            baseline_interval_sec=float(
                _get(file_values, "MANATEE_BASELINE_INTERVAL_SEC", "60")
            ),
            config_path=path,
        )

    @property
    def api_key_prefix(self) -> str:
        """Safe prefix for logging (never log the full key)."""
        return self.api_key[:9] + "..." if len(self.api_key) > 9 else self.api_key


def save_device_id(path: str, device_id: int) -> None:
    """Append the discovered device ID to the config file."""
    with open(path, "a") as f:
        f.write(f"MANATEE_DEVICE_ID={device_id}\n")
