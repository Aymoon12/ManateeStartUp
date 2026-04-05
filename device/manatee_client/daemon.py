"""Main daemon for the Manatee device client.

Sends heartbeats and processes the outbox on a configurable interval.
Designed to run under systemd (logs to stderr → journald).
"""

import logging
import os
import platform
import shutil
import signal
import socket
import sys
import threading
import time

from manatee_client import __version__
from manatee_client.config import Config, save_device_id
from manatee_client.api_client import ManateeAPIClient
from manatee_client.outbox import OutboxProcessor
from manatee_client.detection_log import DetectionLog

logger = logging.getLogger("manatee_client")

# Global flag for clean shutdown
running = True


def _handle_signal(signum, frame):
    global running
    logger.info("Received signal %d, shutting down...", signum)
    running = False


def _get_disk_space_gb() -> float:
    """Get free disk space on root filesystem in GB."""
    usage = shutil.disk_usage("/")
    return round(usage.free / (1024 ** 3), 2)


def _get_uptime_seconds() -> int | None:
    """Read system uptime from /proc/uptime. Returns None if unavailable."""
    try:
        with open("/proc/uptime") as f:
            return int(float(f.read().split()[0]))
    except (FileNotFoundError, ValueError, IndexError):
        return None


def _discover_device_id(client: ManateeAPIClient, config: Config) -> int:
    """Call /api/devices/me to discover device ID. Retries until success."""
    while running:
        try:
            info = client.get_device_info()
            device_id = info["id"]
            device_name = info.get("device_name", "unknown")
            logger.info("Discovered device: %s (ID: %d)", device_name, device_id)

            # Cache to config file so we don't need to look up again
            save_device_id(config.config_path, device_id)
            return device_id
        except Exception as e:
            logger.warning(
                "Failed to discover device ID (will retry in 60s): %s", e
            )
            # Sleep in small increments so we can respond to shutdown signals
            for _ in range(60):
                if not running:
                    sys.exit(0)
                time.sleep(1)

    sys.exit(0)


def main():
    global running

    # Determine config path (allow override for testing)
    config_path = os.environ.get("MANATEE_CONFIG_PATH", "/etc/manatee/config.env")

    # Load config
    try:
        config = Config.load(config_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    logger.info(
        "Manatee device client v%s starting (key: %s)",
        __version__,
        config.api_key_prefix,
    )

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Create API client
    client = ManateeAPIClient(config.server_url, config.api_key)

    # Discover device ID if not cached
    if config.device_id is not None:
        device_id = config.device_id
        logger.info("Using cached device ID: %d", device_id)
    else:
        logger.info("Device ID not cached, discovering via /api/devices/me...")
        device_id = _discover_device_id(client, config)

    # Create detection log (shared by recorder and outbox)
    detection_log = DetectionLog(config.detection_log_path)

    # Create outbox processor
    outbox = OutboxProcessor(
        api_client=client,
        outbox_dir=config.outbox_dir,
        failed_dir=config.failed_dir,
        retry_max=config.retry_max,
        detection_log=detection_log,
    )

    # Start recorder thread if inference is enabled
    stop_event = threading.Event()
    recorder_thread = None

    if config.inference_enabled:
        try:
            from manatee_client.inference import EdgeDetector
            from manatee_client.audio_source import FileWatcherSource
            from manatee_client.recorder import Recorder

            detector = EdgeDetector(config.model_path)
            detector.load_model()

            audio_source = FileWatcherSource(config.incoming_dir)
            recorder = Recorder(
                audio_source=audio_source,
                detector=detector,
                detection_log=detection_log,
                outbox_dir=config.outbox_dir,
                archive_dir=config.archive_dir,
            )

            recorder_thread = threading.Thread(
                target=recorder.run, args=(stop_event,), daemon=True,
            )
            recorder_thread.start()
            logger.info("Recorder thread started (incoming: %s)", config.incoming_dir)
        except Exception as e:
            logger.error("Failed to start recorder: %s", e)
            logger.info("Continuing without inference (outbox-only mode)")
    else:
        logger.info("Inference disabled, running in outbox-only mode")

    logger.info(
        "Entering main loop (heartbeat every %ds, outbox: %s)",
        config.heartbeat_interval,
        config.outbox_dir,
    )

    # Main loop
    while running:
        # Send heartbeat
        try:
            disk_gb = _get_disk_space_gb()
            uptime_s = _get_uptime_seconds()
            client.send_heartbeat(
                device_id=device_id,
                disk_space_free_gb=disk_gb,
                uptime_seconds=uptime_s,
                additional_info={
                    "client_version": __version__,
                    "hostname": socket.gethostname(),
                    "python_version": platform.python_version(),
                },
            )
            logger.debug("Heartbeat sent (disk: %.1f GB, uptime: %s s)", disk_gb, uptime_s)
        except Exception as e:
            logger.warning("Heartbeat failed: %s", e)

        # Process outbox
        try:
            outbox.process_all()
        except Exception as e:
            logger.warning("Outbox processing error: %s", e)

        # Sleep in small increments for responsive shutdown
        for _ in range(config.heartbeat_interval):
            if not running:
                break
            time.sleep(1)

    # Signal recorder to stop
    stop_event.set()
    if recorder_thread:
        recorder_thread.join(timeout=15)

    logger.info("Daemon stopped.")


if __name__ == "__main__":
    main()
