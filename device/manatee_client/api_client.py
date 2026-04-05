"""HTTP client for the Manatee backend API."""

import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 10   # seconds
READ_TIMEOUT = 120     # seconds (large audio files on slow connections)


class ManateeAPIClient:
    def __init__(self, server_url: str, api_key: str):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers["X-Device-Key"] = api_key

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def get_device_info(self) -> dict:
        """GET /api/devices/me — discover this device's ID and metadata."""
        resp = self._session.get(
            self._url("/api/devices/me"),
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        resp.raise_for_status()
        return resp.json()

    def send_heartbeat(
        self,
        device_id: int,
        disk_space_free_gb: float | None = None,
        uptime_seconds: int | None = None,
        additional_info: dict | None = None,
    ) -> dict:
        """POST /api/devices/{device_id}/heartbeat"""
        payload: dict = {}
        if disk_space_free_gb is not None:
            payload["disk_space_free_gb"] = disk_space_free_gb
        if uptime_seconds is not None:
            payload["uptime_seconds"] = uptime_seconds
        if additional_info:
            payload["additional_info"] = additional_info

        resp = self._session.post(
            self._url(f"/api/devices/{device_id}/heartbeat"),
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        resp.raise_for_status()
        return resp.json()

    def upload_detection(
        self,
        timestamp: str,
        confidence: float,
        audio_path: str,
        audio_duration: float | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        """POST /api/upload/detection — multipart form upload."""
        path = Path(audio_path)
        data: dict = {
            "timestamp": timestamp,
            "confidence": str(confidence),
        }
        if audio_duration is not None:
            data["audio_duration"] = str(audio_duration)
        if extra_metadata:
            data["extra_metadata"] = json.dumps(extra_metadata)

        with open(path, "rb") as f:
            files = {"audio_file": (path.name, f, "application/octet-stream")}
            resp = self._session.post(
                self._url("/api/upload/detection"),
                data=data,
                files=files,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
        resp.raise_for_status()
        return resp.json()

    def upload_background(
        self,
        timestamp: str,
        audio_path: str,
        audio_duration: float | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        """POST /api/upload/background — multipart form upload."""
        path = Path(audio_path)
        data: dict = {
            "timestamp": timestamp,
        }
        if audio_duration is not None:
            data["audio_duration"] = str(audio_duration)
        if extra_metadata:
            data["extra_metadata"] = json.dumps(extra_metadata)

        with open(path, "rb") as f:
            files = {"audio_file": (path.name, f, "application/octet-stream")}
            resp = self._session.post(
                self._url("/api/upload/background"),
                data=data,
                files=files,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
        resp.raise_for_status()
        return resp.json()
