"""Edge ML inference for manatee vocalization detection.

Ported from ManateeWebsiteBackend/app/services/ml_service.py and
ManateeWebsiteBackend/app/ml/audio_utils.py.

CRITICAL: Spectrogram parameters and model architecture MUST match
the server version exactly to get identical results.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Disable torch dynamo/compiler — unnecessary on Pi CPU and causes
# 'get_call_template' errors on PyTorch 2.5+.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

logger = logging.getLogger(__name__)

# ── Audio / spectrogram config (EXACT copy from server) ───────────────

SPECTROGRAM_CONFIG = {
    'n_mels': 128,
    'n_fft': 1024,
    'hop_length': 128,
    'window': 'hann',
    'fmin': 500,
    'fmax': 20000,
    'power': 2.0,
}

SAMPLE_RATE = 44100
CLIP_DURATION = 0.5  # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 22050


# ── Audio utilities ───────────────────────────────────────────────────

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Load audio file: mono, resampled to 44.1 kHz."""
    audio, sr = sf.read(file_path, always_2d=False)

    # Multi-channel: take first channel only (DO NOT average)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return audio, SAMPLE_RATE


def extract_clips(audio: np.ndarray, sr: int = SAMPLE_RATE,
                  stride: float = 0.5) -> list[tuple[np.ndarray, float, float]]:
    """Extract 0.5s clips from audio using sliding window."""
    clips = []
    stride_samples = int(stride * sr)

    for start_sample in range(0, len(audio), stride_samples):
        end_sample = start_sample + CLIP_SAMPLES

        if end_sample > len(audio):
            if start_sample >= len(audio):
                break
            clip = np.zeros(CLIP_SAMPLES, dtype=audio.dtype)
            available = len(audio) - start_sample
            clip[:available] = audio[start_sample:]
        else:
            clip = audio[start_sample:end_sample]

        start_time = start_sample / sr
        end_time = min((start_sample + CLIP_SAMPLES) / sr, len(audio) / sr)
        clips.append((clip, start_time, end_time))

    return clips


def compute_spectrogram(audio_segment: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Compute mel spectrogram matching EXACT training configuration.

    Returns torch.Tensor of shape [1, 128, 173].
    """
    # Normalize audio to [-1, 1]
    audio_segment = audio_segment / (np.max(np.abs(audio_segment)) + 1e-8)

    # Compute mel spectrogram with EXACT training parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=sr,
        n_mels=SPECTROGRAM_CONFIG['n_mels'],
        n_fft=SPECTROGRAM_CONFIG['n_fft'],
        hop_length=SPECTROGRAM_CONFIG['hop_length'],
        window=SPECTROGRAM_CONFIG['window'],
        fmin=SPECTROGRAM_CONFIG['fmin'],
        fmax=SPECTROGRAM_CONFIG['fmax'],
        power=SPECTROGRAM_CONFIG['power'],
    )

    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, amin=1e-10)

    # Per-clip normalization (zero mean, unit variance)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

    # Convert to torch tensor [1, 128, 173]
    spec_tensor = torch.from_numpy(mel_spec_db).float().unsqueeze(0)

    return spec_tensor


# ── Model architecture (EXACT copy from server) ──────────────────────

class ManateeClassifier(nn.Module):
    """MobileNetV3-Large based manatee vocalization classifier."""

    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v3_large(weights=None)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(960, 1),
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ── Inference result ──────────────────────────────────────────────────

@dataclass
class InferenceResult:
    confidence: float
    is_manatee: bool
    clips_analyzed: int
    clips_positive: int
    avg_positive_confidence: float
    max_confidence: float


# ── Edge detector ─────────────────────────────────────────────────────

class EdgeDetector:
    """Runs manatee inference on CPU (Raspberry Pi)."""

    MAX_CLIPS = 60
    CLIP_THRESHOLD = 0.5
    MIN_POSITIVE_CLIPS = 15
    HIGH_CONFIDENCE_THRESHOLD = 1.01  # disables OR rule; only count rule fires

    def __init__(
        self,
        model_path: str,
        clip_threshold: float | None = None,
        min_positive_clips: int | None = None,
        high_confidence_threshold: float | None = None,
    ):
        self.model_path = Path(model_path)
        self.model: nn.Module | None = None
        self.device = torch.device("cpu")
        self.clip_threshold = (
            clip_threshold if clip_threshold is not None else self.CLIP_THRESHOLD
        )
        self.min_positive_clips = (
            min_positive_clips if min_positive_clips is not None else self.MIN_POSITIVE_CLIPS
        )
        self.high_confidence_threshold = (
            high_confidence_threshold
            if high_confidence_threshold is not None
            else self.HIGH_CONFIDENCE_THRESHOLD
        )

    def load_model(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = ManateeClassifier()
        state_dict = torch.load(
            str(self.model_path), map_location=self.device, weights_only=False,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded from %s (CPU)", self.model_path)

    def predict_file(self, audio_path: str) -> InferenceResult:
        """Run inference on an audio file."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio, sr = load_audio(audio_path)
        clips = extract_clips(audio, sr, stride=0.5)

        if not clips:
            return InferenceResult(
                confidence=0.0, is_manatee=False,
                clips_analyzed=0, clips_positive=0,
                avg_positive_confidence=0.0, max_confidence=0.0,
            )

        # Cap clips to bound inference time
        clips = clips[:self.MAX_CLIPS]

        clip_confidences: list[float] = []

        with torch.no_grad():
            for clip_audio, _, _ in clips:
                spec = compute_spectrogram(clip_audio, sr)  # [1, 128, 173]
                # Model expects 3 channels — repeat spectrogram 3x
                spec = spec.repeat(3, 1, 1)  # [3, 128, 173]
                spec = spec.unsqueeze(0)  # [1, 3, 128, 173]

                output = self.model(spec)
                prob = torch.sigmoid(output).item()
                clip_confidences.append(prob)

        positive_confidences = [c for c in clip_confidences if c > self.clip_threshold]
        clips_positive = len(positive_confidences)
        max_conf = max(clip_confidences)

        # Event-aware rule for sparse calls in continuous monitoring windows:
        # sustained activity (>=N positive clips) OR a single very confident clip.
        is_manatee = (
            clips_positive >= self.min_positive_clips
            or max_conf >= self.high_confidence_threshold
        )

        if is_manatee:
            confidence = (
                sum(positive_confidences) / clips_positive
                if positive_confidences
                else max_conf
            )
        else:
            confidence = sum(clip_confidences) / len(clip_confidences)

        avg_pos = (sum(positive_confidences) / clips_positive) if clips_positive > 0 else 0.0

        return InferenceResult(
            confidence=round(confidence, 4),
            is_manatee=is_manatee,
            clips_analyzed=len(clip_confidences),
            clips_positive=clips_positive,
            avg_positive_confidence=round(avg_pos, 4),
            max_confidence=round(max_conf, 4),
        )
