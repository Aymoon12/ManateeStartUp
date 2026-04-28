"""Spectrogram rendering on the device.

Mirrors the server's training-time SPECTROGRAM_CONFIG so the visuals that ship
with each upload match what the model sees. Uses the Figure + FigureCanvasAgg
pattern (no pyplot global state) to avoid the matplotlib leak that hit the
server side under long-running workers — the Pi has even less RAM to spare.
"""

import matplotlib
matplotlib.use("Agg")

import logging

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SPECTROGRAM_CONFIG = {
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 128,
    "window": "hann",
    "fmin": 500,
    "fmax": 20000,
    "power": 2.0,
}

SAMPLE_RATE = 44100


def _load_audio(file_path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(file_path, always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio, SAMPLE_RATE


def render_spectrogram_png(audio_path: str, output_path: str, title: str = "") -> None:
    """Render a mel-spectrogram PNG for the given audio file."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import librosa.display

    audio, sr = _load_audio(audio_path)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=SPECTROGRAM_CONFIG["n_mels"],
        n_fft=SPECTROGRAM_CONFIG["n_fft"],
        hop_length=SPECTROGRAM_CONFIG["hop_length"],
        window=SPECTROGRAM_CONFIG["window"],
        fmin=SPECTROGRAM_CONFIG["fmin"],
        fmax=SPECTROGRAM_CONFIG["fmax"],
        power=SPECTROGRAM_CONFIG["power"],
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, amin=1e-10)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

    fig = Figure(figsize=(6.4, 4.8))
    FigureCanvasAgg(fig)
    try:
        ax = fig.subplots()
        img = librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            hop_length=SPECTROGRAM_CONFIG["hop_length"],
            fmin=SPECTROGRAM_CONFIG["fmin"],
            fmax=SPECTROGRAM_CONFIG["fmax"],
            x_axis="time",
            y_axis="mel",
            cmap="viridis",
            ax=ax,
        )
        fig.colorbar(img, ax=ax, label="dB (normalized)")
        ax.set_ylabel("Frequency (Hz, mel scale)")
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=80)
    finally:
        fig.clear()
