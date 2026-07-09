"""
echoguard.utils.audio
~~~~~~~~~~~~~~~~~~~~~
Shared audio I/O and preprocessing helpers.

All functions operate on file paths rather than raw arrays so that
modules remain loosely coupled — each module loads audio independently
using these helpers rather than passing tensors between modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np


def load_audio(
    audio_path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to the requested sample rate.

    Parameters
    ----------
    audio_path:
        Path to the WAV (or any librosa-supported) audio file.
    sample_rate:
        Target sample rate in Hz.  Defaults to 16 000 Hz, which is the
        standard for most speech models used in EchoGuard.
    mono:
        If ``True`` (default), mix down to a single channel.

    Returns
    -------
    audio : np.ndarray
        Float32 audio samples in the range [-1, 1].
    sr : int
        Actual sample rate of the returned array (equals ``sample_rate``).

    Raises
    ------
    FileNotFoundError
        If ``audio_path`` does not point to an existing file.
    """
    import librosa  # lazy import — keeps startup fast when audio unused

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=mono)
    return audio, sr


def save_audio(
    audio: np.ndarray,
    output_path: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """Write a float32 numpy array to a WAV file.

    Parameters
    ----------
    audio:
        Float32 audio samples.
    output_path:
        Destination file path.  Parent directories are created automatically.
    sample_rate:
        Sample rate of the audio data.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    import soundfile as sf  # lazy import

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)
    return output_path.resolve()


def get_duration(audio_path: str | Path) -> float:
    """Return the duration of an audio file in seconds without loading samples.

    Parameters
    ----------
    audio_path:
        Path to the audio file.

    Returns
    -------
    float
        Duration in seconds.
    """
    import librosa  # lazy import

    return librosa.get_duration(path=str(audio_path))


def validate_wav(audio_path: str | Path) -> bool:
    """Return ``True`` if the file exists and is a readable WAV file.

    Parameters
    ----------
    audio_path:
        Path to check.

    Returns
    -------
    bool
    """
    import soundfile as sf  # lazy import

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return False
    try:
        info = sf.info(str(audio_path))
        return info.channels >= 1
    except Exception:
        return False
