"""
echoguard.modules.noise_cancellation.enhancer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Audio noise cancellation / enhancement for the EchoGuard pipeline.

Two enhancement strategies are supported and selected automatically:

1. **SpeechBrain** (``strategy="speechbrain"``)
   Uses the ``speechbrain/mtl-mimic-voicebank`` spectral-mask model for
   high-quality neural enhancement.  Falls back to strategy 2 if
   speechbrain is not installed.

2. **Classical DSP** (``strategy="dsp"``)
   Applies noisereduce (spectral subtraction) followed by a telephone
   band-pass filter (300-3400 Hz) and peak normalisation.  No GPU or
   large model required.

Running this as step 0 improves every downstream module:
  - Whisper gets cleaner speech → better transcript accuracy
  - wav2vec2 emotion model is more sensitive to clean signals
  - AASIST deepfake detector relies on fine spectral features that noise
    can obscure

Output contract
---------------
:func:`enhance_audio` returns the **path to the enhanced WAV file**.
All downstream modules (Whisper, wav2vec2, AASIST) receive this path
instead of the raw recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# SpeechBrain strategy
# ---------------------------------------------------------------------------

def _enhance_speechbrain(
    audio_path: Path,
    output_path: Path,
    savedir: str = "models/speechbrain_model",
) -> Path:
    """Enhance audio using SpeechBrain's SpectralMaskEnhancement model.

    Parameters
    ----------
    audio_path:
        Path to the input WAV/FLAC file.
    output_path:
        Destination WAV file path.
    savedir:
        Local directory where pretrained model weights are cached.

    Returns
    -------
    Path
        Path to the enhanced WAV file.
    """
    # Support both old and new speechbrain import paths
    try:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
    except ImportError:
        from speechbrain.pretrained import SpectralMaskEnhancement  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)

    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir=savedir,
    )
    enhancer.enhance_file(str(audio_path), str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Classical DSP strategy
# ---------------------------------------------------------------------------

def _enhance_dsp(
    audio_path: Path,
    output_path: Path,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
) -> Path:
    """Enhance audio using spectral noise reduction + band-pass filter.

    Steps
    -----
    1. Load audio with librosa (preserves original sample rate).
    2. Spectral noise reduction via ``noisereduce`` (spectral subtraction).
    3. Telephone-band band-pass filter (300-3400 Hz by default).
    4. Peak normalisation to [-1, 1].
    5. Save as 16-bit PCM WAV.

    Parameters
    ----------
    audio_path:
        Path to the input audio file.
    output_path:
        Destination WAV file path.
    lowcut:
        Low-cut frequency of the band-pass filter in Hz.
    highcut:
        High-cut frequency of the band-pass filter in Hz.

    Returns
    -------
    Path
        Path to the enhanced WAV file.
    """
    import numpy as np
    import librosa
    import soundfile as sf
    import noisereduce as nr
    from scipy.signal import butter, filtfilt

    # 1. Load (mono, original sample rate)
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)

    # 2. Spectral noise reduction
    reduced = nr.reduce_noise(y=audio, sr=sr)

    # 3. Band-pass filter (clamp to valid range to avoid errors)
    nyquist = 0.5 * sr
    low = max(0.001, min(lowcut / nyquist, 0.999))
    high = max(low + 0.001, min(highcut / nyquist, 0.999))
    b, a = butter(5, [low, high], btype="band")
    filtered = filtfilt(b, a, reduced)

    # 4. Peak normalise
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = filtered / peak

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), filtered, sr, subtype="PCM_16")
    return output_path


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class AudioEnhancer:
    """Noise cancellation wrapper for the EchoGuard pipeline.

    Parameters
    ----------
    strategy:
        ``"auto"`` (default) tries SpeechBrain first, falls back to
        ``"dsp"`` if speechbrain is not installed.
        Pass ``"speechbrain"`` or ``"dsp"`` to force a specific strategy.
    output_dir:
        Directory to write enhanced files to.  If ``None``, the enhanced
        file is saved next to the original with an ``_enhanced`` suffix.
    speechbrain_savedir:
        Local cache directory for the SpeechBrain pretrained weights.

    Example
    -------
    >>> enhancer = AudioEnhancer(strategy="dsp")
    >>> enhanced_path = enhancer.enhance("echoguard/audio/call001.wav")
    >>> print(enhanced_path)
    echoguard/audio/call001_enhanced.wav
    """

    def __init__(
        self,
        strategy: str = "auto",
        output_dir: Optional[str | Path] = None,
        speechbrain_savedir: str = "models/speechbrain_model",
    ) -> None:
        if strategy not in ("auto", "speechbrain", "dsp"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose from: 'auto', 'speechbrain', 'dsp'."
            )
        self.strategy = strategy
        self.output_dir = Path(output_dir) if output_dir else None
        self.speechbrain_savedir = speechbrain_savedir

    def _resolve_output_path(self, audio_path: Path) -> Path:
        filename = audio_path.stem + "_enhanced.wav"
        if self.output_dir:
            return self.output_dir / filename
        return audio_path.parent / filename

    @staticmethod
    def _has_speechbrain() -> bool:
        try:
            import speechbrain  # noqa: F401
            return True
        except ImportError:
            return False

    def enhance(self, audio_path: str | Path) -> str:
        """Denoise and enhance a single audio file.

        Parameters
        ----------
        audio_path:
            Path to the input audio file (WAV or FLAC, any sample rate).

        Returns
        -------
        str
            Path to the enhanced WAV file.

        Raises
        ------
        FileNotFoundError
            If ``audio_path`` does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_path = self._resolve_output_path(audio_path)

        active = self.strategy
        if active == "auto":
            active = "speechbrain" if self._has_speechbrain() else "dsp"

        if active == "speechbrain":
            try:
                _enhance_speechbrain(
                    audio_path,
                    output_path,
                    savedir=self.speechbrain_savedir,
                )
            except Exception as exc:
                print(
                    f"    [noise_cancellation] SpeechBrain failed ({exc}), "
                    "falling back to DSP strategy."
                )
                _enhance_dsp(audio_path, output_path)
        else:
            _enhance_dsp(audio_path, output_path)

        return str(output_path)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def enhance_audio(
    audio_path: str | Path,
    strategy: str = "auto",
    output_dir: Optional[str | Path] = None,
    speechbrain_savedir: str = "models/speechbrain_model",
) -> str:
    """Denoise and enhance an audio file before further analysis.

    This is the primary public interface for this module and is called by
    ``main.py`` as step 0, before Whisper, wav2vec2, and AASIST.

    Parameters
    ----------
    audio_path:
        Path to the input audio file (WAV or FLAC).
    strategy:
        ``"auto"`` | ``"speechbrain"`` | ``"dsp"``.
        ``"auto"`` uses SpeechBrain if installed, otherwise falls back to
        the classical DSP pipeline.
    output_dir:
        Directory to write the enhanced file.  Defaults to the same
        directory as the input file.
    speechbrain_savedir:
        Cache directory for SpeechBrain pretrained weights.

    Returns
    -------
    str
        Path to the enhanced WAV file (e.g. ``call001_enhanced.wav``).

    Example
    -------
    >>> from echoguard.modules.noise_cancellation import enhance_audio
    >>> enhanced = enhance_audio("echoguard/audio/call001.wav")
    >>> print(enhanced)
    echoguard/audio/call001_enhanced.wav
    """
    enhancer = AudioEnhancer(
        strategy=strategy,
        output_dir=output_dir,
        speechbrain_savedir=speechbrain_savedir,
    )
    return enhancer.enhance(audio_path)
