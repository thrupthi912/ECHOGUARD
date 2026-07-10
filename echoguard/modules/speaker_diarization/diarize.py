"""
echoguard.modules.speaker_diarization.diarize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Speaker diarization using the pretrained pyannote.audio pipeline.

This module wraps ``pyannote/speaker-diarization-3.1`` from Hugging Face.
No model training is performed here — the pretrained pipeline is used
directly for inference.

Requirements
------------
* ``pyannote.audio >= 3.1.0``  (pip install pyannote.audio)
* A Hugging Face account with accepted terms of use for:
    - pyannote/speaker-diarization-3.1
    - pyannote/segmentation-3.0
  Visit https://hf.co/pyannote/speaker-diarization-3.1 and accept the
  conditions, then set the HF_TOKEN environment variable.

Public interface
----------------
    segments = diarize("call001.wav")

    # Returns a list of segment dicts:
    # [
    #   {"speaker": "SPEAKER_00", "start": 0.000, "end": 2.345},
    #   {"speaker": "SPEAKER_01", "start": 2.500, "end": 5.100},
    #   ...
    # ]

    # Side-effects (optional, controlled by save_json / save_rttm):
    #   echoguard/outputs/<stem>.diarization.json
    #   echoguard/outputs/<stem>.rttm
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Module-level lazy pipeline cache — loaded once per process.
# ---------------------------------------------------------------------------
_pipeline = None


def _load_pipeline(model_id: str, hf_token: Optional[str]) -> object:
    """Load (or return cached) pyannote diarization pipeline.

    Parameters
    ----------
    model_id:
        HuggingFace model repository ID.
    hf_token:
        HuggingFace access token.  Falls back to the ``HF_TOKEN``
        environment variable if ``None``.

    Returns
    -------
    pyannote.audio.Pipeline
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is not installed.  "
            "Run: pip install pyannote.audio"
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "A Hugging Face token is required to load the pyannote pipeline.\n"
            "Set it via the HF_TOKEN environment variable or pass it as "
            "hf_token in the config.\n"
            "Get a token at https://huggingface.co/settings/tokens and accept "
            "the model terms at https://hf.co/pyannote/speaker-diarization-3.1"
        )

    _pipeline = Pipeline.from_pretrained(model_id, token=token)
    return _pipeline


def _annotation_to_segments(annotation) -> List[Dict]:
    """Convert a pyannote ``Annotation`` object to a list of dicts.

    Parameters
    ----------
    annotation:
        ``pyannote.core.Annotation`` returned by the diarization pipeline.

    Returns
    -------
    list of dict
        Each element: ``{"speaker": str, "start": float, "end": float}``
        sorted by start time.
    """
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            }
        )
    segments.sort(key=lambda s: s["start"])
    return segments


def _save_json(segments: List[Dict], output_path: Path) -> None:
    """Persist diarization segments as a JSON file.

    Parameters
    ----------
    segments:
        List of segment dicts produced by :func:`_annotation_to_segments`.
    output_path:
        Destination ``.json`` file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(segments, fh, indent=2, ensure_ascii=False)


def _save_rttm(segments: List[Dict], audio_path: Path, output_path: Path) -> None:
    """Write diarization output in RTTM format for reproducibility.

    RTTM (Rich Transcription Time Marked) is the standard evaluation
    format used in DIHARD and VoxSRC speaker diarization challenges.

    Format per line::

        SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>

    Parameters
    ----------
    segments:
        Diarization segments.
    audio_path:
        Original audio file path — used to derive the file ID field.
    output_path:
        Destination ``.rttm`` file path.
    """
    file_id = audio_path.stem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for seg in segments:
            duration = round(seg["end"] - seg["start"], 3)
            fh.write(
                f"SPEAKER {file_id} 1 {seg['start']:.3f} {duration:.3f} "
                f"<NA> <NA> {seg['speaker']} <NA> <NA>\n"
            )


class Diarizer:
    """Object-oriented interface to the pyannote diarization pipeline.

    Useful when processing many files in a batch — the pipeline is loaded
    only once and reused across calls.

    Parameters
    ----------
    model_id:
        HuggingFace model repository ID.
        Defaults to ``"pyannote/speaker-diarization-3.1"``.
    hf_token:
        HuggingFace access token.  Falls back to ``HF_TOKEN`` env var.
    output_dir:
        Directory where JSON and RTTM outputs are saved.
        Defaults to ``"echoguard/outputs"``.
    min_segment_duration:
        Segments shorter than this value (in seconds) are discarded.

    Example
    -------
    >>> d = Diarizer()
    >>> segments = d.diarize("call001.wav")
    >>> for seg in segments:
    ...     print(seg["speaker"], seg["start"], seg["end"])
    """

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        output_dir: str | Path = "echoguard/outputs",
        min_segment_duration: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.output_dir = Path(output_dir)
        self.min_segment_duration = min_segment_duration
        self._pipeline = None  # loaded lazily on first call

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = _load_pipeline(self.model_id, self.hf_token)
        return self._pipeline

    def diarize(
        self,
        audio_path: str | Path,
        save_json: bool = True,
        save_rttm: bool = True,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict]:
        """Run speaker diarization on a WAV file.

        Parameters
        ----------
        audio_path:
            Path to the input audio file (WAV recommended, 16 kHz mono).
        save_json:
            If ``True``, save segments as ``<output_dir>/<stem>.diarization.json``.
        save_rttm:
            If ``True``, save segments as ``<output_dir>/<stem>.rttm``.
        num_speakers:
            Exact number of speakers if known.  Overrides min/max when set.
        min_speakers:
            Minimum number of speakers hint.
        max_speakers:
            Maximum number of speakers hint.

        Returns
        -------
        list of dict
            ``[{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.3}, ...]``
            sorted by start time, with short segments filtered out.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        pipeline = self._get_pipeline()

        # Build optional speaker-count hints accepted by pyannote 3.x
        kwargs: Dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

        annotation = pipeline(str(audio_path), **kwargs)
        segments = _annotation_to_segments(annotation)

        # Filter very short segments that are usually noise
        segments = [
            s for s in segments
            if (s["end"] - s["start"]) >= self.min_segment_duration
        ]

        if save_json:
            json_path = self.output_dir / f"{audio_path.stem}.diarization.json"
            _save_json(segments, json_path)

        if save_rttm:
            rttm_path = self.output_dir / f"{audio_path.stem}.rttm"
            _save_rttm(segments, audio_path, rttm_path)

        return segments


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def diarize(
    audio_path: str | Path,
    model_id: str = "pyannote/speaker-diarization-3.1",
    hf_token: Optional[str] = None,
    output_dir: str | Path = "echoguard/outputs",
    save_json: bool = True,
    save_rttm: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    min_segment_duration: float = 0.5,
) -> List[Dict]:
    """Perform speaker diarization on a WAV file.

    This is the primary public interface for this module.  It accepts
    an audio path and returns structured speaker segment information.

    Parameters
    ----------
    audio_path:
        Path to the input WAV file.
    model_id:
        HuggingFace model ID for the pyannote pipeline.
    hf_token:
        HuggingFace access token.  Falls back to ``HF_TOKEN`` env var.
    output_dir:
        Directory where JSON and RTTM outputs are written.
    save_json:
        Persist segments as a JSON file alongside the audio.
    save_rttm:
        Persist segments in RTTM format for evaluation tooling.
    num_speakers:
        Known exact number of speakers (optional).
    min_speakers:
        Lower bound on speaker count hint (optional).
    max_speakers:
        Upper bound on speaker count hint (optional).
    min_segment_duration:
        Minimum segment length in seconds; shorter ones are dropped.

    Returns
    -------
    list of dict
        Sorted list of speaker segments::

            [
                {"speaker": "SPEAKER_00", "start": 0.000, "end": 2.345},
                {"speaker": "SPEAKER_01", "start": 2.500, "end": 5.100},
                ...
            ]

    Example
    -------
    >>> from echoguard.modules.speaker_diarization import diarize
    >>> segments = diarize("echoguard/audio/call001.wav")
    >>> for seg in segments:
    ...     print(f"{seg['speaker']}  {seg['start']:.1f}s – {seg['end']:.1f}s")
    SPEAKER_00  0.0s – 2.3s
    SPEAKER_01  2.5s – 5.1s
    """
    d = Diarizer(
        model_id=model_id,
        hf_token=hf_token,
        output_dir=output_dir,
        min_segment_duration=min_segment_duration,
    )
    return d.diarize(
        audio_path=audio_path,
        save_json=save_json,
        save_rttm=save_rttm,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
