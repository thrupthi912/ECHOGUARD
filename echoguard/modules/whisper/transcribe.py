"""
echoguard.modules.whisper.transcribe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Speech-to-text and speaker-aware transcription using OpenAI Whisper.

This module provides three levels of transcription:

1. ``transcribe()``               — plain text transcript
2. ``transcribe_with_timestamps()`` — text + word-level timestamps
3. ``diarized_transcribe()``      — speaker-labelled transcript by merging
                                    Whisper word timestamps with pyannote
                                    speaker diarization segments

The diarized pipeline:

    Audio
      │
      ├─► Speaker Diarization (pyannote)  →  speaker segments
      │
      └─► Whisper (word timestamps)       →  words with [start, end]
              │
              └─► Merge: assign each word to its speaker
                      │
                      └─► Speaker-aware transcript
                              │
                              ├─► saved to diarized_transcripts/<stem>.txt
                              └─► returned as list[dict]

Whisper and pyannote are intentionally decoupled — this module calls
the diarization module through its public interface only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level model cache — Whisper models are expensive to load.
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_model_name: Optional[str] = None


def _load_model(model_name: str):
    """Load (or return cached) Whisper model.

    Parameters
    ----------
    model_name:
        Whisper model size: ``tiny | base | small | medium | large``.

    Returns
    -------
    whisper.Whisper
    """
    global _whisper_model, _whisper_model_name

    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model

    try:
        import whisper
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is not installed.  Run: pip install openai-whisper"
        ) from exc

    _whisper_model = whisper.load_model(model_name)
    _whisper_model_name = model_name
    return _whisper_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_words_to_speakers(
    words: List[Dict],
    diarization_segments: List[Dict],
) -> List[Dict]:
    """Assign each Whisper word to the speaker active at its midpoint.

    Uses the midpoint of each word's ``[start, end]`` interval to find the
    overlapping diarization segment.  Words that fall outside all speaker
    segments (silence / overlap) are assigned to ``"UNKNOWN"``.

    Parameters
    ----------
    words:
        List of word dicts from Whisper with keys ``word``, ``start``, ``end``.
    diarization_segments:
        List of dicts from the diarization module:
        ``[{"speaker": str, "start": float, "end": float}, ...]``

    Returns
    -------
    list of dict
        Each element: ``{"word": str, "start": float, "end": float,
        "speaker": str}``
    """
    assigned = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        speaker = "UNKNOWN"
        for seg in diarization_segments:
            if seg["start"] <= mid <= seg["end"]:
                speaker = seg["speaker"]
                break
        assigned.append(
            {
                "word": w["word"].strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
                "speaker": speaker,
            }
        )
    return assigned


def _group_words_by_speaker(assigned_words: List[Dict]) -> List[Dict]:
    """Group consecutive words belonging to the same speaker into segments.

    Parameters
    ----------
    assigned_words:
        Output of :func:`_assign_words_to_speakers`.

    Returns
    -------
    list of dict
        Each element::

            {
                "speaker": "SPEAKER_00",
                "text": "Hello sir .",
                "start": 0.0,
                "end": 1.4,
            }
    """
    if not assigned_words:
        return []

    segments: List[Dict] = []
    current_speaker = assigned_words[0]["speaker"]
    buffer_words: List[str] = []
    seg_start = assigned_words[0]["start"]
    seg_end = assigned_words[0]["end"]

    for w in assigned_words:
        if w["speaker"] == current_speaker:
            buffer_words.append(w["word"])
            seg_end = w["end"]
        else:
            segments.append(
                {
                    "speaker": current_speaker,
                    "text": " ".join(buffer_words).strip(),
                    "start": seg_start,
                    "end": seg_end,
                }
            )
            current_speaker = w["speaker"]
            buffer_words = [w["word"]]
            seg_start = w["start"]
            seg_end = w["end"]

    # flush last group
    if buffer_words:
        segments.append(
            {
                "speaker": current_speaker,
                "text": " ".join(buffer_words).strip(),
                "start": seg_start,
                "end": seg_end,
            }
        )

    return segments


def _save_transcript(text: str, output_path: Path) -> None:
    """Write a transcript string to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _save_diarized_json(segments: List[Dict], output_path: Path) -> None:
    """Write diarized segments to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(segments, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# WhisperTranscriber class
# ---------------------------------------------------------------------------

class WhisperTranscriber:
    """Stateful Whisper transcriber that reuses a loaded model.

    Parameters
    ----------
    model_name:
        Whisper model size.  Defaults to ``"base"``.
    language:
        Language code (e.g. ``"en"``) or ``None`` for auto-detection.
    transcripts_dir:
        Directory to save plain transcripts.
    diarized_transcripts_dir:
        Directory to save speaker-aware transcripts.

    Example
    -------
    >>> t = WhisperTranscriber(model_name="small")
    >>> text = t.transcribe("call001.wav")
    >>> segments = t.diarized_transcribe("call001.wav")
    """

    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = None,
        transcripts_dir: str | Path = "echoguard/transcripts",
        diarized_transcripts_dir: str | Path = "echoguard/diarized_transcripts",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.transcripts_dir = Path(transcripts_dir)
        self.diarized_transcripts_dir = Path(diarized_transcripts_dir)

    def _model(self):
        return _load_model(self.model_name)

    def transcribe(
        self,
        audio_path: str | Path,
        save: bool = True,
    ) -> str:
        """Transcribe audio to plain text.

        Parameters
        ----------
        audio_path:
            Path to the input audio file.
        save:
            If ``True``, write the transcript to
            ``transcripts_dir/<stem>.txt``.

        Returns
        -------
        str
            Plain text transcript.
        """
        audio_path = Path(audio_path)
        model = self._model()

        kwargs: Dict = {}
        if self.language:
            kwargs["language"] = self.language

        result = model.transcribe(str(audio_path), **kwargs)
        text: str = result["text"].strip()

        if save:
            out = self.transcripts_dir / f"{audio_path.stem}.txt"
            _save_transcript(text, out)

        return text

    def transcribe_with_timestamps(
        self,
        audio_path: str | Path,
    ) -> Dict:
        """Transcribe audio and return word-level timestamps.

        Parameters
        ----------
        audio_path:
            Path to the input audio file.

        Returns
        -------
        dict
            Whisper result dict with an additional ``"words"`` key
            containing ``[{"word": str, "start": float, "end": float}]``.
        """
        audio_path = Path(audio_path)
        model = self._model()

        kwargs: Dict = {"word_timestamps": True}
        if self.language:
            kwargs["language"] = self.language

        result = model.transcribe(str(audio_path), **kwargs)

        # Flatten word-level timestamps from all segments
        words: List[Dict] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append(
                    {
                        "word": w["word"].strip(),
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    }
                )
        result["words"] = words
        return result

    def diarized_transcribe(
        self,
        audio_path: str | Path,
        diarization_segments: Optional[List[Dict]] = None,
        save: bool = True,
        # Diarizer kwargs — only used when diarization_segments is None
        hf_token: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict]:
        """Produce a speaker-aware transcript.

        Merges Whisper word timestamps with pyannote speaker diarization
        segments.  Each returned dict carries the speaker label, text, and
        timestamps.

        Parameters
        ----------
        audio_path:
            Path to the input WAV file.
        diarization_segments:
            Pre-computed diarization output.  If ``None``, diarization is
            run automatically using the pyannote pipeline.
        save:
            If ``True``:
            - writes a human-readable ``.txt`` file to
              ``diarized_transcripts_dir/<stem>.txt``
            - writes a machine-readable ``.json`` file to
              ``diarized_transcripts_dir/<stem>.json``
        hf_token:
            HuggingFace token for pyannote (only used when
            ``diarization_segments`` is ``None``).
        num_speakers / min_speakers / max_speakers:
            Speaker count hints passed to the diarization pipeline.

        Returns
        -------
        list of dict
            Speaker-grouped segments::

                [
                    {"speaker": "SPEAKER_00", "text": "Hello sir.",
                     "start": 0.0, "end": 1.2},
                    {"speaker": "SPEAKER_01", "text": "Hello.",
                     "start": 1.3, "end": 1.9},
                    ...
                ]

        Example
        -------
        >>> t = WhisperTranscriber()
        >>> segments = t.diarized_transcribe("call001.wav", num_speakers=2)
        >>> for s in segments:
        ...     print(f"{s['speaker']}: {s['text']}")
        SPEAKER_00: Hello sir.
        SPEAKER_01: Hello.
        SPEAKER_00: Your account has been blocked.
        """
        from echoguard.modules.speaker_diarization import diarize as run_diarize
        from echoguard.utils.text import format_diarized_transcript

        audio_path = Path(audio_path)

        # Step 1: get diarization segments
        if diarization_segments is None:
            diarization_segments = run_diarize(
                audio_path=audio_path,
                hf_token=hf_token,
                save_json=save,
                save_rttm=save,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        # Step 2: transcribe with word-level timestamps
        whisper_result = self.transcribe_with_timestamps(audio_path)
        words = whisper_result.get("words", [])

        # Step 3: assign words to speakers
        assigned = _assign_words_to_speakers(words, diarization_segments)

        # Step 4: group into speaker turns
        segments = _group_words_by_speaker(assigned)

        # Step 5: persist outputs
        if save:
            formatted = format_diarized_transcript(segments)
            txt_path = self.diarized_transcripts_dir / f"{audio_path.stem}.txt"
            _save_transcript(formatted, txt_path)

            json_path = self.diarized_transcripts_dir / f"{audio_path.stem}.json"
            _save_diarized_json(segments, json_path)

        return segments


# ---------------------------------------------------------------------------
# Module-level convenience functions (mirror WhisperTranscriber methods)
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str | Path,
    model_name: str = "base",
    language: Optional[str] = None,
    save: bool = True,
    transcripts_dir: str | Path = "echoguard/transcripts",
) -> str:
    """Transcribe an audio file to plain text using Whisper.

    Parameters
    ----------
    audio_path:
        Path to the WAV file to transcribe.
    model_name:
        Whisper model size (``tiny | base | small | medium | large``).
    language:
        Language code or ``None`` for auto-detect.
    save:
        Persist transcript to ``transcripts_dir/<stem>.txt``.
    transcripts_dir:
        Output directory for saved transcripts.

    Returns
    -------
    str
        Plain text transcript.

    Example
    -------
    >>> from echoguard.modules.whisper import transcribe
    >>> text = transcribe("echoguard/audio/call001.wav", model_name="small")
    """
    t = WhisperTranscriber(
        model_name=model_name,
        language=language,
        transcripts_dir=transcripts_dir,
    )
    return t.transcribe(audio_path, save=save)


def transcribe_with_timestamps(
    audio_path: str | Path,
    model_name: str = "base",
    language: Optional[str] = None,
) -> Dict:
    """Transcribe an audio file and return word-level timestamps.

    Parameters
    ----------
    audio_path:
        Path to the WAV file.
    model_name:
        Whisper model size.
    language:
        Language code or ``None``.

    Returns
    -------
    dict
        Whisper result dict extended with a ``"words"`` list.

    Example
    -------
    >>> from echoguard.modules.whisper import transcribe_with_timestamps
    >>> result = transcribe_with_timestamps("echoguard/audio/call001.wav")
    >>> for w in result["words"][:3]:
    ...     print(w)
    {'word': 'Hello', 'start': 0.0, 'end': 0.3}
    """
    t = WhisperTranscriber(model_name=model_name, language=language)
    return t.transcribe_with_timestamps(audio_path)


def diarized_transcribe(
    audio_path: str | Path,
    model_name: str = "base",
    language: Optional[str] = None,
    diarization_segments: Optional[List[Dict]] = None,
    save: bool = True,
    diarized_transcripts_dir: str | Path = "echoguard/diarized_transcripts",
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict]:
    """Produce a speaker-aware transcript by fusing Whisper + pyannote.

    This is the main integration point between the Whisper module and the
    speaker diarization module.  The two modules are not directly coupled —
    ``diarize()`` is called through its own public interface.

    Pipeline:

        Audio → pyannote diarization → speaker segments
        Audio → Whisper (word timestamps) → words
        Merge → assign words to speakers → group into turns

    Parameters
    ----------
    audio_path:
        Input WAV file.
    model_name:
        Whisper model size.
    language:
        Language hint for Whisper.
    diarization_segments:
        Pass pre-computed diarization to skip running pyannote again.
    save:
        Write ``.txt`` and ``.json`` to ``diarized_transcripts_dir``.
    diarized_transcripts_dir:
        Output directory for diarized transcripts.
    hf_token:
        HuggingFace access token for pyannote (required on first run).
    num_speakers / min_speakers / max_speakers:
        Speaker count hints for pyannote.

    Returns
    -------
    list of dict
        Speaker-grouped segments with ``speaker``, ``text``, ``start``, ``end``.

    Example
    -------
    >>> from echoguard.modules.whisper import diarized_transcribe
    >>> segments = diarized_transcribe("echoguard/audio/call001.wav",
    ...                                num_speakers=2)
    >>> for s in segments:
    ...     print(f"{s['speaker']}: {s['text']}")
    SPEAKER_00: Hello sir.
    SPEAKER_01: Hello.
    SPEAKER_00: Your account has been blocked.
    SPEAKER_01: Really?
    """
    t = WhisperTranscriber(
        model_name=model_name,
        language=language,
        diarized_transcripts_dir=diarized_transcripts_dir,
    )
    return t.diarized_transcribe(
        audio_path=audio_path,
        diarization_segments=diarization_segments,
        save=save,
        hf_token=hf_token,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
