"""
echoguard.utils.text
~~~~~~~~~~~~~~~~~~~~
Shared text-processing helpers used across transcript and NLP modules.
"""

from __future__ import annotations

import re
from typing import List, Dict


def sanitize_text(text: str) -> str:
    """Normalize whitespace and strip leading/trailing spaces.

    Parameters
    ----------
    text:
        Raw transcript string.

    Returns
    -------
    str
        Cleaned string with collapsed internal whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()


def format_diarized_transcript(
    segments: List[Dict],
) -> str:
    """Convert a list of diarized segments into a readable transcript string.

    Each segment is expected to have the keys ``speaker`` and ``text``.
    Consecutive segments from the same speaker are merged into one block.

    Parameters
    ----------
    segments:
        List of dicts with at minimum::

            {
                "speaker": "SPEAKER_00",
                "text": "Hello sir.",
                "start": 0.0,   # optional
                "end": 1.2,     # optional
            }

    Returns
    -------
    str
        Human-readable, speaker-labelled transcript, e.g.::

            Speaker 1: Hello sir.
            Speaker 2: Hello.
            Speaker 1: Your account has been blocked.

    Example
    -------
    >>> segs = [
    ...     {"speaker": "SPEAKER_00", "text": "Hello sir."},
    ...     {"speaker": "SPEAKER_01", "text": "Hello."},
    ... ]
    >>> print(format_diarized_transcript(segs))
    Speaker 1: Hello sir.
    Speaker 2: Hello.
    """
    if not segments:
        return ""

    # Build a stable speaker-number mapping in order of first appearance
    speaker_map: Dict[str, int] = {}
    counter = 1
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        if spk not in speaker_map:
            speaker_map[spk] = counter
            counter += 1

    lines: List[str] = []
    current_speaker: str | None = None
    buffer: List[str] = []

    def _flush(spk: str | None, buf: List[str]) -> None:
        if spk is not None and buf:
            label = f"Speaker {speaker_map.get(spk, spk)}"
            merged = sanitize_text(" ".join(buf))
            if merged:
                lines.append(f"{label}: {merged}")

    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if spk != current_speaker:
            _flush(current_speaker, buffer)
            buffer = [text]
            current_speaker = spk
        else:
            buffer.append(text)

    _flush(current_speaker, buffer)
    return "\n".join(lines)


def speaker_label(speaker_id: str, speaker_map: Dict[str, int]) -> str:
    """Return a human-readable label for a pyannote speaker ID.

    Parameters
    ----------
    speaker_id:
        Raw pyannote speaker identifier, e.g. ``"SPEAKER_00"``.
    speaker_map:
        Mapping from speaker ID to integer index (1-based).

    Returns
    -------
    str
        e.g. ``"Speaker 1"``
    """
    return f"Speaker {speaker_map.get(speaker_id, speaker_id)}"
