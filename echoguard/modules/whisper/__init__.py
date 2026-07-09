"""
echoguard.modules.whisper
~~~~~~~~~~~~~~~~~~~~~~~~~
Speech-to-text transcription using OpenAI Whisper.

Public interface
----------------
    from echoguard.modules.whisper import transcribe
    from echoguard.modules.whisper import transcribe_with_timestamps
    from echoguard.modules.whisper import diarized_transcribe

    # Plain transcript
    text = transcribe("call001.wav")

    # Transcript with word-level timestamps
    result = transcribe_with_timestamps("call001.wav")

    # Speaker-aware transcript (requires pyannote)
    segments = diarized_transcribe("call001.wav")
"""

from echoguard.modules.whisper.transcribe import (
    transcribe,
    transcribe_with_timestamps,
    diarized_transcribe,
    WhisperTranscriber,
)

__all__ = [
    "transcribe",
    "transcribe_with_timestamps",
    "diarized_transcribe",
    "WhisperTranscriber",
]
