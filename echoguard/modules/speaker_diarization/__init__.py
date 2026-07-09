"""
echoguard.modules.speaker_diarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Speaker diarization using the pretrained pyannote pipeline.

Public interface
----------------
    from echoguard.modules.speaker_diarization import diarize

    segments = diarize("path/to/audio.wav")
    # [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.3}, ...]
"""

from echoguard.modules.speaker_diarization.diarize import diarize, Diarizer

__all__ = ["diarize", "Diarizer"]
