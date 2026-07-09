"""
EchoGuard — Multimodal Phone Scam Detection
============================================
Version: 0.1.0

Quick start
-----------
    from echoguard.modules.whisper import transcribe, diarized_transcribe
    from echoguard.modules.speaker_diarization import diarize
    from echoguard.modules.emotion import analyze_emotion
    from echoguard.modules.deepfake import detect_deepfake
    from echoguard.modules.keyword_detection import detect_keywords
    from echoguard.modules.fusion import fuse_scores

Or run the full end-to-end pipeline:
    python main.py --audio echoguard/audio/call001.wav
"""

__version__ = "0.1.0"
__author__ = "EchoGuard Team"
