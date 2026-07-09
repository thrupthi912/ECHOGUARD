"""
echoguard.modules.keyword_detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rule-based keyword detection for scam indicator phrases.

Public interface
----------------
    from echoguard.modules.keyword_detection import detect_keywords

    result = detect_keywords("Your account has been blocked. Please verify.")
    # {
    #   "matched_keywords": ["account", "blocked", "verify"],
    #   "keyword_score": 0.6,
    #   "is_flagged": True,
    # }
"""

from echoguard.modules.keyword_detection.detector import detect_keywords, KeywordDetector

__all__ = ["detect_keywords", "KeywordDetector"]
