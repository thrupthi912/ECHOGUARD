"""
echoguard.modules.keyword_detection.detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rule-based scam keyword detection.

Scans a transcript for known scam indicator keywords and phrases, returning
a normalized score and the list of matched terms.

The keyword list lives in ``echoguard/configs/default.yaml`` under
``keyword_detection.keywords``.  Override it at runtime by passing a custom
``keywords`` list to :class:`KeywordDetector`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Default keyword list (mirrors default.yaml for standalone use)
# ---------------------------------------------------------------------------

_DEFAULT_KEYWORDS: List[str] = [
    "bank account",
    "credit card",
    "social security",
    "wire transfer",
    "otp",
    "one-time password",
    "blocked",
    "suspended",
    "arrested",
    "warrant",
    "irs",
    "refund",
    "gift card",
    "bitcoin",
    "urgent",
    "verify your identity",
    "confirm your details",
    "press 1",
    "do not hang up",
]


class KeywordDetector:
    """Detect scam-indicator keywords in a transcript.

    Parameters
    ----------
    keywords:
        List of keywords / phrases to match.  Case-insensitive.
        Defaults to the built-in scam keyword list.
    score_cap:
        Maximum value for the normalized keyword score (default: 1.0).
        Each matched keyword contributes ``1 / score_cap`` to the score,
        so the score saturates at 1.0 after ``score_cap`` matches.

    Example
    -------
    >>> d = KeywordDetector()
    >>> result = d.detect("Your account has been blocked. Please verify.")
    >>> result["matched_keywords"]
    ['blocked', 'verify your identity']
    """

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        score_cap: float = 5.0,
    ) -> None:
        if keywords is None:
            keywords = self._load_keywords_from_config()
        self.keywords: List[str] = [kw.lower() for kw in keywords]
        self.score_cap = max(score_cap, 1.0)

    @staticmethod
    def _load_keywords_from_config() -> List[str]:
        """Try to load keywords from the project config; fall back to default."""
        try:
            from echoguard.utils.config import load_config
            cfg = load_config()
            kws = cfg.get("keyword_detection", {}).get("keywords")
            if kws:
                return kws
        except Exception:
            pass
        return _DEFAULT_KEYWORDS

    def detect(self, text: str) -> Dict:
        """Scan text for scam indicator keywords.

        Parameters
        ----------
        text:
            Transcript or any text string to scan.

        Returns
        -------
        dict
            ::

                {
                    "matched_keywords": ["blocked", "otp"],
                    "keyword_score": 0.4,
                    "is_flagged": True,
                    "match_count": 2,
                }
        """
        if not text:
            return {
                "matched_keywords": [],
                "keyword_score": 0.0,
                "is_flagged": False,
                "match_count": 0,
            }

        text_lower = text.lower()
        matched: List[str] = []

        for keyword in self.keywords:
            # Use word-boundary-aware matching for single words;
            # substring match for multi-word phrases
            if " " in keyword:
                if keyword in text_lower:
                    matched.append(keyword)
            else:
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, text_lower):
                    matched.append(keyword)

        # Remove duplicates while preserving order
        seen = set()
        unique_matched: List[str] = []
        for kw in matched:
            if kw not in seen:
                seen.add(kw)
                unique_matched.append(kw)

        match_count = len(unique_matched)
        keyword_score = round(min(match_count / self.score_cap, 1.0), 4)
        is_flagged = match_count > 0

        return {
            "matched_keywords": unique_matched,
            "keyword_score": keyword_score,
            "is_flagged": is_flagged,
            "match_count": match_count,
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def detect_keywords(
    text: str,
    keywords: Optional[List[str]] = None,
    score_cap: float = 5.0,
) -> Dict:
    """Detect scam indicator keywords in a transcript.

    Parameters
    ----------
    text:
        Transcript string to scan.
    keywords:
        Custom keyword list.  Loads from config if ``None``.
    score_cap:
        Normalization divisor for the keyword score.

    Returns
    -------
    dict
        ``{"matched_keywords": list, "keyword_score": float,
           "is_flagged": bool, "match_count": int}``

    Example
    -------
    >>> from echoguard.modules.keyword_detection import detect_keywords
    >>> result = detect_keywords("Press 1 to verify your identity.")
    >>> result["matched_keywords"]
    ['press 1', 'verify your identity']
    """
    detector = KeywordDetector(keywords=keywords, score_cap=score_cap)
    return detector.detect(text)
