"""
echoguard.modules.fusion
~~~~~~~~~~~~~~~~~~~~~~~~
Score fusion module — combines outputs from all analysis modules into a
single scam probability score.

Public interface
----------------
    from echoguard.modules.fusion import fuse_scores

    result = fuse_scores({
        "keyword":  {"keyword_score": 0.6, "is_flagged": True},
        "emotion":  {"stress_score": 0.74},
        "deepfake": {"spoof_score": 0.12},
        "context":  [{"score": 0.91, "label": "scam"}, ...],
    })
    # {"scam_probability": 0.63, "verdict": "SCAM", "confidence": "HIGH"}
"""

from echoguard.modules.fusion.fusion import fuse_scores, ScoreFusion

__all__ = ["fuse_scores", "ScoreFusion"]
