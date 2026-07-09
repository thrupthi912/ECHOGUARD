"""
echoguard.modules.deepfake
~~~~~~~~~~~~~~~~~~~~~~~~~~
Audio deepfake / spoofing detection using AASIST and related models.

Public interface
----------------
    from echoguard.modules.deepfake import detect_deepfake

    result = detect_deepfake("call001.wav", weights_path="models/AASIST.pth")
    # {
    #   "is_deepfake": False,
    #   "spoof_score": 0.12,
    #   "bonafide_score": 0.88,
    # }
"""

from echoguard.modules.deepfake.detector import detect_deepfake, DeepfakeDetector

__all__ = ["detect_deepfake", "DeepfakeDetector"]
