"""
echoguard.modules.emotion
~~~~~~~~~~~~~~~~~~~~~~~~~
Speech emotion recognition using a fine-tuned wav2vec2 model.

Public interface
----------------
    from echoguard.modules.emotion import analyze_emotion

    result = analyze_emotion("call001.wav")
    # {
    #   "predicted_emotion": "angry",
    #   "confidence": 0.82,
    #   "stress_score": 0.74,
    #   "probabilities": {"neutral": 0.03, "angry": 0.82, ...}
    # }
"""

from echoguard.modules.emotion.wav2vec2_emotion import analyze_emotion, EmotionAnalyzer

__all__ = ["analyze_emotion", "EmotionAnalyzer"]
