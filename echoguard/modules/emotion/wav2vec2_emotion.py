"""
echoguard.modules.emotion.wav2vec2_emotion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Speech emotion recognition via a fine-tuned wav2vec2 model.

Uses the ``superb/wav2vec2-base-superb-er`` checkpoint from Hugging Face,
which was trained on the SUPERB emotion recognition benchmark.

The original implementation was a raw Google Colab notebook
(``echoguard/emotion_analysis/wav2vec2.py``) that could not be imported
as a Python module.  This file replaces it with a clean, importable class.

Label mapping (RAVDESS-style, 8 classes)
-----------------------------------------
    0 neutral | 1 calm | 2 happy | 3 sad
    4 angry   | 5 fearful | 6 disgust | 7 surprised

Stress score
------------
A composite stress indicator used by the scam-detection pipeline::

    stress = fearful + 0.8 × angry + 0.4 × sad

This is a heuristic — adjust the weights in ``default.yaml`` as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

EMOTION_LABELS: List[str] = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

# Default stress-score weights — can be overridden via config
_DEFAULT_STRESS_WEIGHTS: Dict[str, float] = {
    "fearful": 1.0,
    "angry": 0.8,
    "sad": 0.4,
}

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_model = None
_feature_extractor = None
_loaded_model_id: Optional[str] = None


def _load_model(model_id: str):
    """Load (or return cached) wav2vec2 emotion model and feature extractor.

    Parameters
    ----------
    model_id:
        HuggingFace model repository ID.

    Returns
    -------
    tuple (model, feature_extractor)
    """
    global _model, _feature_extractor, _loaded_model_id

    if _model is not None and _loaded_model_id == model_id:
        return _model, _feature_extractor

    try:
        import torch
        from transformers import (
            AutoModelForAudioClassification,
            Wav2Vec2FeatureExtractor,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for emotion analysis.\n"
            "Run: pip install transformers torch"
        ) from exc

    _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    _model = AutoModelForAudioClassification.from_pretrained(model_id)
    _model.eval()
    _loaded_model_id = model_id
    return _model, _feature_extractor


# ---------------------------------------------------------------------------
# EmotionAnalyzer class
# ---------------------------------------------------------------------------


class EmotionAnalyzer:
    """Analyze speech emotion from an audio file using wav2vec2.

    Parameters
    ----------
    model_id:
        HuggingFace model repository ID.
        Defaults to ``"superb/wav2vec2-base-superb-er"``.
    sample_rate:
        Expected sample rate (Hz).  Audio is resampled to this rate.
    stress_weights:
        Dict mapping emotion label to weight for the stress score.
        Keys must be a subset of ``EMOTION_LABELS``.
    device:
        ``"cpu"`` or ``"cuda"``.  Auto-detected if ``None``.

    Example
    -------
    >>> analyzer = EmotionAnalyzer()
    >>> result = analyzer.analyze("echoguard/audio/call001.wav")
    >>> print(result["predicted_emotion"], result["stress_score"])
    angry 0.74
    """

    def __init__(
        self,
        model_id: str = "superb/wav2vec2-base-superb-er",
        sample_rate: int = 16000,
        stress_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.stress_weights = stress_weights or dict(_DEFAULT_STRESS_WEIGHTS)

        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def _get_model(self):
        return _load_model(self.model_id)

    def _compute_stress_score(self, probs: Dict[str, float]) -> float:
        """Compute composite stress score from emotion probabilities.

        Parameters
        ----------
        probs:
            Mapping of emotion label to probability.

        Returns
        -------
        float
            Stress score in [0, 1] (approximately).
        """
        score = 0.0
        for emotion, weight in self.stress_weights.items():
            score += weight * probs.get(emotion, 0.0)
        return round(float(score), 4)

    def analyze(self, audio_path: str | Path) -> Dict:
        """Run emotion analysis on a WAV file.

        Parameters
        ----------
        audio_path:
            Path to the input audio file.

        Returns
        -------
        dict
            ::

                {
                    "predicted_emotion": "angry",
                    "confidence": 0.82,
                    "stress_score": 0.74,
                    "probabilities": {
                        "neutral": 0.02,
                        "calm": 0.01,
                        "happy": 0.03,
                        "sad": 0.05,
                        "angry": 0.82,
                        "fearful": 0.04,
                        "disgust": 0.02,
                        "surprised": 0.01,
                    },
                }

        Raises
        ------
        FileNotFoundError
            If the audio file does not exist.
        """
        import torch
        import librosa

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model, feature_extractor = self._get_model()
        model = model.to(self.device)

        # Load and resample audio
        audio, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        # Extract features
        inputs = feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        # Inference
        with torch.no_grad():
            logits = model(input_values=input_values).logits

        probs_tensor = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        probs_np: np.ndarray = probs_tensor.cpu().numpy()

        # Build label → probability mapping
        # The SUPERB model may have a different label order — use the model's
        # own id2label if available, otherwise fall back to RAVDESS order.
        id2label: Dict[int, str] = getattr(model.config, "id2label", {})
        if id2label:
            label_list = [id2label[i] for i in range(len(id2label))]
        else:
            label_list = EMOTION_LABELS[: len(probs_np)]

        probabilities = {
            label: round(float(p), 4)
            for label, p in zip(label_list, probs_np)
        }

        predicted_idx = int(np.argmax(probs_np))
        predicted_emotion = label_list[predicted_idx]
        confidence = round(float(probs_np[predicted_idx]), 4)
        stress_score = self._compute_stress_score(probabilities)

        return {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "stress_score": stress_score,
            "probabilities": probabilities,
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def analyze_emotion(
    audio_path: str | Path,
    model_id: str = "superb/wav2vec2-base-superb-er",
    sample_rate: int = 16000,
    stress_weights: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
) -> Dict:
    """Analyze the emotion in a speech audio file.

    This is the primary public interface for this module.

    Parameters
    ----------
    audio_path:
        Path to the input WAV file.
    model_id:
        HuggingFace model ID for the wav2vec2 emotion classifier.
    sample_rate:
        Target sample rate for audio loading.
    stress_weights:
        Custom weights for stress score computation.
        Defaults to ``{"fearful": 1.0, "angry": 0.8, "sad": 0.4}``.
    device:
        ``"cpu"`` or ``"cuda"``.  Auto-detected if ``None``.

    Returns
    -------
    dict
        ``{"predicted_emotion": str, "confidence": float,
           "stress_score": float, "probabilities": dict}``

    Example
    -------
    >>> from echoguard.modules.emotion import analyze_emotion
    >>> result = analyze_emotion("echoguard/audio/call001.wav")
    >>> print(result["predicted_emotion"])
    angry
    """
    analyzer = EmotionAnalyzer(
        model_id=model_id,
        sample_rate=sample_rate,
        stress_weights=stress_weights,
        device=device,
    )
    return analyzer.analyze(audio_path)
