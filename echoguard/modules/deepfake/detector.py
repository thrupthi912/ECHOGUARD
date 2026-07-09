"""
echoguard.modules.deepfake.detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clean inference interface for AASIST-based audio deepfake detection.

This module wraps the AASIST model architecture for single-file inference,
deliberately separating inference from the original training script
(``aasist/main.py``) which required the full ASVspoof2019 dataset.

Supported architectures
-----------------------
* AASIST
* AASIST-L
* RawNet2Spoof
* RawNetGatSpoofST

Model weights
-------------
Pre-trained weights (.pth) are NOT bundled in this repository due to
file size.  Set the weight path in ``echoguard/configs/default.yaml``
under ``deepfake.model_weights`` or pass ``weights_path`` directly.

To obtain weights:
  - AASIST & AASIST-L: https://github.com/clovaai/aasist (NAVER Corp.)
  - RawNet2: https://github.com/asvspoof-challenge/2021

Output format
-------------
    {
        "is_deepfake": bool,     # True when spoof_score > threshold
        "spoof_score": float,    # P(spoof)  in [0, 1]
        "bonafide_score": float, # P(bonafide) in [0, 1]
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Architecture name → module attribute mapping
_ARCH_MAP = {
    "AASIST": ("echoguard.modules.deepfake.aasist_model", "Model"),
    "AASIST-L": ("echoguard.modules.deepfake.aasist_model", "Model"),
    "RawNet2Spoof": ("echoguard.modules.deepfake.rawnet2_model", "Model"),
    "RawNetGatSpoofST": ("echoguard.modules.deepfake.rawgatst_model", "Model"),
}

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_model_cache: Dict[str, object] = {}


def _load_model(
    architecture: str,
    weights_path: str | Path,
    model_config: Dict,
    device: str,
):
    """Load (or return cached) deepfake detection model.

    Parameters
    ----------
    architecture:
        Model architecture name (key in ``_ARCH_MAP``).
    weights_path:
        Path to the ``.pth`` weight file.
    model_config:
        Architecture hyper-parameters (matches the ``model_config``
        section of the original AASIST `.conf` files).
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on the specified device.
    """
    import torch
    from importlib import import_module

    cache_key = f"{architecture}:{weights_path}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if architecture not in _ARCH_MAP:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(_ARCH_MAP.keys())}"
        )

    module_path, class_name = _ARCH_MAP[architecture]
    module = import_module(module_path)
    ModelClass = getattr(module, class_name)

    model = ModelClass(model_config).to(device)

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Download pre-trained weights from:\n"
            "  AASIST: https://github.com/clovaai/aasist"
        )

    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache[cache_key] = model
    return model


def _load_and_pad_audio(
    audio_path: Path,
    max_samples: int = 64600,
    sample_rate: int = 16000,
) -> "torch.Tensor":
    """Load a WAV file, resample, and pad/trim to ``max_samples``.

    Parameters
    ----------
    audio_path:
        Path to the input audio file.
    max_samples:
        Fixed length in samples (default: 64600 ≈ 4 s at 16 kHz).
    sample_rate:
        Target sample rate.

    Returns
    -------
    torch.Tensor
        Shape ``(1, max_samples)`` float32 tensor.
    """
    import torch
    import soundfile as sf
    import numpy as np

    x, sr = sf.read(str(audio_path))
    if x.ndim > 1:
        x = x[:, 0]  # take first channel

    # Resample if necessary
    if sr != sample_rate:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=sample_rate)

    x = x.astype(np.float32)

    # Pad or trim to max_samples
    x_len = len(x)
    if x_len >= max_samples:
        x = x[:max_samples]
    else:
        repeats = int(max_samples / x_len) + 1
        x = np.tile(x, repeats)[:max_samples]

    return torch.FloatTensor(x).unsqueeze(0)  # (1, max_samples)


class DeepfakeDetector:
    """Inference wrapper for AASIST-family audio deepfake detection models.

    Parameters
    ----------
    architecture:
        Model architecture name: ``AASIST | AASIST-L | RawNet2Spoof |
        RawNetGatSpoofST``.
    weights_path:
        Path to the pre-trained ``.pth`` weight file.
    model_config:
        Architecture hyper-parameters.  If ``None``, loaded from
        ``echoguard/configs/aasist.yaml``.
    threshold:
        Spoof probability above which a sample is flagged as deepfake.
    device:
        ``"cpu"`` or ``"cuda"``.  Auto-detected if ``None``.

    Example
    -------
    >>> detector = DeepfakeDetector(
    ...     architecture="AASIST",
    ...     weights_path="echoguard/models/deepfake/AASIST.pth",
    ... )
    >>> result = detector.detect("echoguard/audio/call001.wav")
    >>> print(result["is_deepfake"], result["spoof_score"])
    False 0.12
    """

    def __init__(
        self,
        architecture: str = "AASIST",
        weights_path: Optional[str | Path] = None,
        model_config: Optional[Dict] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.architecture = architecture
        self.weights_path = Path(weights_path) if weights_path else None
        self.threshold = threshold

        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Load model config from YAML if not provided
        if model_config is None:
            from echoguard.utils.config import load_config
            cfg_path = (
                Path(__file__).resolve().parents[2] / "configs" / "aasist.yaml"
            )
            all_configs = load_config(cfg_path)
            model_config = all_configs.get(architecture, {})

        self.model_config = model_config

    def detect(self, audio_path: str | Path) -> Dict:
        """Run deepfake detection on a single audio file.

        Parameters
        ----------
        audio_path:
            Path to the WAV file.

        Returns
        -------
        dict
            ::

                {
                    "is_deepfake": bool,
                    "spoof_score": float,   # P(spoof)
                    "bonafide_score": float, # P(bonafide)
                }

        Raises
        ------
        ValueError
            If no weights path is configured.
        FileNotFoundError
            If the audio file or weight file does not exist.
        """
        import torch

        if self.weights_path is None:
            raise ValueError(
                "No model weights configured for deepfake detection.\n"
                "Set 'deepfake.model_weights' in default.yaml or pass "
                "'weights_path' to DeepfakeDetector."
            )

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = _load_model(
            self.architecture,
            self.weights_path,
            self.model_config,
            self.device,
        )

        max_samples = self.model_config.get("nb_samp", 64600)
        x = _load_and_pad_audio(audio_path, max_samples=max_samples)
        x = x.to(self.device)

        with torch.no_grad():
            _, logits = model(x)  # AASIST returns (embedding, logits)
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()

        probs_np = probs.cpu().numpy()
        # Convention: index 0 = bonafide, index 1 = spoof
        bonafide_score = float(probs_np[0]) if len(probs_np) > 1 else 1.0 - float(probs_np[0])
        spoof_score = float(probs_np[1]) if len(probs_np) > 1 else float(probs_np[0])

        return {
            "is_deepfake": spoof_score >= self.threshold,
            "spoof_score": round(spoof_score, 4),
            "bonafide_score": round(bonafide_score, 4),
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def detect_deepfake(
    audio_path: str | Path,
    weights_path: Optional[str | Path] = None,
    architecture: str = "AASIST",
    model_config: Optional[Dict] = None,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Dict:
    """Detect whether an audio file is a deepfake / synthetic voice.

    This is the primary public interface for this module.

    Parameters
    ----------
    audio_path:
        Path to the WAV file to analyse.
    weights_path:
        Path to the pre-trained model weight file (``.pth``).
        Falls back to ``deepfake.model_weights`` in ``default.yaml``.
    architecture:
        Model architecture to use.
    model_config:
        Architecture hyper-parameters (loaded from YAML if ``None``).
    threshold:
        Spoof probability decision threshold.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict
        ``{"is_deepfake": bool, "spoof_score": float,
           "bonafide_score": float}``

    Example
    -------
    >>> from echoguard.modules.deepfake import detect_deepfake
    >>> result = detect_deepfake(
    ...     "echoguard/audio/call001.wav",
    ...     weights_path="echoguard/models/deepfake/AASIST.pth",
    ... )
    >>> print(result["spoof_score"])
    0.12
    """
    # Fall back to config if no weights path given
    if weights_path is None:
        try:
            from echoguard.utils.config import load_config
            cfg = load_config()
            weights_path = cfg.get("deepfake", {}).get("model_weights")
        except Exception:
            pass

    detector = DeepfakeDetector(
        architecture=architecture,
        weights_path=weights_path,
        model_config=model_config,
        threshold=threshold,
        device=device,
    )
    return detector.detect(audio_path)
