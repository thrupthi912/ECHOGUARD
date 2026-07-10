"""
echoguard.modules.noise_cancellation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Audio noise cancellation / enhancement stage for the EchoGuard pipeline.

This module runs *before* any other analysis module.  Cleaning the audio
first improves accuracy across all downstream models:

  - Whisper transcription is more accurate on denoised speech
  - wav2vec2 emotion detection is more sensitive to clean signals
  - AASIST deepfake detection relies on fine-grained spectral features
    that noise can obscure

Two strategies are available (selected automatically):

  ``speechbrain``  — neural spectral-mask enhancement via
                     ``speechbrain/mtl-mimic-voicebank`` (best quality)
  ``dsp``          — classical noisereduce + band-pass filter + normalise
                     (no GPU / large model required, good fallback)

Public interface
----------------
    from echoguard.modules.noise_cancellation import enhance_audio

    enhanced_path = enhance_audio("echoguard/audio/call001.wav")
    # Returns: "echoguard/audio/call001_enhanced.wav"
"""

from echoguard.modules.noise_cancellation.enhancer import enhance_audio, AudioEnhancer

__all__ = ["enhance_audio", "AudioEnhancer"]
