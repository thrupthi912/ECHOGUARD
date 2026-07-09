"""
echoguard.modules
~~~~~~~~~~~~~~~~~
All EchoGuard analysis modules.

Each sub-module exposes a single clean public function so that the
pipeline (main.py / fusion) can call them without knowing implementation
details.

  Module                Public function
  ──────────────────── ──────────────────────────────────────────
  whisper              transcribe(audio_path) -> str
                       transcribe_with_timestamps(audio_path) -> dict
  speaker_diarization  diarize(audio_path) -> list[dict]
  whisper (combined)   diarized_transcribe(audio_path) -> list[dict]
  emotion              analyze_emotion(audio_path) -> dict
  deepfake             detect_deepfake(audio_path) -> dict
  keyword_detection    detect_keywords(text) -> dict
  context              retrieve_context(query) -> list[dict]
  fusion               fuse_scores(results) -> dict
"""
