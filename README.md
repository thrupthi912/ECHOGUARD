# EchoGuard

**Multimodal Phone Scam Detection**

EchoGuard is a modular AI system that analyses phone call audio to detect potential scams. It combines speech-to-text, speaker diarization, emotion recognition, deepfake detection, keyword analysis, and semantic context retrieval into a unified, end-to-end pipeline.

---

## Architecture

```
Audio
  │
  ├─► Speaker Diarization (pyannote.audio)
  │         └─► Whisper (word timestamps)
  │                   └─► Diarized Transcript ──► Keyword Detection
  │
  ├─► Emotion Analysis (wav2vec2)
  │
  ├─► Deepfake Detection (AASIST)
  │
  ├─► Context Retrieval (FAISS)
  │
  └─► Score Fusion ──► Verdict: SCAM / SUSPICIOUS / BENIGN
```

---

## Repository Structure

```
ECHOGUARD/
├── echoguard/
│   ├── __init__.py
│   ├── modules/
│   │   ├── whisper/             # Whisper STT + diarized transcription
│   │   ├── speaker_diarization/ # pyannote speaker diarization
│   │   ├── emotion/             # wav2vec2 emotion recognition
│   │   ├── deepfake/            # AASIST deepfake detection
│   │   ├── keyword_detection/   # Rule-based scam keyword matching
│   │   ├── context/             # FAISS semantic context retrieval
│   │   └── fusion/              # Weighted score fusion
│   ├── utils/
│   │   ├── audio.py             # Shared audio I/O helpers
│   │   ├── text.py              # Transcript formatting helpers
│   │   └── config.py            # YAML config loader
│   ├── configs/
│   │   ├── default.yaml         # Main configuration file
│   │   └── aasist.yaml          # AASIST model architecture configs
│   ├── datasets/
│   │   └── metadata.csv         # Call metadata (filename, label, category)
│   ├── audio/                   # Input audio files (not tracked in git)
│   ├── transcripts/             # Plain Whisper transcripts
│   ├── diarized_transcripts/    # Speaker-labelled transcripts
│   └── outputs/                 # Inference results (JSON, RTTM)
├── app/                         # API / web interface (future)
├── main.py                      # End-to-end pipeline CLI
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU**: For CUDA support, install PyTorch separately following the
> [official guide](https://pytorch.org/get-started/locally/).

### 2. Configure HuggingFace access

Speaker diarization requires accepting the pyannote terms of use:
1. Visit [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and accept the terms.
2. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Set it as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

### 3. Run the pipeline

```bash
# Full pipeline (with speaker diarization)
python main.py --audio echoguard/audio/call001.wav

# Specify whisper model and number of speakers
python main.py --audio call001.wav --model small --speakers 2

# Skip modules that need external weights/tokens
python main.py --audio call001.wav --skip-deepfake --skip-diarization
```

---

## Module Reference

### Speaker Diarization

```python
from echoguard.modules.speaker_diarization import diarize

segments = diarize("echoguard/audio/call001.wav", num_speakers=2)
# [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.3}, ...]
```

Outputs are saved to `echoguard/outputs/`:
- `<stem>.diarization.json` — structured segments
- `<stem>.rttm` — RTTM format for evaluation tooling

### Whisper Transcription

```python
from echoguard.modules.whisper import transcribe, diarized_transcribe

# Plain transcript
text = transcribe("call001.wav", model_name="small")

# Speaker-aware transcript (runs diarization automatically)
segments = diarized_transcribe("call001.wav", num_speakers=2)
# [{"speaker": "SPEAKER_00", "text": "Hello sir.", "start": 0.0, ...}, ...]
```

The diarized transcript is formatted and saved to `echoguard/diarized_transcripts/`:
```
Speaker 1: Hello sir.
Speaker 2: Hello.
Speaker 1: Your account has been blocked.
Speaker 2: Really?
```

### Emotion Analysis

```python
from echoguard.modules.emotion import analyze_emotion

result = analyze_emotion("call001.wav")
# {"predicted_emotion": "angry", "confidence": 0.82, "stress_score": 0.74, ...}
```

### Deepfake Detection (AASIST)

Requires a pre-trained `.pth` weight file.

```python
from echoguard.modules.deepfake import detect_deepfake

result = detect_deepfake(
    "call001.wav",
    weights_path="echoguard/models/deepfake/AASIST.pth"
)
# {"is_deepfake": False, "spoof_score": 0.12, "bonafide_score": 0.88}
```

Download pre-trained weights from: https://github.com/clovaai/aasist

### Keyword Detection

```python
from echoguard.modules.keyword_detection import detect_keywords

result = detect_keywords("Your account has been blocked. Press 1 to verify.")
# {"matched_keywords": ["blocked", "press 1", "verify your identity"],
#  "keyword_score": 0.6, "is_flagged": True, "match_count": 3}
```

### Score Fusion

```python
from echoguard.modules.fusion import fuse_scores

result = fuse_scores({
    "keyword":  {"keyword_score": 0.6},
    "emotion":  {"stress_score": 0.74},
    "deepfake": {"spoof_score": 0.12},
})
# {"scam_probability": 0.49, "verdict": "SUSPICIOUS", "confidence": "MEDIUM", ...}
```

---

## Configuration

All settings live in `echoguard/configs/default.yaml`. Key sections:

| Section | Description |
|---|---|
| `whisper` | Model size, language |
| `speaker_diarization` | pyannote model ID, HF token, min segment duration |
| `emotion` | wav2vec2 model ID, stress score weights |
| `deepfake` | Weights path, architecture, threshold |
| `keyword_detection` | Keyword list |
| `context` | Embedding model, top-k |
| `fusion` | Per-module score weights |

---

## Attribution

### AASIST (Audio Anti-Spoofing)
The AASIST model architecture (`echoguard/modules/deepfake/aasist_model.py`) is
adapted from the [clovaai/aasist](https://github.com/clovaai/aasist) repository.

Copyright (c) 2021-present NAVER Corp.  
Licensed under the MIT License.

> Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
> Graph Attention Networks", arXiv:2110.01200, 2021.

### pyannote.audio
Speaker diarization uses the [pyannote.audio](https://github.com/pyannote/pyannote-audio)
library. If you use it in research, please cite:

> Bredin, H. et al., "pyannote.audio: neural building blocks for speaker
> diarization", ICASSP 2020.

---

## Future Improvements

- [ ] Add TF-IDF + SVM / Random Forest transcript classifiers
- [ ] Add DistilBERT / RoBERTa text classifiers
- [ ] Add MFCC + Random Forest audio feature classifier
- [ ] Build FAISS context index from labelled call data
- [ ] Add FastAPI web service in `app/`
- [ ] Add training scripts for fine-tuning emotion and deepfake models on call data
- [ ] Add evaluation scripts and benchmark results
- [ ] Docker / containerization support
