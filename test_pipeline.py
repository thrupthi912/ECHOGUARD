"""
test_pipeline.py — EchoGuard end-to-end verification script
============================================================
Run from the project root:
    python test_pipeline.py
    python test_pipeline.py --audio path/to/file.wav

This script:
  1. Verifies every module can be imported
  2. Checks all required packages are installed
  3. Runs each module on a real audio file (synthetic if none provided)
  4. Prints a clear status summary
  5. Never crashes — missing weights / tokens are reported, not raised
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import wave
from pathlib import Path
from typing import Dict, List, Tuple

# ── Make sure the project root is on sys.path ─────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── ANSI colours (disabled automatically on Windows / non-TTY) ───────────────
_USE_COLOUR = sys.stdout.isatty() and sys.platform != "win32"
GREEN  = "\033[92m" if _USE_COLOUR else ""
YELLOW = "\033[93m" if _USE_COLOUR else ""
RED    = "\033[91m" if _USE_COLOUR else ""
BOLD   = "\033[1m"  if _USE_COLOUR else ""
RESET  = "\033[0m"  if _USE_COLOUR else ""

TICK  = f"{GREEN}✓{RESET}"
WARN  = f"{YELLOW}⚠{RESET}"
CROSS = f"{RED}✗{RESET}"
SEP   = "-" * 57

# ─────────────────────────────────────────────────────────────────────────────
# Result tracker
# ─────────────────────────────────────────────────────────────────────────────

class Result:
    OK      = "ok"
    WARN    = "warn"     # works but missing optional asset (weights/token)
    SKIP    = "skip"     # skipped because a hard dep is missing
    FAIL    = "fail"     # unexpected error

    def __init__(self):
        self._items: List[Tuple[str, str, str]] = []  # (label, status, note)

    def add(self, label: str, status: str, note: str = ""):
        self._items.append((label, status, note))

    def print_summary(self):
        print(f"\n{BOLD}{SEP}{RESET}")
        print(f"{BOLD}  Pipeline Status{RESET}")
        print(SEP)
        for label, status, note in self._items:
            if status == Result.OK:
                icon = TICK
            elif status == Result.WARN:
                icon = WARN
            elif status == Result.SKIP:
                icon = f"{YELLOW}↷{RESET}"
            else:
                icon = CROSS
            suffix = f"  ({note})" if note else ""
            print(f"  {icon}  {label}{suffix}")
        print(SEP)
        n_ok   = sum(1 for _, s, _ in self._items if s == Result.OK)
        n_warn = sum(1 for _, s, _ in self._items if s == Result.WARN)
        n_skip = sum(1 for _, s, _ in self._items if s == Result.SKIP)
        n_fail = sum(1 for _, s, _ in self._items if s == Result.FAIL)
        print(f"  {TICK} {n_ok} passed  "
              f"{WARN} {n_warn} warnings  "
              f"{YELLOW}↷{RESET} {n_skip} skipped  "
              f"{CROSS} {n_fail} failed")
        print(f"{BOLD}{SEP}{RESET}\n")

R = Result()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str):
    print(f"\n{BOLD}{SEP}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{SEP}{RESET}")


def _step(msg: str):
    print(f"  {BOLD}→{RESET} {msg}", end="", flush=True)


def _ok(detail: str = ""):
    suffix = f"  {detail}" if detail else ""
    print(f"  {TICK}{suffix}")


def _warn(detail: str):
    print(f"  {WARN}  {YELLOW}{detail}{RESET}")


def _fail(detail: str):
    print(f"  {CROSS}  {RED}{detail}{RESET}")


def _info(detail: str):
    print(f"       {detail}")


def _pkg_installed(pkg_import: str) -> bool:
    try:
        __import__(pkg_import)
        return True
    except ImportError:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic audio generator — used when no real WAV is provided
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_wav(path: Path, duration_s: float = 3.0, sr: int = 16000) -> Path:
    """Write a short sine-wave WAV so modules that need a real file can run."""
    import math, array
    n_samples = int(sr * duration_s)
    freq = 440.0  # A4
    samples = array.array(
        "h",
        [int(32767 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n_samples)],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Package availability
# ─────────────────────────────────────────────────────────────────────────────

def check_packages():
    _header("1 / 6  Package Availability")

    required = [
        ("whisper",               "openai-whisper",          True),
        ("torch",                 "torch",                   True),
        ("transformers",          "transformers",            True),
        ("yaml",                  "PyYAML",                  True),
        ("numpy",                 "numpy",                   True),
        ("sklearn",               "scikit-learn",            True),
        ("tqdm",                  "tqdm",                    True),
        ("pandas",                "pandas",                  True),
        ("librosa",               "librosa",                 False),
        ("soundfile",             "soundfile",               False),
        ("torchaudio",            "torchaudio",              False),
        ("pyannote.audio",        "pyannote.audio",          False),
        ("faiss",                 "faiss-cpu",               False),
        ("sentence_transformers", "sentence-transformers",   False),
    ]

    all_hard_ok = True
    for imp, pip_name, is_hard in required:
        _step(f"{pip_name:<30}")
        if _pkg_installed(imp):
            mod = sys.modules.get(imp) or __import__(imp)
            ver = getattr(mod, "__version__", "?")
            _ok(ver)
        else:
            if is_hard:
                _fail(f"MISSING (required) — pip install {pip_name}")
                all_hard_ok = False
            else:
                _warn(f"not installed — pip install {pip_name}")

    if not all_hard_ok:
        _info("  Some required packages are missing. Install them before running.")
    R.add("Package availability",
          Result.OK if all_hard_ok else Result.FAIL,
          "" if all_hard_ok else "hard deps missing")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Module imports
# ─────────────────────────────────────────────────────────────────────────────

def check_imports():
    _header("2 / 6  Module Imports")

    modules = [
        ("echoguard.utils.config",                        "Config loader"),
        ("echoguard.utils.audio",                         "Audio utils"),
        ("echoguard.utils.text",                          "Text utils"),
        ("echoguard.modules.keyword_detection.detector",  "Keyword detection"),
        ("echoguard.modules.fusion.fusion",               "Fusion"),
        ("echoguard.modules.deepfake.aasist_model",       "AASIST model architecture"),
        ("echoguard.modules.deepfake.aasist_utils",       "AASIST utils"),
        ("echoguard.modules.deepfake.rawnet2_model",      "RawNet2 model architecture"),
        ("echoguard.modules.deepfake.rawgatst_model",     "RawGAT-ST model architecture"),
        ("echoguard.modules.deepfake.detector",           "Deepfake detector"),
        ("echoguard.modules.whisper.transcribe",          "Whisper transcriber"),
        ("echoguard.modules.speaker_diarization.diarize", "Speaker diarization"),
        ("echoguard.modules.emotion.wav2vec2_emotion",    "Emotion analyzer"),
        ("echoguard.modules.context.retrieval",           "Context retrieval"),
    ]

    all_ok = True
    for mod_path, label in modules:
        _step(f"{label:<35}")
        try:
            __import__(mod_path)
            _ok()
        except ImportError as e:
            _fail(str(e))
            all_ok = False
        except Exception as e:
            _fail(f"Unexpected: {e}")
            all_ok = False

    R.add("Module imports", Result.OK if all_ok else Result.FAIL,
          "" if all_ok else "see above")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Repository structure
# ─────────────────────────────────────────────────────────────────────────────

def check_structure():
    _header("3 / 6  Repository Structure")

    expected = [
        "echoguard/__init__.py",
        "echoguard/configs/default.yaml",
        "echoguard/configs/aasist.yaml",
        "echoguard/datasets/metadata.csv",
        "echoguard/modules/__init__.py",
        "echoguard/modules/whisper/__init__.py",
        "echoguard/modules/whisper/transcribe.py",
        "echoguard/modules/speaker_diarization/__init__.py",
        "echoguard/modules/speaker_diarization/diarize.py",
        "echoguard/modules/emotion/__init__.py",
        "echoguard/modules/emotion/wav2vec2_emotion.py",
        "echoguard/modules/deepfake/__init__.py",
        "echoguard/modules/deepfake/aasist_model.py",
        "echoguard/modules/deepfake/aasist_utils.py",
        "echoguard/modules/deepfake/rawnet2_model.py",
        "echoguard/modules/deepfake/rawgatst_model.py",
        "echoguard/modules/deepfake/detector.py",
        "echoguard/modules/keyword_detection/__init__.py",
        "echoguard/modules/keyword_detection/detector.py",
        "echoguard/modules/context/__init__.py",
        "echoguard/modules/context/retrieval.py",
        "echoguard/modules/fusion/__init__.py",
        "echoguard/modules/fusion/fusion.py",
        "echoguard/utils/__init__.py",
        "echoguard/utils/audio.py",
        "echoguard/utils/text.py",
        "echoguard/utils/config.py",
        "main.py",
        "requirements.txt",
        "README.md",
    ]

    missing = []
    for rel in expected:
        p = ROOT / rel
        _step(f"{rel:<55}")
        if p.exists():
            _ok()
        else:
            _fail("MISSING")
            missing.append(rel)

    R.add("Repository structure",
          Result.OK if not missing else Result.FAIL,
          f"{len(missing)} files missing" if missing else "")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Config loading
# ─────────────────────────────────────────────────────────────────────────────

def check_config():
    _header("4 / 6  Configuration Files")

    _step("Loading default.yaml          ")
    try:
        from echoguard.utils.config import load_config
        cfg = load_config()
        required_keys = ["whisper", "speaker_diarization", "emotion",
                         "deepfake", "keyword_detection", "fusion"]
        missing_keys = [k for k in required_keys if k not in cfg]
        if missing_keys:
            _fail(f"Missing keys: {missing_keys}")
            R.add("default.yaml", Result.FAIL, f"missing keys: {missing_keys}")
        else:
            _ok(f"{len(cfg)} top-level sections")
            R.add("default.yaml", Result.OK)
    except Exception as e:
        _fail(str(e))
        R.add("default.yaml", Result.FAIL, str(e))

    _step("Loading aasist.yaml           ")
    try:
        from echoguard.utils.config import load_config
        acfg = load_config(ROOT / "echoguard/configs/aasist.yaml")
        archs = list(acfg.keys())
        _ok(f"architectures: {archs}")
        R.add("aasist.yaml", Result.OK)
    except Exception as e:
        _fail(str(e))
        R.add("aasist.yaml", Result.FAIL, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Per-module runtime tests (with audio)
# ─────────────────────────────────────────────────────────────────────────────

def run_module_tests(audio_path: Path):
    _header("5 / 6  Module Runtime Tests")
    print(f"  Audio file: {audio_path}")

    # ── Whisper ───────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Loading Whisper...{RESET}")
    if not _pkg_installed("whisper"):
        _warn("openai-whisper not installed — skipping")
        _info("Fix: pip install openai-whisper")
        R.add("Whisper", Result.SKIP, "openai-whisper not installed")
    else:
        try:
            from echoguard.modules.whisper.transcribe import WhisperTranscriber
            _step("Initializing WhisperTranscriber (base model)  ")
            t = WhisperTranscriber(model_name="base",
                                   transcripts_dir=str(ROOT / "echoguard/transcripts"))
            _ok("initialized")
            _step("Running Speech-to-Text                        ")
            transcript = t.transcribe(audio_path, save=True)
            _ok(f"{len(transcript)} chars — \"{transcript[:60].strip()}\"")
            R.add("Whisper", Result.OK)
        except Exception as e:
            _fail(str(e))
            R.add("Whisper", Result.FAIL, str(e)[:80])
        else:
            print(f"  {TICK} Whisper loaded")

    # ── Speaker Diarization ───────────────────────────────────────────────────
    print(f"\n  {BOLD}Loading Speaker Diarization...{RESET}")
    hf_token = os.environ.get("HF_TOKEN")
    if not _pkg_installed("pyannote.audio"):
        _warn("pyannote.audio not installed — skipping")
        _info("Fix: pip install pyannote.audio")
        R.add("Speaker Diarization", Result.SKIP, "pyannote.audio not installed")
    elif not hf_token:
        _warn("HF_TOKEN environment variable not set")
        _info("Fix: export HF_TOKEN=your_token_here")
        _info("Then: accept terms at https://hf.co/pyannote/speaker-diarization-3.1")
        R.add("Speaker Diarization", Result.WARN, "HF_TOKEN missing")
        print(f"  {WARN} Speaker Diarization (HF_TOKEN not set)")
    else:
        try:
            from echoguard.modules.speaker_diarization.diarize import Diarizer
            _step("Initializing Diarizer                         ")
            d = Diarizer(hf_token=hf_token,
                         output_dir=str(ROOT / "echoguard/outputs"))
            _ok("initialized")
            _step("Running Speaker Diarization                   ")
            segs = d.diarize(audio_path, save_json=True, save_rttm=True)
            speakers = list({s["speaker"] for s in segs})
            _ok(f"{len(segs)} segments, speakers: {speakers}")
            R.add("Speaker Diarization", Result.OK)
            print(f"  {TICK} Diarization loaded")
        except ValueError as e:
            _warn(str(e))
            R.add("Speaker Diarization", Result.WARN, "token/terms issue")
        except Exception as e:
            _fail(str(e))
            R.add("Speaker Diarization", Result.FAIL, str(e)[:80])

    # ── Keyword Detection ─────────────────────────────────────────────────────
    print(f"\n  {BOLD}Running Keyword Detection...{RESET}")
    try:
        from echoguard.modules.keyword_detection.detector import KeywordDetector
        _step("Initializing KeywordDetector                  ")
        kd = KeywordDetector()
        _ok(f"{len(kd.keywords)} keywords loaded")
        _step("Running on sample text                        ")
        test_text = ("Your bank account has been blocked. "
                     "Please verify your identity immediately.")
        result = kd.detect(test_text)
        _ok(f"matched={result['matched_keywords']}, score={result['keyword_score']}")
        R.add("Keyword Detection", Result.OK)
        print(f"  {TICK} Keywords detected")
    except Exception as e:
        _fail(str(e))
        R.add("Keyword Detection", Result.FAIL, str(e)[:80])

    # ── wav2vec2 Emotion ──────────────────────────────────────────────────────
    print(f"\n  {BOLD}Running wav2vec2 Emotion Analysis...{RESET}")
    if not _pkg_installed("transformers"):
        _warn("transformers not installed — skipping")
        R.add("wav2vec2 Emotion", Result.SKIP, "transformers not installed")
    elif not _pkg_installed("librosa"):
        _warn("librosa not installed — skipping")
        _info("Fix: pip install librosa soundfile")
        R.add("wav2vec2 Emotion", Result.SKIP, "librosa not installed")
    else:
        try:
            from echoguard.modules.emotion.wav2vec2_emotion import EmotionAnalyzer
            _step("Initializing EmotionAnalyzer                  ")
            ea = EmotionAnalyzer()
            _ok(f"model_id={ea.model_id}, device={ea.device}")
            _step("Downloading wav2vec2 checkpoint & running      ")
            result = ea.analyze(audio_path)
            _ok(f"emotion={result['predicted_emotion']}, "
                f"confidence={result['confidence']}, "
                f"stress={result['stress_score']}")
            R.add("wav2vec2 Emotion", Result.OK)
            print(f"  {TICK} Emotion detected")
        except Exception as e:
            err = str(e)
            if "checkpoint" in err.lower() or "404" in err or "not found" in err.lower():
                _warn("Model checkpoint unavailable from HuggingFace")
                _info("The model superb/wav2vec2-base-superb-er may need HF access")
                _info("Or try: from_pretrained('facebook/wav2vec2-base')")
                R.add("wav2vec2 Emotion", Result.WARN, "checkpoint unavailable")
            else:
                _fail(err[:120])
                R.add("wav2vec2 Emotion", Result.FAIL, err[:80])

    # ── AASIST Deepfake Detection ─────────────────────────────────────────────
    print(f"\n  {BOLD}Running AASIST Deepfake Detection...{RESET}")
    if not _pkg_installed("torch"):
        _warn("torch not installed — skipping")
        R.add("AASIST Deepfake", Result.SKIP, "torch not installed")
    else:
        try:
            from echoguard.modules.deepfake.detector import DeepfakeDetector
            from echoguard.utils.config import load_config
            _step("Initializing DeepfakeDetector (AASIST)        ")
            cfg = load_config()
            weights = cfg.get("deepfake", {}).get("model_weights")
            detector = DeepfakeDetector(architecture="AASIST")
            _ok("initialized (architecture loaded)")

            _step("Checking for AASIST weights                   ")
            weights_path = None
            search_paths = [
                ROOT / "echoguard/models/deepfake/AASIST.pth",
                ROOT / "models/AASIST.pth",
                ROOT / "AASIST.pth",
            ]
            if weights and Path(weights).exists():
                weights_path = Path(weights)
            else:
                for p in search_paths:
                    if p.exists():
                        weights_path = p
                        break

            if weights_path:
                _ok(f"found at {weights_path}")
                _step("Running deepfake inference                    ")
                result = detector.detect(audio_path)
                _ok(f"is_deepfake={result['is_deepfake']}, "
                    f"spoof_score={result['spoof_score']}")
                R.add("AASIST Deepfake", Result.OK)
                print(f"  {TICK} Spoof probability computed")
            else:
                _warn("weights not found — cannot run inference")
                _info("Download: https://github.com/clovaai/aasist")
                _info("  → Models > AASIST.pth (or AASIST-L.pth)")
                _info("Place at: echoguard/models/deepfake/AASIST.pth")
                _info("Then set in echoguard/configs/default.yaml:")
                _info("  deepfake:")
                _info("    model_weights: echoguard/models/deepfake/AASIST.pth")
                R.add("AASIST Deepfake", Result.WARN, "weights missing")
                print(f"  {WARN} AASIST (weights missing)")
        except Exception as e:
            _fail(str(e)[:120])
            R.add("AASIST Deepfake", Result.FAIL, str(e)[:80])

    # ── Context Retrieval ─────────────────────────────────────────────────────
    print(f"\n  {BOLD}Running Context Retrieval...{RESET}")
    if not _pkg_installed("sentence_transformers"):
        _warn("sentence-transformers not installed — skipping")
        _info("Fix: pip install sentence-transformers faiss-cpu")
        R.add("Context Retrieval (FAISS)", Result.SKIP, "sentence-transformers missing")
    elif not _pkg_installed("faiss"):
        _warn("faiss-cpu not installed — skipping")
        _info("Fix: pip install faiss-cpu")
        R.add("Context Retrieval (FAISS)", Result.SKIP, "faiss-cpu missing")
    else:
        try:
            from echoguard.modules.context.retrieval import ContextRetriever
            _step("Building in-memory FAISS index (2 examples)   ")
            cr = ContextRetriever(top_k=2)
            cr.build_index(
                texts=["Your account has been blocked.",
                       "Your flight to London is confirmed."],
                labels=["scam", "benign"],
            )
            _ok("index built")
            _step("Running similarity retrieval                   ")
            results = cr.retrieve("Your bank account is suspended.")
            top = results[0] if results else {}
            _ok(f"top match: \"{top.get('text','')}\" "
                f"label={top.get('label','')} score={top.get('score','')}")
            R.add("Context Retrieval (FAISS)", Result.OK)
        except Exception as e:
            _fail(str(e)[:120])
            R.add("Context Retrieval (FAISS)", Result.FAIL, str(e)[:80])

    # ── Fusion ────────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Running Score Fusion...{RESET}")
    try:
        from echoguard.modules.fusion.fusion import ScoreFusion
        _step("Initializing ScoreFusion                      ")
        fusion = ScoreFusion()
        _ok(f"weights={fusion.weights}")
        _step("Fusing mock module outputs                    ")
        mock = {
            "keyword":  {"keyword_score": 0.6,  "is_flagged": True},
            "emotion":  {"stress_score":  0.74},
            "deepfake": {"spoof_score":   0.12},
        }
        result = fusion.fuse(mock)
        _ok(f"verdict={result['verdict']}, "
            f"scam_probability={result['scam_probability']}, "
            f"confidence={result['confidence']}")
        R.add("Fusion", Result.OK)
    except Exception as e:
        _fail(str(e)[:120])
        R.add("Fusion", Result.FAIL, str(e)[:80])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Diarized transcript integration test
# ─────────────────────────────────────────────────────────────────────────────

def run_integration_test(audio_path: Path):
    _header("6 / 6  Integration Test  (Whisper + Diarization merge)")

    hf_token = os.environ.get("HF_TOKEN")
    has_whisper  = _pkg_installed("whisper")
    has_pyannote = _pkg_installed("pyannote.audio") and bool(hf_token)

    if not has_whisper:
        _warn("Whisper not installed — skipping integration test")
        R.add("Diarized transcript (integration)", Result.SKIP,
              "openai-whisper not installed")
        return

    # If pyannote not available, test the merge logic with mock segments
    print(f"  Testing word-to-speaker assignment with {'real pyannote' if has_pyannote else 'mock segments'}...")

    try:
        from echoguard.modules.whisper.transcribe import (
            _assign_words_to_speakers,
            _group_words_by_speaker,
        )
        from echoguard.utils.text import format_diarized_transcript

        mock_words = [
            {"word": "Hello",  "start": 0.0, "end": 0.3},
            {"word": "sir",    "start": 0.3, "end": 0.6},
            {"word": "Hello",  "start": 1.0, "end": 1.2},
            {"word": "Your",   "start": 2.0, "end": 2.2},
            {"word": "account","start": 2.2, "end": 2.6},
            {"word": "is",     "start": 2.6, "end": 2.7},
            {"word": "blocked","start": 2.7, "end": 3.0},
            {"word": "Really", "start": 3.5, "end": 3.8},
        ]
        mock_segs = [
            {"speaker": "SPEAKER_00", "start": 0.0,  "end": 0.8},
            {"speaker": "SPEAKER_01", "start": 0.9,  "end": 1.5},
            {"speaker": "SPEAKER_00", "start": 1.9,  "end": 3.2},
            {"speaker": "SPEAKER_01", "start": 3.4,  "end": 4.0},
        ]

        _step("Word-to-speaker assignment                    ")
        assigned = _assign_words_to_speakers(mock_words, mock_segs)
        assert all("speaker" in w for w in assigned), "Assignment failed"
        _ok(f"{len(assigned)} words assigned")

        _step("Grouping words into speaker turns             ")
        grouped = _group_words_by_speaker(assigned)
        _ok(f"{len(grouped)} turns")

        _step("Formatting diarized transcript                ")
        formatted = format_diarized_transcript(grouped)
        _ok("success")
        print()
        for line in formatted.splitlines():
            print(f"       {line}")

        # Save to diarized_transcripts/
        out_dir = ROOT / "echoguard" / "diarized_transcripts"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "test_mock.txt").write_text(formatted, encoding="utf-8")
        _info(f"Saved to echoguard/diarized_transcripts/test_mock.txt")

        R.add("Diarized transcript (integration)", Result.OK)
        print(f"\n  {TICK} Integration test passed")

    except Exception as e:
        _fail(str(e)[:120])
        R.add("Diarized transcript (integration)", Result.FAIL, str(e)[:80])

# ─────────────────────────────────────────────────────────────────────────────
# Unreferenced file check
# ─────────────────────────────────────────────────────────────────────────────

def check_unreferenced_files():
    _header("Unreferenced File Check")

    import ast

    # Collect all .py source files
    all_py = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for f in files:
            if f.endswith(".py"):
                all_py.append(Path(root) / f)

    # Collect every string that appears as an import target across all files
    referenced_modules: set = set()
    for fpath in all_py:
        try:
            src = fpath.read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    referenced_modules.add(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        referenced_modules.add(alias.name)
        except Exception:
            pass

    # Check which project .py files are never imported
    unreferenced = []
    for fpath in all_py:
        rel = fpath.relative_to(ROOT)
        # Convert path to dotted module name
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            mod = ".".join(parts[:-1])
        else:
            mod = ".".join(parts).removesuffix(".py")

        # Skip the test script itself and main.py (they are entry points)
        if rel.name in ("test_pipeline.py", "main.py"):
            continue

        if mod not in referenced_modules:
            # Check if any sub-path is referenced (e.g. echoguard.utils)
            if not any(r.startswith(mod) or mod.startswith(r)
                       for r in referenced_modules):
                unreferenced.append(str(rel))

    if unreferenced:
        print(f"  {WARN}  The following files are not imported anywhere:")
        for f in unreferenced:
            print(f"       {f}")
        print()
        print("  These may be safe to delete, but please verify manually")
        print("  before removing them.")
        R.add("Unreferenced files", Result.WARN,
              f"{len(unreferenced)} file(s) — see above")
    else:
        _ok("All source files are referenced")
        print()
        R.add("Unreferenced files", Result.OK)

# ─────────────────────────────────────────────────────────────────────────────
# Missing-asset summary helper
# ─────────────────────────────────────────────────────────────────────────────

def print_missing_assets():
    _header("What Is Still Missing")

    items = [
        (
            "AASIST model weights",
            "echoguard/models/deepfake/AASIST.pth",
            [
                "1. Visit https://github.com/clovaai/aasist",
                "2. Download AASIST.pth from the Releases page",
                "3. Place it at:  echoguard/models/deepfake/AASIST.pth",
                "4. Update echoguard/configs/default.yaml:",
                "     deepfake:",
                "       model_weights: echoguard/models/deepfake/AASIST.pth",
            ],
        ),
        (
            "pyannote.audio package",
            None,
            [
                "pip install pyannote.audio",
                "Then accept terms of use at:",
                "  https://hf.co/pyannote/speaker-diarization-3.1",
                "  https://hf.co/pyannote/segmentation-3.0",
            ],
        ),
        (
            "HuggingFace token (HF_TOKEN)",
            None,
            [
                "1. Create a token at https://huggingface.co/settings/tokens",
                "2. export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx",
                "   (add to your shell profile for persistence)",
            ],
        ),
        (
            "librosa + soundfile",
            None,
            [
                "pip install librosa soundfile resampy",
                "(required by emotion analysis and deepfake detection)",
            ],
        ),
        (
            "faiss-cpu",
            None,
            [
                "pip install faiss-cpu",
                "(required by context retrieval module)",
            ],
        ),
        (
            "Real audio files",
            "echoguard/audio/",
            [
                "Place WAV files in echoguard/audio/",
                "  e.g. echoguard/audio/call001.wav",
                "Run:  python main.py --audio echoguard/audio/call001.wav",
            ],
        ),
    ]

    for title, check_path, instructions in items:
        already_present = False
        if check_path:
            already_present = (ROOT / check_path).exists()

        if already_present:
            print(f"  {TICK}  {title}")
        else:
            print(f"  {WARN}  {YELLOW}{title}{RESET}")
            for line in instructions:
                print(f"       {line}")
            print()

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EchoGuard pipeline verification script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_pipeline.py\n"
            "  python test_pipeline.py --audio echoguard/audio/call001.wav\n"
        ),
    )
    parser.add_argument(
        "--audio",
        default=None,
        metavar="PATH",
        help="Path to a WAV file. A synthetic 3s tone is used if omitted.",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{SEP}{RESET}")
    print(f"{BOLD}  EchoGuard — Pipeline Verification{RESET}")
    print(f"{BOLD}{SEP}{RESET}")

    # ── Resolve audio file ────────────────────────────────────────────────────
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"\n{RED}[ERROR]{RESET} Audio file not found: {audio_path}")
            sys.exit(1)
        print(f"\n  Using provided audio: {audio_path}")
    else:
        audio_path = ROOT / "echoguard" / "audio" / "_test_tone.wav"
        if not audio_path.exists():
            print(f"\n  {YELLOW}No audio file provided — generating a synthetic 3s tone.{RESET}")
            _make_synthetic_wav(audio_path)
            print(f"  Synthetic WAV written to: {audio_path}")
        else:
            print(f"\n  Using existing synthetic tone: {audio_path}")

    # ── Run all checks ────────────────────────────────────────────────────────
    check_packages()
    check_imports()
    check_structure()
    check_config()
    run_module_tests(audio_path)
    run_integration_test(audio_path)
    check_unreferenced_files()

    # ── Final summary ─────────────────────────────────────────────────────────
    print_missing_assets()
    R.print_summary()
    print(f"  Run the full pipeline with:")
    print(f"  {BOLD}python main.py --audio echoguard/audio/call001.wav{RESET}")
    print()


if __name__ == "__main__":
    main()
