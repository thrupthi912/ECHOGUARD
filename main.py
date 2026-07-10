"""
EchoGuard — End-to-End Phone Scam Detection Pipeline
=====================================================
Usage
-----
    python main.py --audio echoguard/audio/call001.wav
    python main.py --audio call001.wav --model small --speakers 2
    python main.py --audio call001.wav --skip-deepfake  # no weights needed

Full pipeline
-------------
    Audio
      │
      ├─► [0] Noise Cancellation               [optional — skippable]
      │         SpeechBrain (neural) or DSP fallback
      │         └─► Enhanced audio passed to all downstream modules
      │
      ├─► [1] Speaker Diarization (pyannote)
      │         └─► Whisper (word timestamps)
      │                   └─► Diarized transcript  ──► Keyword Detection
      │
      ├─► [2] Emotion Analysis (wav2vec2)
      │
      ├─► [3] Deepfake Detection (AASIST)      [optional — needs .pth weights]
      │
      ├─► [4] Context Retrieval (FAISS)        [optional — needs saved index]
      │
      └─► Score Fusion  ──► Verdict (SCAM / SUSPICIOUS / BENIGN)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Disable SSL cert verification for corporate/university proxy networks.
# Must be imported before any huggingface_hub / requests calls.
import echoguard.utils.ssl_patch  # noqa: F401


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    audio_path: str | Path,
    whisper_model: str = "base",
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    skip_noise_cancel: bool = False,
    skip_diarization: bool = False,
    skip_emotion: bool = False,
    skip_deepfake: bool = False,
    skip_context: bool = False,
    noise_cancel_strategy: str = "dsp",
    deepfake_weights: Optional[str] = None,
    context_index_dir: Optional[str] = None,
    output_dir: str | Path = "echoguard/outputs",
) -> Dict:
    """Run the full EchoGuard scam detection pipeline on a single audio file.

    Parameters
    ----------
    audio_path:
        Path to the input WAV file.
    whisper_model:
        Whisper model size (tiny | base | small | medium | large).
    hf_token:
        HuggingFace token for pyannote speaker diarization.
        Falls back to the ``HF_TOKEN`` environment variable.
    num_speakers:
        Hint for the number of speakers in the call.
    skip_noise_cancel:
        Skip noise cancellation (step 0). When skipped the original audio
        is passed directly to all downstream modules.
    skip_diarization:
        If ``True``, run plain Whisper transcription instead of
        speaker-diarized transcription.
    skip_emotion:
        Skip emotion analysis.
    skip_deepfake:
        Skip deepfake detection (useful when no weights are available).
    skip_context:
        Skip FAISS context retrieval.
    noise_cancel_strategy:
        Strategy for noise cancellation: ``"auto"`` | ``"speechbrain"`` |
        ``"dsp"``.  ``"auto"`` uses SpeechBrain if installed, otherwise
        falls back to the classical DSP pipeline.
    deepfake_weights:
        Path to AASIST .pth weights file.
    context_index_dir:
        Path to a saved FAISS index directory.
    output_dir:
        Root directory for all output files.

    Returns
    -------
    dict
        Full pipeline result including transcript, per-module outputs, and
        the final fusion verdict.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    results: Dict = {"audio_file": str(audio_path)}
    module_scores: Dict = {}

    # ── 0. Noise Cancellation ─────────────────────────────────────────────────
    # Runs before all other modules so Whisper, wav2vec2, and AASIST all
    # receive a cleaner audio signal.
    if not skip_noise_cancel:
        print("[0/5] Running noise cancellation...")
        from echoguard.modules.noise_cancellation import enhance_audio
        try:
            enhanced_path = enhance_audio(
                audio_path,
                strategy=noise_cancel_strategy,
                output_dir=output_dir,
            )
            audio_path = Path(enhanced_path)
            results["noise_cancellation"] = {
                "enhanced_file": str(audio_path),
                "strategy": noise_cancel_strategy,
            }
            print(f"    → Enhanced audio saved to: {audio_path}")
        except Exception as exc:
            print(f"    → [SKIPPED] Noise cancellation failed: {exc}")
            results["noise_cancellation"] = None
    else:
        print("[0/5] Noise cancellation skipped.")
        results["noise_cancellation"] = None

    # ── 1. Transcription ─────────────────────────────────────────────────────
    if skip_diarization:
        print("[1/5] Transcribing with Whisper (plain mode)...")
        from echoguard.modules.whisper import transcribe
        transcript = transcribe(
            audio_path,
            model_name=whisper_model,
            save=True,
        )
        results["transcript"] = transcript
        results["diarized_segments"] = []

    else:
        print("[1/5] Running speaker diarization + Whisper...")
        from echoguard.modules.whisper import diarized_transcribe
        from echoguard.utils.text import format_diarized_transcript

        token = hf_token or os.environ.get("HF_TOKEN")
        diarized_segments = diarized_transcribe(
            audio_path,
            model_name=whisper_model,
            save=True,
            hf_token=token,
            num_speakers=num_speakers,
        )
        results["diarized_segments"] = diarized_segments
        results["transcript"] = format_diarized_transcript(diarized_segments)
        transcript = " ".join(s["text"] for s in diarized_segments)

    print(f"    → Transcript length: {len(transcript)} characters")

    # ── 2. Keyword Detection ─────────────────────────────────────────────────
    print("[2/5] Running keyword detection...")
    from echoguard.modules.keyword_detection import detect_keywords
    keyword_result = detect_keywords(transcript)
    results["keyword_detection"] = keyword_result
    module_scores["keyword"] = keyword_result
    print(
        f"    → Matched: {keyword_result['matched_keywords']} "
        f"(score={keyword_result['keyword_score']})"
    )

    # ── 3. Emotion Analysis ───────────────────────────────────────────────────
    if not skip_emotion:
        print("[3/5] Analyzing emotion...")
        from echoguard.modules.emotion import analyze_emotion
        emotion_result = analyze_emotion(audio_path)
        results["emotion"] = emotion_result
        module_scores["emotion"] = emotion_result
        print(
            f"    → {emotion_result['predicted_emotion']} "
            f"(confidence={emotion_result['confidence']}, "
            f"stress={emotion_result['stress_score']})"
        )
    else:
        print("[3/5] Emotion analysis skipped.")
        results["emotion"] = None

    # ── 4. Deepfake Detection ─────────────────────────────────────────────────
    if not skip_deepfake:
        print("[4/5] Running deepfake detection...")
        from echoguard.modules.deepfake import detect_deepfake
        try:
            deepfake_result = detect_deepfake(
                audio_path,
                weights_path=deepfake_weights,
            )
            results["deepfake"] = deepfake_result
            module_scores["deepfake"] = deepfake_result
            print(
                f"    → is_deepfake={deepfake_result['is_deepfake']} "
                f"(spoof_score={deepfake_result['spoof_score']})"
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"    → [SKIPPED] {exc}")
            results["deepfake"] = None
    else:
        print("[4/5] Deepfake detection skipped.")
        results["deepfake"] = None

    # ── 5. Context Retrieval ──────────────────────────────────────────────────
    if not skip_context:
        print("[5/5] Running context retrieval...")
        from echoguard.modules.context import retrieve_context
        context_results = retrieve_context(
            transcript,
            index_dir=context_index_dir,
        )
        results["context"] = context_results
        if context_results:
            module_scores["context"] = context_results
            print(f"    → Retrieved {len(context_results)} nearest neighbours")
        else:
            print("    → [SKIPPED] No FAISS index found.")
    else:
        print("[5/5] Context retrieval skipped.")
        results["context"] = []

    # ── Fusion ────────────────────────────────────────────────────────────────
    print("\n[Fusion] Computing final scam probability...")
    from echoguard.modules.fusion import fuse_scores
    fusion_result = fuse_scores(module_scores)
    results["fusion"] = fusion_result

    verdict = fusion_result["verdict"]
    prob = fusion_result["scam_probability"]
    conf = fusion_result["confidence"]
    print(f"\n{'=' * 50}")
    print(f"  VERDICT : {verdict}")
    print(f"  SCORE   : {prob:.2%}")
    print(f"  CONFIDENCE: {conf}")
    print(f"{'=' * 50}\n")

    # ── Save full results ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{audio_path.stem}_result.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"Full results saved to: {out_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EchoGuard — Multimodal Phone Scam Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --audio echoguard/audio/call001.wav
  python main.py --audio call001.wav --model small --speakers 2
  python main.py --audio call001.wav --skip-deepfake --skip-diarization
  python main.py --audio call001.wav --noise-cancel-strategy dsp
  python main.py --audio call001.wav --skip-noise-cancel
        """,
    )
    parser.add_argument(
        "--audio",
        required=True,
        metavar="PATH",
        help="Path to the input WAV audio file.",
    )
    parser.add_argument(
        "--model",
        default="base",
        metavar="SIZE",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        metavar="N",
        help="Hint: number of speakers in the call (optional).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace token for pyannote (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--deepfake-weights",
        default=None,
        metavar="PATH",
        help="Path to AASIST .pth weight file.",
    )
    parser.add_argument(
        "--context-index",
        default=None,
        metavar="DIR",
        help="Path to saved FAISS context index directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="echoguard/outputs",
        metavar="DIR",
        help="Output directory for results (default: echoguard/outputs).",
    )
    parser.add_argument(
        "--skip-noise-cancel",
        action="store_true",
        help="Skip noise cancellation (step 0).",
    )
    parser.add_argument(
        "--noise-cancel-strategy",
        default="dsp",
        metavar="STRATEGY",
        choices=["auto", "speechbrain", "dsp"],
        help="Noise cancellation strategy: auto | speechbrain | dsp (default: dsp).",
    )
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip speaker diarization and use plain Whisper transcription.",
    )
    parser.add_argument(
        "--skip-emotion",
        action="store_true",
        help="Skip emotion analysis.",
    )
    parser.add_argument(
        "--skip-deepfake",
        action="store_true",
        help="Skip deepfake detection.",
    )
    parser.add_argument(
        "--skip-context",
        action="store_true",
        help="Skip FAISS context retrieval.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    run_pipeline(
        audio_path=args.audio,
        whisper_model=args.model,
        hf_token=args.hf_token,
        num_speakers=args.speakers,
        skip_noise_cancel=args.skip_noise_cancel,
        noise_cancel_strategy=args.noise_cancel_strategy,
        skip_diarization=args.skip_diarization,
        skip_emotion=args.skip_emotion,
        skip_deepfake=args.skip_deepfake,
        skip_context=args.skip_context,
        deepfake_weights=args.deepfake_weights,
        context_index_dir=args.context_index,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
