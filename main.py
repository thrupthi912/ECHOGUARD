"""
EchoGuard — End-to-End Phone Scam Detection Pipeline
=====================================================
Usage
-----
    python main.py --audio echoguard/audio/call001.wav
    python main.py --audio call001.wav --model small --speakers 2
    python main.py --audio call001.wav --skip-deepfake

Full pipeline
-------------
    Audio
      │
      ├─► [1] Speaker Diarization (pyannote)
      │         └─► Whisper (word timestamps)
      │                   └─► Diarized transcript ──► Keyword Detection
      │
      ├─► [2] Emotion Analysis (wav2vec2)
      │
      ├─► [3] Deepfake Detection (AASIST)      [needs .pth weights]
      │
      ├─► [4] Context Retrieval (FAISS)        [needs saved index]
      │
      └─► Score Fusion ──► Verdict (SCAM / SUSPICIOUS / BENIGN)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional


def run_pipeline(
    audio_path: str | Path,
    whisper_model: str = "base",
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    skip_diarization: bool = False,
    skip_emotion: bool = False,
    skip_deepfake: bool = False,
    skip_context: bool = False,
    deepfake_weights: Optional[str] = None,
    context_index_dir: Optional[str] = None,
    output_dir: str | Path = "echoguard/outputs",
) -> Dict:
    """Run the full EchoGuard scam detection pipeline on a single audio file.

    Parameters
    ----------
    audio_path : Path to the input WAV file.
    whisper_model : Whisper model size (tiny | base | small | medium | large).
    hf_token : HuggingFace token for pyannote. Falls back to HF_TOKEN env var.
    num_speakers : Hint for number of speakers in the call.
    skip_diarization : Use plain Whisper instead of speaker-diarized transcript.
    skip_emotion : Skip wav2vec2 emotion analysis.
    skip_deepfake : Skip AASIST deepfake detection.
    skip_context : Skip FAISS context retrieval.
    deepfake_weights : Path to AASIST .pth weights file.
    context_index_dir : Path to a saved FAISS index directory.
    output_dir : Root directory for all output files.

    Returns
    -------
    dict
        Full pipeline result with transcript, per-module outputs, and verdict.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    results: Dict = {"audio_file": str(audio_path)}
    module_scores: Dict = {}

    # ── 1. Transcription ─────────────────────────────────────────────────────
    if skip_diarization:
        print("[1/4] Transcribing with Whisper (plain mode)...")
        from echoguard.modules.whisper import transcribe
        transcript = transcribe(audio_path, model_name=whisper_model, save=True)
        results["transcript"] = transcript
        results["diarized_segments"] = []

    else:
        print("[1/4] Running speaker diarization + Whisper...")
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
    print("[2/4] Running keyword detection...")
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
        print("[3/4] Analyzing emotion...")
        from echoguard.modules.emotion import analyze_emotion
        try:
            emotion_result = analyze_emotion(audio_path)
            results["emotion"] = emotion_result
            module_scores["emotion"] = emotion_result
            print(
                f"    → {emotion_result['predicted_emotion']} "
                f"(confidence={emotion_result['confidence']}, "
                f"stress={emotion_result['stress_score']})"
            )
        except Exception as exc:
            print(f"    → [SKIPPED] {exc}")
            results["emotion"] = None
    else:
        print("[3/4] Emotion analysis skipped.")
        results["emotion"] = None

    # ── 4. Deepfake Detection ─────────────────────────────────────────────────
    if not skip_deepfake:
        print("[4/4] Running deepfake detection...")
        from echoguard.modules.deepfake import detect_deepfake
        try:
            deepfake_result = detect_deepfake(audio_path, weights_path=deepfake_weights)
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
        print("[4/4] Deepfake detection skipped.")
        results["deepfake"] = None

    # ── Context Retrieval (bonus, no step counter) ────────────────────────────
    if not skip_context:
        from echoguard.modules.context import retrieve_context
        context_results = retrieve_context(transcript, index_dir=context_index_dir)
        results["context"] = context_results
        if context_results:
            module_scores["context"] = context_results
    else:
        results["context"] = []

    # ── Fusion ────────────────────────────────────────────────────────────────
    print("\n[Fusion] Computing final scam probability...")
    from echoguard.modules.fusion import fuse_scores
    fusion_result = fuse_scores(module_scores)
    results["fusion"] = fusion_result

    verdict   = fusion_result["verdict"]
    prob      = fusion_result["scam_probability"]
    conf      = fusion_result["confidence"]
    scores    = fusion_result["component_scores"]

    print(f"\n{'=' * 50}")
    print(f"  VERDICT    : {verdict}")
    print(f"  SCORE      : {prob:.2%}")
    print(f"  CONFIDENCE : {conf}")
    print(f"  ── Component Scores ──────────────────")
    for mod, score in scores.items():
        print(f"    {mod:<12}: {score:.2%}")
    print(f"{'=' * 50}\n")

    # ── Save full results ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize filename (remove characters that are invalid on some OSes)
    safe_stem = "".join(
        c if c.isalnum() or c in "-_. " else "_" for c in audio_path.stem
    ).strip()
    out_file = output_dir / f"{safe_stem}_result.json"
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
  # Fastest demo (no extra packages needed):
  python main.py --audio "echoguard/audio/call001.wav" --skip-diarization --skip-emotion --skip-deepfake

  # With speaker diarization (needs pyannote.audio + HF_TOKEN):
  python main.py --audio "echoguard/audio/call001.wav" --speakers 2

  # Full pipeline minus deepfake (no .pth needed):
  python main.py --audio "echoguard/audio/call001.wav" --speakers 2 --skip-deepfake
        """,
    )
    parser.add_argument("--audio", required=True, metavar="PATH",
                        help="Path to the input WAV audio file.")
    parser.add_argument("--model", default="base", metavar="SIZE",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base).")
    parser.add_argument("--speakers", type=int, default=None, metavar="N",
                        help="Number of speakers in the call (optional hint).")
    parser.add_argument("--hf-token", default=None, metavar="TOKEN",
                        help="HuggingFace token for pyannote (or set HF_TOKEN env var).")
    parser.add_argument("--deepfake-weights", default=None, metavar="PATH",
                        help="Path to AASIST .pth weight file.")
    parser.add_argument("--context-index", default=None, metavar="DIR",
                        help="Path to saved FAISS index directory.")
    parser.add_argument("--output-dir", default="echoguard/outputs", metavar="DIR",
                        help="Output directory for results.")
    parser.add_argument("--skip-diarization", action="store_true",
                        help="Use plain Whisper instead of speaker-diarized transcript.")
    parser.add_argument("--skip-emotion", action="store_true",
                        help="Skip emotion analysis.")
    parser.add_argument("--skip-deepfake", action="store_true",
                        help="Skip deepfake detection.")
    parser.add_argument("--skip-context", action="store_true",
                        help="Skip FAISS context retrieval.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_pipeline(
        audio_path=args.audio,
        whisper_model=args.model,
        hf_token=args.hf_token,
        num_speakers=args.speakers,
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
