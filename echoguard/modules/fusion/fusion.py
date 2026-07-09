"""
echoguard.modules.fusion.fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Weighted score fusion for the EchoGuard scam detection pipeline.

Combines the outputs of all sub-modules into a single ``scam_probability``
score and a human-readable ``verdict``.

Fusion weights are configured in ``echoguard/configs/default.yaml``
under the ``fusion.weights`` key.

Score normalization contract
-----------------------------
Each module is expected to produce a score in [0, 1] where 1 = scam/high-risk:

    Module          Score key           Range
    ──────────────── ─────────────────── ──────
    keyword          keyword_score       [0, 1]
    emotion          stress_score        [0, 1]
    deepfake         spoof_score         [0, 1]
    context          mean score of       [0, 1]
                     scam neighbours
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------

_VERDICT_THRESHOLDS = {
    "SCAM": 0.65,
    "SUSPICIOUS": 0.40,
    "BENIGN": 0.0,
}


def _context_to_score(context_results: List[Dict]) -> float:
    """Derive a single context score from FAISS retrieval results.

    Computes the mean similarity score of results labelled as scam,
    minus a discount for benign neighbours.

    Parameters
    ----------
    context_results:
        Output of :func:`~echoguard.modules.context.retrieve_context`.

    Returns
    -------
    float
        Context-based scam score in [0, 1].
    """
    if not context_results:
        return 0.5  # neutral when no index is available

    scam_scores = [r["score"] for r in context_results if r.get("label") == "scam"]
    benign_scores = [r["score"] for r in context_results if r.get("label") != "scam"]

    scam_mean = sum(scam_scores) / len(scam_scores) if scam_scores else 0.0
    benign_mean = sum(benign_scores) / len(benign_scores) if benign_scores else 0.0

    return round(max(min(scam_mean - 0.3 * benign_mean, 1.0), 0.0), 4)


class ScoreFusion:
    """Combine module scores into a final scam probability.

    Parameters
    ----------
    weights:
        Dict of ``{module_name: weight}`` where all weights sum to 1.0.
        Modules not present in the input are skipped and remaining weights
        are renormalized automatically.
    verdict_thresholds:
        Dict mapping verdict label to minimum score threshold (descending).

    Example
    -------
    >>> fusion = ScoreFusion()
    >>> result = fusion.fuse({
    ...     "keyword": {"keyword_score": 0.6},
    ...     "emotion": {"stress_score": 0.74},
    ...     "deepfake": {"spoof_score": 0.12},
    ... })
    >>> print(result["verdict"], result["scam_probability"])
    SUSPICIOUS 0.49
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        verdict_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        if weights is None:
            weights = self._load_weights_from_config()
        self.weights = weights
        self.verdict_thresholds = verdict_thresholds or dict(_VERDICT_THRESHOLDS)

    @staticmethod
    def _load_weights_from_config() -> Dict[str, float]:
        """Load fusion weights from the project config."""
        try:
            from echoguard.utils.config import load_config
            cfg = load_config()
            w = cfg.get("fusion", {}).get("weights")
            if w:
                return w
        except Exception:
            pass
        return {"keyword": 0.25, "emotion": 0.25, "deepfake": 0.25, "context": 0.25}

    def _extract_scores(self, module_outputs: Dict) -> Dict[str, float]:
        """Extract a normalized score from each module's output dict.

        Parameters
        ----------
        module_outputs:
            Mapping of module name to that module's output dict.

        Returns
        -------
        dict
            ``{module_name: score}`` with scores in [0, 1].
        """
        scores: Dict[str, float] = {}

        if "keyword" in module_outputs:
            kw = module_outputs["keyword"]
            scores["keyword"] = float(kw.get("keyword_score", 0.0))

        if "emotion" in module_outputs:
            em = module_outputs["emotion"]
            scores["emotion"] = float(em.get("stress_score", 0.0))

        if "deepfake" in module_outputs:
            df = module_outputs["deepfake"]
            scores["deepfake"] = float(df.get("spoof_score", 0.0))

        if "context" in module_outputs:
            ctx = module_outputs["context"]
            if isinstance(ctx, list):
                scores["context"] = _context_to_score(ctx)
            elif isinstance(ctx, dict):
                scores["context"] = float(ctx.get("context_score", 0.5))

        return scores

    def fuse(self, module_outputs: Dict) -> Dict:
        """Fuse module scores into a final prediction.

        Parameters
        ----------
        module_outputs:
            Dict of module name → module output dict.  Missing modules are
            handled gracefully by renormalizing weights.

        Returns
        -------
        dict
            ::

                {
                    "scam_probability": float,  # weighted average in [0, 1]
                    "verdict": str,             # "SCAM" | "SUSPICIOUS" | "BENIGN"
                    "confidence": str,          # "HIGH" | "MEDIUM" | "LOW"
                    "component_scores": dict,   # per-module scores used
                    "weights_used": dict,       # renormalized weights
                }
        """
        scores = self._extract_scores(module_outputs)

        # Renormalize weights for available modules only
        available = {k: v for k, v in self.weights.items() if k in scores}
        total_weight = sum(available.values())

        if total_weight == 0:
            return {
                "scam_probability": 0.0,
                "verdict": "BENIGN",
                "confidence": "LOW",
                "component_scores": scores,
                "weights_used": {},
            }

        normalized_weights = {k: v / total_weight for k, v in available.items()}

        # Weighted average
        scam_prob = sum(
            scores[k] * normalized_weights[k] for k in normalized_weights
        )
        scam_prob = round(float(scam_prob), 4)

        # Verdict
        verdict = "BENIGN"
        for label, threshold in sorted(
            self.verdict_thresholds.items(), key=lambda x: -x[1]
        ):
            if scam_prob >= threshold:
                verdict = label
                break

        # Confidence — based on how far from the nearest boundary
        boundaries = sorted(self.verdict_thresholds.values(), reverse=True)
        min_dist = min(abs(scam_prob - b) for b in boundaries)
        if min_dist >= 0.2:
            confidence = "HIGH"
        elif min_dist >= 0.1:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "scam_probability": scam_prob,
            "verdict": verdict,
            "confidence": confidence,
            "component_scores": scores,
            "weights_used": normalized_weights,
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def fuse_scores(
    module_outputs: Dict,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Fuse module outputs into a final scam probability score.

    Parameters
    ----------
    module_outputs:
        Dict of module name → output dict.  Keys can be any subset of
        ``{"keyword", "emotion", "deepfake", "context"}``.
    weights:
        Custom fusion weights.  Loads from config if ``None``.

    Returns
    -------
    dict
        ``{"scam_probability": float, "verdict": str,
           "confidence": str, "component_scores": dict,
           "weights_used": dict}``

    Example
    -------
    >>> from echoguard.modules.fusion import fuse_scores
    >>> result = fuse_scores({
    ...     "keyword": {"keyword_score": 0.6},
    ...     "emotion": {"stress_score": 0.74},
    ... })
    >>> print(result["verdict"])
    SUSPICIOUS
    """
    fusion = ScoreFusion(weights=weights)
    return fusion.fuse(module_outputs)
