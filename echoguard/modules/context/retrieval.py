"""
echoguard.modules.context.retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FAISS-based semantic context retrieval for scam pattern matching.

This module embeds a query transcript and retrieves the most semantically
similar examples from a FAISS index built over a reference corpus of known
scam and benign call transcripts.

The retrieved context can be used to:
  - Enrich classification decisions with nearest-neighbour evidence
  - Provide explainability ("this call is similar to known bank scam calls")
  - Feed a retrieval-augmented classification pipeline

Usage
-----
    # Build an index from a list of example transcripts
    retriever = ContextRetriever()
    retriever.build_index(texts, labels)
    retriever.save("echoguard/outputs/context.index")

    # At inference time
    results = retriever.retrieve("Your account has been suspended.")

Dependencies
------------
* faiss-cpu (or faiss-gpu) — pip install faiss-cpu
* sentence-transformers    — pip install sentence-transformers
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional


class ContextRetriever:
    """FAISS-based semantic retrieval over a transcript corpus.

    Parameters
    ----------
    embedding_model:
        SentenceTransformers model ID for encoding text.
    top_k:
        Number of nearest neighbours to return.

    Example
    -------
    >>> retriever = ContextRetriever()
    >>> retriever.build_index(
    ...     texts=["Your account is blocked.", "Flight booking confirmed."],
    ...     labels=["scam", "benign"],
    ... )
    >>> results = retriever.retrieve("Your account has been suspended.")
    >>> print(results[0]["text"], results[0]["label"])
    Your account is blocked. scam
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> None:
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._index = None
        self._texts: List[str] = []
        self._labels: List[str] = []
        self._encoder = None

    def _get_encoder(self):
        """Lazy-load the sentence encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for context retrieval.\n"
                    "Run: pip install sentence-transformers"
                ) from exc
            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    def _embed(self, texts: List[str]) -> "np.ndarray":
        """Encode a list of texts into dense vectors."""
        encoder = self._get_encoder()
        return encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def build_index(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
    ) -> None:
        """Build a FAISS index from a corpus of transcripts.

        Parameters
        ----------
        texts:
            List of transcript strings.
        labels:
            Optional list of labels (e.g. ``"scam"`` / ``"benign"``) aligned
            with ``texts``.
        """
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for context retrieval.\n"
                "Run: pip install faiss-cpu"
            ) from exc

        if labels is None:
            labels = ["unknown"] * len(texts)

        self._texts = list(texts)
        self._labels = list(labels)

        embeddings = self._embed(texts).astype(np.float32)
        dim = embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
        self._index.add(embeddings)

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve the most similar transcripts for a query string.

        Parameters
        ----------
        query:
            Query transcript string.

        Returns
        -------
        list of dict
            Up to ``top_k`` results, each::

                {
                    "text": str,
                    "label": str,
                    "score": float,  # cosine similarity in [0, 1]
                    "rank": int,
                }

        Raises
        ------
        RuntimeError
            If no index has been built or loaded yet.
        """
        import numpy as np

        if self._index is None:
            raise RuntimeError(
                "No FAISS index available.  "
                "Call build_index() or load() first."
            )

        query_vec = self._embed([query]).astype(np.float32)
        k = min(self.top_k, len(self._texts))
        scores, indices = self._index.search(query_vec, k)

        results: List[Dict] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue
            results.append(
                {
                    "text": self._texts[idx],
                    "label": self._labels[idx],
                    "score": round(float(score), 4),
                    "rank": rank + 1,
                }
            )
        return results

    def save(self, output_dir: str | Path) -> None:
        """Persist the FAISS index and corpus metadata to disk.

        Parameters
        ----------
        output_dir:
            Directory to save ``index.faiss`` and ``corpus.json``.
        """
        import faiss

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(output_dir / "index.faiss"))
        with open(output_dir / "corpus.json", "w", encoding="utf-8") as fh:
            json.dump(
                {"texts": self._texts, "labels": self._labels},
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def load(self, index_dir: str | Path) -> None:
        """Load a previously saved FAISS index from disk.

        Parameters
        ----------
        index_dir:
            Directory containing ``index.faiss`` and ``corpus.json``.
        """
        import faiss

        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        corpus_path = index_dir / "corpus.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus metadata not found: {corpus_path}")

        self._index = faiss.read_index(str(index_path))
        with open(corpus_path, "r", encoding="utf-8") as fh:
            corpus = json.load(fh)
        self._texts = corpus["texts"]
        self._labels = corpus["labels"]


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_retriever: Optional[ContextRetriever] = None


def retrieve_context(
    query: str,
    index_dir: Optional[str | Path] = None,
    top_k: int = 5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict]:
    """Retrieve semantically similar transcripts for a query.

    Parameters
    ----------
    query:
        Query transcript string.
    index_dir:
        Directory containing a saved FAISS index.  If ``None``, attempts to
        load from ``echoguard/outputs/context_index``.
    top_k:
        Number of results to return.
    embedding_model:
        SentenceTransformers model to use for embedding.

    Returns
    -------
    list of dict
        Nearest-neighbour results with ``text``, ``label``, ``score``, ``rank``.

    Example
    -------
    >>> from echoguard.modules.context import retrieve_context
    >>> results = retrieve_context("Your account is blocked.", top_k=3)
    """
    global _default_retriever

    if _default_retriever is None:
        _default_retriever = ContextRetriever(
            embedding_model=embedding_model,
            top_k=top_k,
        )

    if index_dir is None:
        default_dir = Path("echoguard/outputs/context_index")
        if default_dir.exists():
            _default_retriever.load(default_dir)
        else:
            # No index available — return empty results gracefully
            return []

    else:
        _default_retriever.load(index_dir)

    return _default_retriever.retrieve(query)
