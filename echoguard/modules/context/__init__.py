"""
echoguard.modules.context
~~~~~~~~~~~~~~~~~~~~~~~~~
FAISS-based context retrieval for scam pattern matching.

Public interface
----------------
    from echoguard.modules.context import retrieve_context

    results = retrieve_context("Your account has been blocked.")
    # [{"text": "...", "score": 0.91, "label": "scam"}, ...]
"""

from echoguard.modules.context.retrieval import retrieve_context, ContextRetriever

__all__ = ["retrieve_context", "ContextRetriever"]
