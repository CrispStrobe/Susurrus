"""Semantic search over transcription history via CrispEmbed (stub).

This module provides a semantic search interface for transcription history.
When CrispEmbed is available (binary or Python bindings), it uses embedding-based
similarity. Otherwise falls back to substring search.

CrispEmbed: https://github.com/CrispStrobe/CrispEmbed
"""

import logging

logger = logging.getLogger(__name__)


def is_available():
    """Check if CrispEmbed is available for semantic search."""
    try:
        import shutil

        return shutil.which("crispembed") is not None
    except Exception:
        return False


def semantic_search(query, entries, top_k=10):
    """Search history entries semantically.

    Falls back to substring search if CrispEmbed is not available.

    Args:
        query: Search query string.
        entries: List of HistoryEntry objects.
        top_k: Maximum results to return.

    Returns:
        List of (entry, score) tuples, sorted by relevance.
    """
    if not entries or not query:
        return []

    if is_available():
        return _embed_search(query, entries, top_k)
    return _substring_search(query, entries, top_k)


def _substring_search(query, entries, top_k):
    """Simple substring search fallback."""
    query_lower = query.lower()
    results = []
    for entry in entries:
        title = (entry.title or "").lower()
        text = (entry.full_text or "").lower()
        if query_lower in title:
            results.append((entry, 1.0))
        elif query_lower in text:
            results.append((entry, 0.5))
    return results[:top_k]


def _embed_search(query, entries, top_k):
    """Embedding-based search via CrispEmbed binary."""
    # Stub — will be implemented when CrispEmbed Python bindings are available.
    # For now, CrispEmbed exists as a C++ binary with Go/Dart/Rust bindings
    # but no Python ctypes wrapper yet.
    logger.info("CrispEmbed binary found but Python bindings not yet available, using substring")
    return _substring_search(query, entries, top_k)
