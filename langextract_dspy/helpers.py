"""Shared utility functions for the langextract-dspy package.

Centralises helpers used by both :mod:`optimizer` and :mod:`config`
to avoid code duplication.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "extraction_key_set_from_dicts",
    "extraction_key_set_from_extractions",
]


def extraction_key_set_from_dicts(
    extractions: list[dict[str, str]],
) -> set[tuple[str, str]]:
    """Build a set of ``(class, text)`` keys from plain dicts.

    Used when parsing raw JSON extraction output (e.g. from DSPy
    predictions).

    Parameters:
        extractions: A list of dicts with ``extraction_class`` and
            ``extraction_text`` keys.

    Returns:
        A set of normalised ``(class, text)`` tuples.
    """
    keys: set[tuple[str, str]] = set()
    for ex in extractions:
        cls = ex.get("extraction_class", "").lower()
        txt = ex.get("extraction_text", "").lower()
        if cls and txt:
            keys.add((cls, txt))
    return keys


def extraction_key_set_from_extractions(
    extractions: list[Any],
) -> set[tuple[str, str]]:
    """Build a set of ``(class, text)`` keys from ``Extraction`` objects.

    Used when evaluating against ground-truth ``Extraction``
    instances.

    Parameters:
        extractions: A list of ``Extraction`` objects (or any
            object with ``extraction_class`` and
            ``extraction_text`` attributes).

    Returns:
        A set of normalised ``(class, text)`` tuples.
    """
    return {
        (e.extraction_class.lower(), e.extraction_text.lower()) for e in extractions
    }
