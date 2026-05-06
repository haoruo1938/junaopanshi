"""Target normalization utilities for natural-language commands."""

from __future__ import annotations


TARGET_SYNONYMS = {
    "sofa": {"sofa", "couch"},
    "bed": {"bed"},
    "table": {"table", "desk", "dining table"},
    "apple": {"apple"},
}


def normalize_target(text: str) -> str | None:
    """Map user command text into a canonical target class."""
    lower = text.lower().strip()
    for canonical, aliases in TARGET_SYNONYMS.items():
        if any(alias in lower for alias in aliases):
            return canonical
    return None
