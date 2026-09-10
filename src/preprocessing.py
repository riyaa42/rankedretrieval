"""Text preprocessing shared by clothing documents and future user queries."""

import re

from nltk.stem import PorterStemmer


# Finalized after reviewing the automated NLTK stop-word analysis. Keep this
# set separate from NLTK's full list so the project retains clothing sizes such
# as M and S.
STOP_WORDS = {
    "and",
    "be",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "on",
    "or",
    "other",
    "the",
    "this",
    "with",
}

STEMMER = PorterStemmer()


def preprocess(text):
    """Lowercase, tokenize, remove selected stop words, and stem ``text``."""
    if not text:
        return []

    # The same ordered steps must be used for both corpus documents and queries.
    lowercase_text = text.lower()

    # Keep common T-shirt spellings together as one searchable clothing term.
    normalized_text = re.sub(
        r"\bt\s*(?:-\s*|\s+)shirt\b|\btshirt\b", "tshirt", lowercase_text
    )

    # Apostrophes are not meaningful tokens, so men's becomes mens.
    normalized_text = re.sub(r"['’]", "", normalized_text)

    # Keep alphabetic and numeric tokens while treating other punctuation as a
    # separator. This preserves values such as clothing sizes and 100.
    tokens = re.findall(r"[a-z0-9]+", normalized_text)

    # Only the manually finalized stop words are removed.
    tokens = [token for token in tokens if token not in STOP_WORDS]

    # Porter stemming reduces related word forms to a common stem.
    return [STEMMER.stem(token) for token in tokens]
