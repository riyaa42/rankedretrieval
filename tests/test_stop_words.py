"""Report NLTK stop words that occur in the clothing corpus.

Run with: pytest -s tests/test_stop_words.py
"""

from collections import Counter
from pathlib import Path
import re

from nltk.corpus import stopwords

from src.corpusloader import load_corpus


CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus_100.txt"

# These words can sometimes describe garment features or styles, so their
# removal should be decided by hand rather than solely because NLTK lists them.
REVIEW_REASONS = {
    "down": "may refer to down-filled outerwear",
    "m": "appears as the clothing size M in product descriptions",
    "off": "may occur in style descriptions such as off-shoulder",
    "over": "may occur in garment or layering descriptions",
    "s": "appears as the clothing size S, as well as a possessive fragment",
    "under": "may occur in underlayer or underwire descriptions",
    "up": "may occur in style descriptions such as zip-up",
}


def tokenize(text):
    """Lowercase and keep alphabetic tokens for the stop-word report."""
    return re.findall(r"[a-z]+", text.lower())


def corpus_tokens(documents):
    """Collect tokens from every searchable document field."""
    for document in documents:
        # Title, category, and description can all be searched later, so the
        # analysis considers every one of them.
        searchable_text = " ".join(
            document[field] for field in ("title", "category", "text")
        )
        yield from tokenize(searchable_text)


def test_stop_word_analysis():
    documents = load_corpus(CORPUS_PATH)
    assert documents, f"No documents were loaded from {CORPUS_PATH}"

    token_counts = Counter(corpus_tokens(documents))
    nltk_stop_words = set(stopwords.words("english"))
    found_stop_words = {
        word: frequency
        for word, frequency in token_counts.items()
        if word in nltk_stop_words
    }

    assert found_stop_words, "No NLTK stop words were found in the loaded corpus"

    likely_safe = []
    needs_review = []
    for word in sorted(found_stop_words):
        frequency = found_stop_words[word]
        if word in REVIEW_REASONS:
            needs_review.append((word, frequency, REVIEW_REASONS[word]))
        else:
            likely_safe.append((word, frequency))

    print("\nLikely safe to remove")
    for word, frequency in likely_safe:
        print(f"{word} {frequency}")

    print("\nNeeds manual review")
    for word, frequency, reason in needs_review:
        print(f"{word} {frequency} - {reason}")

    print(f"\nTotal unique stop words found: {len(found_stop_words)}")
