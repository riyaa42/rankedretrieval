"""Runnable manual check for the positional index and exact phrase search."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpusloader import load_corpus
from src.indexer import build_positional_index
from src.preprocessing import preprocess
from src.search import phrase_search


def print_phrase_results(phrase, positional_index, documents):
    results = phrase_search(phrase, positional_index, documents)
    print(f"\nPhrase: {phrase}")
    if not results:
        print("No documents match.")
        return

    for result in results:
        print(f"{result['docid']}: starting positions {result['positions']}")


def documents_containing_all_terms(phrase, positional_index):
    """Show co-occurrence separately from exact phrase matching."""
    terms = preprocess(phrase)
    if not terms or any(term not in positional_index for term in terms):
        return []

    docids = set(positional_index[terms[0]]["postings"])
    for term in terms[1:]:
        docids &= set(positional_index[term]["postings"])
    return sorted(docids)


if __name__ == "__main__":
    documents = load_corpus(PROJECT_ROOT / "data" / "corpus_100.txt")
    positional_index = build_positional_index(documents)

    for phrase in (
        "cotton shirt",
        "stretch denim",
        "winter wear",
        "regular fit",
        "nonexistent phrase",
    ):
        print_phrase_results(phrase, positional_index, documents)

    non_consecutive_phrase = "cotton black"
    print(f"\nConsecutive-position check: {non_consecutive_phrase}")
    print(
        "Documents containing both terms:",
        documents_containing_all_terms(non_consecutive_phrase, positional_index),
    )
    print(
        "Exact phrase matches:",
        phrase_search(non_consecutive_phrase, positional_index, documents),
    )
