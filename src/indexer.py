"""Basic inverted-index construction for the clothing corpus."""

import math
from collections import Counter

from src.preprocessing import preprocess


def _searchable_text(document):
    """Combine the searchable fields provided by the corpus loader."""
    return " ".join(document[field] for field in ("title", "category", "text"))


def build_inverted_index(documents):
    """Build an inverted index and Euclidean document lengths for cosine normalization."""
    inverted_index = {}
    document_lengths = {}

    for document in documents:
        docid = document["docid"]

        # All searchable fields from the existing corpus structure are
        # preprocessed in the same way.
        processed_tokens = preprocess(_searchable_text(document))

        # Count each term within this document before adding it to postings.
        # This keeps one posting per document and stores its term frequency.
        term_frequencies = Counter(processed_tokens)
        for term, frequency in term_frequencies.items():
            if term not in inverted_index:
                inverted_index[term] = {"df": 0, "postings": {}}

            inverted_index[term]["postings"][docid] = frequency

        # Euclidean length (lnc document norm) for cosine normalization:
        # sqrt(sum((1 + log10(tf))**2))
        doc_norm_sq = sum(
            (1.0 + math.log10(freq)) ** 2 for freq in term_frequencies.values()
        )
        document_lengths[docid] = math.sqrt(doc_norm_sq)

    for entry in inverted_index.values():
        # Document frequency counts distinct document IDs, not total term uses.
        entry["df"] = len(entry["postings"])

    return inverted_index, document_lengths


def build_positional_index(documents):
    """Build a positional index using zero-based positions after preprocessing."""
    positional_index = {}

    for document in documents:
        docid = document["docid"]
        processed_tokens = preprocess(_searchable_text(document))

        # Positions are assigned only after normalization, stop-word removal,
        # and stemming, so they match the tokens stored in the index.
        for position, term in enumerate(processed_tokens):
            if term not in positional_index:
                positional_index[term] = {"df": 0, "postings": {}}

            postings = positional_index[term]["postings"]
            if docid not in postings:
                postings[docid] = {"tf": 0, "positions": []}

            postings[docid]["tf"] += 1
            postings[docid]["positions"].append(position)

    for entry in positional_index.values():
        # df counts the distinct document IDs in the term's postings.
        entry["df"] = len(entry["postings"])

    return positional_index
