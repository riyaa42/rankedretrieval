"""Basic inverted-index construction for the clothing corpus."""

from collections import Counter

from src.preprocessing import preprocess


def build_inverted_index(documents):
    """Build an inverted index and processed-token length for each document."""
    inverted_index = {}
    document_lengths = {}

    for document in documents:
        docid = document["docid"]

        # All searchable fields from the existing corpus structure are
        # preprocessed in the same way.
        searchable_text = " ".join(
            document[field] for field in ("title", "category", "text")
        )
        processed_tokens = preprocess(searchable_text)

        # Length is based on processed tokens so it can later be used for
        # cosine normalization with the same representation as the index.
        document_lengths[docid] = len(processed_tokens)

        # Count each term within this document before adding it to postings.
        # This keeps one posting per document and stores its term frequency.
        term_frequencies = Counter(processed_tokens)
        for term, frequency in term_frequencies.items():
            if term not in inverted_index:
                inverted_index[term] = {"df": 0, "postings": {}}

            inverted_index[term]["postings"][docid] = frequency

    for entry in inverted_index.values():
        # Document frequency counts distinct document IDs, not total term uses.
        entry["df"] = len(entry["postings"])

    return inverted_index, document_lengths
