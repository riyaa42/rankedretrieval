"""lnc.ltc ranked retrieval for free-text clothing queries."""

from collections import Counter
import math

from src.preprocessing import preprocess


def search(query, inverted_index, document_lengths, documents, top_k=10):
    """Return up to ``top_k`` documents ranked by lnc.ltc cosine similarity."""
    query_terms = preprocess(query)
    if not query_terms or not document_lengths:
        return []

    document_count = len(document_lengths)
    query_frequencies = Counter(query_terms)
    query_weights = {}

    # Query weights use tf and idf. Document weights use only tf under lnc.
    for term, frequency in query_frequencies.items():
        entry = inverted_index.get(term)
        if not entry or entry["df"] == 0:
            continue

        weight = (1 + math.log10(frequency)) * math.log10(
            document_count / entry["df"]
        )
        if weight > 0:
            query_weights[term] = weight

    if not query_weights:
        return []

    query_norm = math.sqrt(sum(weight**2 for weight in query_weights.values()))
    if query_norm == 0:
        return []

    candidate_docids = set()
    for term in query_weights:
        candidate_docids.update(inverted_index[term]["postings"])

    # Calculate an lnc Euclidean norm only for matching documents. Processed
    # token counts are kept separately for later use, but are not vector norms.
    document_norm_squares = {docid: 0.0 for docid in candidate_docids}
    for entry in inverted_index.values():
        for docid, frequency in entry["postings"].items():
            if docid in document_norm_squares:
                document_weight = 1 + math.log10(frequency)
                document_norm_squares[docid] += document_weight**2

    metadata_by_docid = {document["docid"]: document for document in documents}
    results = []
    for docid in candidate_docids:
        document_norm = math.sqrt(document_norm_squares[docid])
        if document_norm == 0:
            continue

        dot_product = 0.0
        for term, query_weight in query_weights.items():
            frequency = inverted_index[term]["postings"].get(docid)
            if frequency:
                document_weight = 1 + math.log10(frequency)
                dot_product += (query_weight / query_norm) * (
                    document_weight / document_norm
                )

        document = metadata_by_docid.get(docid, {})
        results.append(
            {
                "docid": docid,
                "score": dot_product,
                "title": document.get("title", ""),
                "category": document.get("category", ""),
            }
        )

    results.sort(key=lambda result: (-result["score"], result["docid"]))
    return results[:top_k]


def phrase_search(phrase, positional_index, documents=None):
    """Return documents and starting positions where a phrase occurs exactly."""
    phrase_terms = preprocess(phrase)
    if not phrase_terms:
        return []

    if any(term not in positional_index for term in phrase_terms):
        return []

    candidate_docids = set(positional_index[phrase_terms[0]]["postings"])
    for term in phrase_terms[1:]:
        candidate_docids &= set(positional_index[term]["postings"])

    metadata_by_docid = {}
    if documents is not None:
        metadata_by_docid = {document["docid"]: document for document in documents}

    results = []
    for docid in sorted(candidate_docids):
        # A valid starting position must be followed by every next phrase term
        # at the immediately consecutive processed-token position.
        start_positions = positional_index[phrase_terms[0]]["postings"][docid][
            "positions"
        ]
        matching_positions = []
        for start in start_positions:
            if all(
                start + offset
                in positional_index[term]["postings"][docid]["positions"]
                for offset, term in enumerate(phrase_terms[1:], start=1)
            ):
                matching_positions.append(start)

        if matching_positions:
            result = {"docid": docid, "positions": matching_positions}
            if documents is not None:
                document = metadata_by_docid.get(docid, {})
                result["title"] = document.get("title", "")
                result["category"] = document.get("category", "")
            results.append(result)

    return results
