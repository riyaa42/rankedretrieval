"""lnc.ltc ranked retrieval for free-text clothing queries."""

from collections import Counter
import math
import re

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

    metadata_by_docid = {document["docid"]: document for document in documents}
    results = []
    for docid in candidate_docids:
        document_norm = document_lengths.get(docid, 0.0)
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


def proximity_search(term1, term2, k, positional_index, documents=None):
    """Return every ordered term pair whose distance is at most ``k`` tokens."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    term1_tokens = preprocess(term1)
    term2_tokens = preprocess(term2)
    if len(term1_tokens) != 1 or len(term2_tokens) != 1:
        return []

    processed_term1 = term1_tokens[0]
    processed_term2 = term2_tokens[0]
    if (
        processed_term1 not in positional_index
        or processed_term2 not in positional_index
    ):
        return []

    candidate_docids = set(positional_index[processed_term1]["postings"])
    candidate_docids &= set(positional_index[processed_term2]["postings"])

    metadata_by_docid = {}
    if documents is not None:
        metadata_by_docid = {document["docid"]: document for document in documents}

    results = []
    for docid in sorted(candidate_docids):
        term1_positions = positional_index[processed_term1]["postings"][docid][
            "positions"
        ]
        term2_positions = positional_index[processed_term2]["postings"][docid][
            "positions"
        ]

        # Ordered matching requires term1 before term2. k is the maximum
        # number of processed-token positions between the two occurrences.
        # Stored positions are necessary because co-occurrence alone cannot
        # prove either order or distance.
        for position1 in term1_positions:
            for position2 in term2_positions:
                if position2 <= position1:
                    continue

                distance = position2 - position1
                if distance > k:
                    break

                # Every valid pair is returned so repeated occurrences within
                # one document remain visible instead of being silently merged.
                result_item = {
                    "docid": docid,
                    "term1_position": position1,
                    "term2_position": position2,
                    "distance": distance,
                }
                if documents is not None:
                    document = metadata_by_docid.get(docid, {})
                    result_item["title"] = document.get("title", "")
                    result_item["category"] = document.get("category", "")
                results.append(result_item)

    return results


def parse_proximity_query(query):
    """
    Parse an ordered proximity query string such as 'cotton WITHIN/3 shirt'.

    Returns (term1, term2, k) if the query matches the pattern, or None otherwise.
    """
    if not isinstance(query, str):
        return None

    match = re.match(
        r"^\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?\s+(?:within\s*/\s*|/\s*)(\d+)\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?\s*$",
        query.strip(),
        re.IGNORECASE,
    )
    if not match:
        return None

    term1 = match.group(1)
    k = int(match.group(2))
    term2 = match.group(3)
    return term1, term2, k
