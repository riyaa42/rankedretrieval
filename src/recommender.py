"""Optional recommender module for content-based product similarity and metadata filtering.

This module is isolated from the core assignment search engine. It provides:
1. filter_documents / get_allowed_doc_ids: Optional metadata filtering (e.g., by category).
2. find_similar_products: Content-based product recommendations using Vector Space
   cosine similarity over lnc term weights.
"""

import math


def filter_documents(documents, filters=None):
    """
    Filter a list of document dictionaries based on metadata fields.

    Parameters:
        documents (list): List of document dicts from load_corpus.
        filters (dict, optional): Metadata filters to apply.
            Example: {"category": "Shirt"} or {"category": ["Shirt", "T-Shirt"]}.
            Multiple filter fields are applied using AND logic.
            Matching is case-insensitive and ignores surrounding whitespace.

    Returns:
        list: Filtered list of document dictionaries. If filters is None or empty,
              returns all documents unchanged.
    """
    if not documents:
        return []

    if not filters:
        return list(documents)

    filtered = []
    for doc in documents:
        matches_all = True
        for key, filter_val in filters.items():
            if filter_val is None:
                continue

            # Safely get the document's string value for this metadata field
            doc_val = str(doc.get(key, "")).strip().lower()

            if isinstance(filter_val, (list, tuple, set)):
                allowed_vals = {str(v).strip().lower() for v in filter_val}
                if doc_val not in allowed_vals:
                    matches_all = False
                    break
            else:
                expected_val = str(filter_val).strip().lower()
                if doc_val != expected_val:
                    matches_all = False
                    break

        if matches_all:
            filtered.append(doc)

    return filtered


def get_allowed_doc_ids(documents, filters=None):
    """
    Return the set of docIDs that satisfy the given metadata filters.

    Parameters:
        documents (list): List of document dicts.
        filters (dict, optional): Filter criteria passed to filter_documents().

    Returns:
        set: Set of matching docID strings. If filters is None or empty,
             returns all docIDs.
    """
    if not documents:
        return set()

    if not filters:
        return {doc["docid"] for doc in documents if "docid" in doc}

    filtered_docs = filter_documents(documents, filters)
    return {doc["docid"] for doc in filtered_docs if "docid" in doc}


def find_similar_products(
    target_doc_id,
    documents,
    inverted_index,
    document_lengths,
    top_k=5,
    allowed_doc_ids=None,
):
    """
    Find top_k products most similar to a target product using content-based
    Vector Space Model (cosine similarity over lnc term weights).

    Parameters:
        target_doc_id (str): The docID of the product to find recommendations for.
        documents (list): List of all document dicts in the corpus.
        inverted_index (dict): The inverted index mapping terms to df and postings.
        document_lengths (dict): Precomputed Euclidean lengths for cosine normalization.
        top_k (int): Maximum number of recommendations to return (default: 5).
        allowed_doc_ids (set or list, optional): If provided, only recommend products
            from this subset of docIDs. If None, considers all documents.

    Returns:
        list of dict: Recommended products sorted by decreasing similarity score,
            breaking ties by increasing docID. Each dict contains:
            - 'docid': Document identifier
            - 'title': Product title
            - 'category': Product category
            - 'score': Cosine similarity score
    """
    # Gracefully handle missing or invalid inputs
    if not target_doc_id or not documents or not inverted_index or not document_lengths:
        return []

    if not isinstance(top_k, int) or top_k <= 0:
        return []

    # Verify that the target document exists in document_lengths
    target_norm = document_lengths.get(target_doc_id, 0.0)
    if target_norm == 0.0:
        return []

    # Convert allowed_doc_ids to a set for O(1) membership testing if provided
    allowed_set = set(allowed_doc_ids) if allowed_doc_ids is not None else None

    # Retrieve all terms occurring in the target document from the inverted index
    target_terms = {}
    for term, entry in inverted_index.items():
        if target_doc_id in entry["postings"]:
            target_terms[term] = entry["postings"][target_doc_id]

    if not target_terms:
        return []

    # Accumulate dot products between the target document vector and other documents
    # Document term weight under lnc: w = 1 + log10(tf)
    dot_products = {}
    for term, target_tf in target_terms.items():
        target_weight = 1.0 + math.log10(target_tf)

        for candidate_id, candidate_tf in inverted_index[term]["postings"].items():
            # Exclude the target product itself from its own recommendations
            if candidate_id == target_doc_id:
                continue

            # If allowed_doc_ids is provided, only consider products in that set
            if allowed_set is not None and candidate_id not in allowed_set:
                continue

            candidate_weight = 1.0 + math.log10(candidate_tf)
            dot_products[candidate_id] = (
                dot_products.get(candidate_id, 0.0) + target_weight * candidate_weight
            )

    # Build quick metadata lookup map by docID
    metadata_by_docid = {doc["docid"]: doc for doc in documents if "docid" in doc}

    # Compute cosine similarity: dot_product / (norm_target * norm_candidate)
    results = []
    for docid, dot_prod in dot_products.items():
        candidate_norm = document_lengths.get(docid, 0.0)
        if candidate_norm == 0.0:
            continue

        score = dot_prod / (target_norm * candidate_norm)
        # Cap score at 1.0 against minor floating-point rounding artifacts
        score = min(1.0, score)

        doc_meta = metadata_by_docid.get(docid, {})
        results.append(
            {
                "docid": docid,
                "title": doc_meta.get("title", ""),
                "category": doc_meta.get("category", ""),
                "score": score,
            }
        )

    # Sort deterministically: highest similarity score first; break ties by ascending docID
    results.sort(key=lambda item: (-item["score"], item["docid"]))

    return results[:top_k]
