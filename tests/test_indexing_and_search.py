"""Automated unit tests for indexing and search functionality."""

import math
from pathlib import Path
import pytest

from src.corpusloader import load_corpus
from src.indexer import build_inverted_index, build_positional_index
from src.search import (
    search,
    phrase_search,
    proximity_search,
    parse_proximity_query,
)

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus_100.txt"


@pytest.fixture(scope="module")
def corpus_data():
    documents = load_corpus(CORPUS_PATH)
    inverted_index, document_lengths = build_inverted_index(documents)
    positional_index = build_positional_index(documents)
    return {
        "documents": documents,
        "inverted_index": inverted_index,
        "document_lengths": document_lengths,
        "positional_index": positional_index,
    }


def test_inverted_index_and_euclidean_lengths(corpus_data):
    inverted_index = corpus_data["inverted_index"]
    document_lengths = corpus_data["document_lengths"]
    documents = corpus_data["documents"]

    assert len(document_lengths) == len(documents) == 100
    for docid, length in document_lengths.items():
        assert isinstance(length, float)
        assert length > 0.0

    # Ensure every term in inverted_index has valid df matching postings length
    for term, data in inverted_index.items():
        assert data["df"] == len(data["postings"])
        assert data["df"] > 0
        for docid, tf in data["postings"].items():
            assert tf > 0


def test_vsm_search_ranking(corpus_data):
    results = search(
        query="cotton shirt",
        inverted_index=corpus_data["inverted_index"],
        document_lengths=corpus_data["document_lengths"],
        documents=corpus_data["documents"],
        top_k=10,
    )

    assert len(results) <= 10
    assert len(results) > 0

    # Check structure of results
    for r in results:
        assert "docid" in r
        assert "score" in r
        assert "title" in r
        assert "category" in r
        assert 0.0 <= r["score"] <= 1.0

    # Check score ordering: descending score, then ascending docid for ties
    for i in range(len(results) - 1):
        if math.isclose(results[i]["score"], results[i + 1]["score"], rel_tol=1e-9):
            assert results[i]["docid"] <= results[i + 1]["docid"]
        else:
            assert results[i]["score"] >= results[i + 1]["score"]


def test_vsm_search_empty_and_oov(corpus_data):
    empty_results = search(
        query="",
        inverted_index=corpus_data["inverted_index"],
        document_lengths=corpus_data["document_lengths"],
        documents=corpus_data["documents"],
    )
    assert empty_results == []

    oov_results = search(
        query="nonexistentclothingtermxyz",
        inverted_index=corpus_data["inverted_index"],
        document_lengths=corpus_data["document_lengths"],
        documents=corpus_data["documents"],
    )
    assert oov_results == []


def test_phrase_search(corpus_data):
    results = phrase_search(
        phrase="cotton shirt",
        positional_index=corpus_data["positional_index"],
        documents=corpus_data["documents"],
    )
    assert len(results) > 0
    for r in results:
        assert "docid" in r
        assert "positions" in r
        assert "title" in r
        assert len(r["positions"]) > 0

    # Non-consecutive words should not match phrase search
    non_consecutive = phrase_search(
        phrase="cotton black",
        positional_index=corpus_data["positional_index"],
        documents=corpus_data["documents"],
    )
    assert non_consecutive == []


def test_parse_proximity_query():
    assert parse_proximity_query("cotton WITHIN/3 shirt") == ("cotton", "shirt", 3)
    assert parse_proximity_query("stretch within/4 denim") == ("stretch", "denim", 4)
    assert parse_proximity_query('"winter" WITHIN/2 "wear"') == ("winter", "wear", 2)
    assert parse_proximity_query("festive / 5 kurta") == ("festive", "kurta", 5)
    assert parse_proximity_query("invalid proximity query") is None
    assert parse_proximity_query("") is None


def test_proximity_search(corpus_data):
    # cotton WITHIN/3 shirt
    results = proximity_search(
        term1="cotton",
        term2="shirt",
        k=3,
        positional_index=corpus_data["positional_index"],
        documents=corpus_data["documents"],
    )

    assert len(results) > 0
    for r in results:
        assert "docid" in r
        assert "term1_position" in r
        assert "term2_position" in r
        assert "distance" in r
        assert 1 <= r["distance"] <= 3
        assert r["term2_position"] > r["term1_position"]
        assert "title" in r
        assert "category" in r

    with pytest.raises(ValueError):
        proximity_search("cotton", "shirt", 0, corpus_data["positional_index"])

    with pytest.raises(ValueError):
        proximity_search("cotton", "shirt", -1, corpus_data["positional_index"])
