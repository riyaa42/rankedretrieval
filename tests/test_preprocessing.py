"""Automated checks for the shared text-preprocessing function."""

from nltk.stem import PorterStemmer

from src.preprocessing import STOP_WORDS, preprocess


def test_preprocess_returns_lowercase_list_with_stemming():
    result = preprocess("The Cotton Shirts are comfortable!")

    assert isinstance(result, list)
    assert all(isinstance(token, str) for token in result)
    assert "the" not in result
    assert "cotton" in result
    assert PorterStemmer().stem("shirts") in result
    assert "comfortable" not in result


def test_preprocess_removes_punctuation_and_finalized_stop_words():
    result = preprocess("The black, cotton shirt is fit and comfortable!")

    assert "the" not in result
    assert "is" not in result
    assert "and" not in result
    assert {"the", "is", "and"}.issubset(STOP_WORDS)
    assert "black" in result
    assert "cotton" in result
    assert "shirt" in result
    assert "fit" in result
    assert "comfort" in result


def test_clothing_terms_are_preserved_when_not_stop_words():
    result = preprocess("Cotton shirt fit wear black")

    for term in ("cotton", "shirt", "fit", "wear", "black"):
        assert term not in STOP_WORDS
        assert term in result


def test_preprocess_returns_empty_list_for_empty_input():
    assert preprocess("") == []
