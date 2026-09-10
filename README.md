# Clothing Ranked Retrieval

## TODO: Clothing Information Retrieval Project

- [x] Implement corpus loader for `DOCID`, title/category, and description.

- [x] Implement preprocessing:
  - [x] Convert text to lowercase.
  - [x] Tokenize text.
  - [x] Remove punctuation.
  - [x] Decide and document the stop-word policy.
  - [x] Apply stemming.

- [x] Build the inverted index:
  - [x] Store term frequency.
  - [x] Store document frequency.

- [x] Store document lengths for cosine normalization.

- [x] Implement `lnc.ltc` cosine-similarity ranking.

- [x] Return the top 10 documents for each free-text query.

- [x] Build the positional index with token positions.

- [x] Implement exact phrase search.

- [x] Implement ordered proximity search with different `k` values.

- [x] Add novelty backend extension:
  - [x] Implement content-based similar product recommendation using vector space similarity.
  - [x] Implement optional metadata filtering (e.g. by category).

- [x] Create a simple interface supporting:
  - [x] Free-text search.
  - [x] Phrase search.
  - [x] Proximity search.
  - [x] Displaying scores and matching positions.

- [ ] Test the system with:
  - [ ] 10 free-text queries.
  - [x] 5 phrase queries.
  - [ ] 3 proximity queries with different `k` values.
  - [ ] 1 out-of-vocabulary query.

- [ ] Compare ordinary retrieval and positional retrieval using two examples.

## Stop-word policy

The corpus was analysed automatically using the existing corpus loader. The analysis included every document title, category, and description, then converted the text to lowercase and tokenized it with a basic alphabetic-token pattern. NLTK's English stop-word list was compared with the tokens found in the corpus, and each matching word was reviewed before making a decision.

The selected stop words are: `and`, `be`, `can`, `for`, `from`, `in`, `is`, `it`, `on`, `or`, `other`, `the`, `this`, and `with`. These are ordinary grammatical words in this corpus.

`m` and `s` are retained because they appear as clothing sizes. `t` is also retained because it can be part of the product term T-Shirt.

## Preprocessing

Preprocessing decisions made aside from normal procedure:

- Treat `t-shirt`, `t shirt`, and `tshirt` as the same clothing term: `tshirt`.
- Remove apostrophes, so words such as `men's` and `women's` become `mens` and `womens`.
- Keep alphabetic and numeric tokens because sizes and numeric product details may be useful for retrieval.

## Novelty Extension: Product Recommendations and Metadata Filtering

An optional backend module was added in `src/recommender.py` to extend the clothing search engine beyond the core requirements:

- Content-based recommendations: Given a target product, it calculates cosine similarity against other products using the existing inverted index and lnc document weights. It returns the top similar items with their titles, categories, and similarity scores, excluding the target item itself. Ties are broken by increasing document ID.
- Metadata filtering: Filters the corpus documents by attributes such as category before retrieval. It matches values case-insensitively and returns allowed document IDs so the interface or recommender can narrow down results when requested.

## Deliverables and Documentation Checklist

- [ ] Complete assignment documentation in `README.md`:
  - [x] Document and justify stop-word policy.
  - [x] Document preprocessing decisions.
  - [ ] Explain the inverted index and positional index structure.
  - [ ] Include `lnc.ltc` weighting and cosine normalization formulas.
  - [ ] Record results for mandatory test queries (10 free-text, 5 phrase, 3 proximity, 1 out-of-vocabulary).
  - [ ] Include comparative analysis of two cases where positional information changes retrieval results.
- [x] Export index files to `deliverables/index_output_files/`:
  - [x] Export dictionary / inverted index output (`inverted_index.json`).
  - [x] Export positional index output (`positional_index.json`).
- [ ] Capture application screenshots showing representative query results.
- [ ] Package final submission into a single ZIP file.
