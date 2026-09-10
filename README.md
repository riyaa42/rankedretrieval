# Clothing Ranked Retrieval

## TODO: Clothing Information Retrieval Project

- [x] Implement corpus loader for `DOCID`, title/category, and description.

- [ ] Implement preprocessing:
  - [ ] Convert text to lowercase.
  - [ ] Tokenize text.
  - [ ] Remove punctuation.
  - [ ] Decide and document the stop-word policy.
  - [ ] Apply stemming.

- [ ] Build the inverted index:
  - [ ] Store term frequency.
  - [ ] Store document frequency.

- [ ] Store document lengths for cosine normalization.

- [ ] Implement `lnc.ltc` cosine-similarity ranking.

- [ ] Return the top 10 documents for each free-text query.

- [ ] Build the positional index with token positions.

- [ ] Implement exact phrase search.

- [ ] Implement ordered proximity search with different `k` values.

- [ ] Create a simple interface supporting:
  - [ ] Free-text search.
  - [ ] Phrase search.
  - [ ] Proximity search.
  - [ ] Displaying scores and matching positions.

- [ ] Test the system with:
  - [ ] 10 free-text queries.
  - [ ] 5 phrase queries.
  - [ ] 3 proximity queries with different `k` values.
  - [ ] 1 out-of-vocabulary query.

- [ ] Compare ordinary retrieval and positional retrieval using two examples.

## README.md info 

- [ ] Explain preprocessing decisions.
- [ ] Explain the inverted index and positional index.
- [ ] Include the `lnc.ltc` formula.
- [ ] Include sample queries and results.