# Clothing Ranked Retrieval

## TODO: Clothing Information Retrieval Project

- [x] Implement corpus loader for `DOCID`, title/category, and description.

- [ ] Implement preprocessing:
  - [x] Convert text to lowercase.
  - [x] Tokenize text.
  - [x] Remove punctuation.
  - [x] Decide and document the stop-word policy.
  - [x] Apply stemming.

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

## Stop-word policy

The corpus was analysed automatically using the existing corpus loader. The analysis included every document title, category, and description, then converted the text to lowercase and tokenized it with a basic alphabetic-token pattern. NLTK's English stop-word list was compared with the tokens found in the corpus, and each matching word was reviewed before making a decision.

The selected stop words are: `and`, `be`, `can`, `for`, `from`, `in`, `is`, `it`, `on`, `or`, `other`, `the`, `this`, and `with`. These are ordinary grammatical words in this corpus.

`m` and `s` are retained because they appear as clothing sizes. `t` is also retained because it can be part of the product term T-Shirt.

## Preprocessing

Preprocessing decisions made aside from normal procedure:

- Treat `t-shirt`, `t shirt`, and `tshirt` as the same clothing term: `tshirt`.
- Remove apostrophes, so words such as `men's` and `women's` become `mens` and `womens`.
- Keep alphabetic and numeric tokens because sizes and numeric product details may be useful for retrieval.

## README.md info 

- [ ] Explain preprocessing decisions.
- [ ] Explain the inverted index and positional index.
- [ ] Include the `lnc.ltc` formula.
- [ ] Include sample queries and results.
