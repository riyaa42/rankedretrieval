"""Streamlit web interface for the Clothing Information Retrieval System.

Supports:
1. Free-text search using lnc.ltc Vector Space Model cosine similarity.
2. Exact phrase search using token positions from the positional index.
3. Ordered proximity search using the WITHIN/k syntax.
4. Optional category filtering and content-based similar product recommendations.
"""

from pathlib import Path
import streamlit as st

from src.corpusloader import load_corpus
from src.indexer import build_inverted_index, build_positional_index
from src.recommender import find_similar_products, get_allowed_doc_ids
from src.search import parse_proximity_query, phrase_search, proximity_search, search


CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus_100.txt"


@st.cache_resource
def load_and_index_corpus():
    """Load corpus and build indexes once, caching them for fast interaction."""
    documents = load_corpus(CORPUS_PATH)
    inverted_index, document_lengths = build_inverted_index(documents)
    positional_index = build_positional_index(documents)
    categories = sorted(
        list({doc.get("category", "") for doc in documents if doc.get("category")})
    )
    return documents, inverted_index, document_lengths, positional_index, categories


def render_similar_products(
    docid, documents, inverted_index, document_lengths, allowed_ids
):
    """Render similar product recommendations inside an expander for a selected item."""
    with st.expander(f"Find similar products to {docid}"):
        recs = find_similar_products(
            target_doc_id=docid,
            documents=documents,
            inverted_index=inverted_index,
            document_lengths=document_lengths,
            top_k=5,
            allowed_doc_ids=allowed_ids,
        )

        if not recs:
            st.info("No similar products found.")
        else:
            st.markdown("**Top Similar Products (VSM Content-Based):**")
            for rec in recs:
                st.markdown(
                    f"- **[{rec['docid']}] {rec['title']}** : "
                    f"Category: `{rec['category']}` | Similarity Score: `{rec['score']:.4f}`"
                )


def main():
    st.set_page_config(
        page_title="Clothing IR Search Engine",
        layout="wide",
    )

    # Magenta / Darker Pink custom CSS styling
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #C2185B;
        }

        /* Radio button selected circle and focus */
        div[role="radiogroup"] label[data-baseweb="radio"] span[aria-checked="true"] {
            border-color: #C2185B !important;
            background-color: #C2185B !important;
        }
        div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div {
            border-color: #C2185B !important;
            background-color: #C2185B !important;
        }

        /* Primary Search button */
        div.stButton > button[kind="primary"], div.stButton > button[data-testid="baseButton-primary"] {
            background-color: #C2185B !important;
            border-color: #C2185B !important;
            color: #FFFFFF !important;
        }
        div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="baseButton-primary"]:hover {
            background-color: #AD1457 !important;
            border-color: #AD1457 !important;
            color: #FFFFFF !important;
        }

        /* Input box focus highlight */
        div[data-baseweb="input"] input:focus, div[data-baseweb="input"]:focus-within {
            border-color: #C2185B !important;
            box-shadow: 0 0 0 1px #C2185B !important;
        }

        /* Inline code snippets (suggested queries and tags) */
        code {
            color: #F06292 !important;
            background-color: rgba(194, 24, 91, 0.18) !important;
            border: 1px solid rgba(194, 24, 91, 0.35) !important;
            border-radius: 4px !important;
            padding: 2px 5px !important;
        }

        /* Matching positions and result notification boxes */
        div[data-testid="stAlert"] {
            background-color: rgba(194, 24, 91, 0.12) !important;
            border-left: 4px solid #C2185B !important;
        }
        div[data-testid="stAlert"] * {
            color: #FCE4EC !important;
        }

        /* Sidebar category alert border */
        div[data-testid="stSidebar"] div[data-testid="stAlert"] {
            border-left-color: #C2185B !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Clothing Information Retrieval System")

    # Load corpus and indexes (cached)
    documents, inverted_index, document_lengths, positional_index, categories = (
        load_and_index_corpus()
    )

    # ---------------- Sidebar Controls ----------------
    st.sidebar.header("Search Settings")

    search_mode = st.sidebar.radio(
        "Select Search Mode",
        options=["Free-text search", "Phrase search", "Proximity search"],
        help="Choose between VSM ranked retrieval, exact phrase matching, or ordered proximity searching.",
    )

    st.sidebar.divider()
    st.sidebar.header("Metadata Filter")

    category_options = ["All Categories"] + categories
    selected_category = st.sidebar.selectbox(
        "Filter by Category",
        options=category_options,
        index=0,
        help="Restrict search results to a specific product category.",
    )

    # Determine allowed document IDs based on metadata filter
    if selected_category != "All Categories":
        allowed_doc_ids = get_allowed_doc_ids(
            documents, filters={"category": selected_category}
        )
        st.sidebar.info(f"Filtering active: {len(allowed_doc_ids)} products in '{selected_category}'")
    else:
        allowed_doc_ids = None

    # ---------------- Mode 1: Free-text Search ----------------
    if search_mode == "Free-text search":
        st.subheader("Free-Text Search")
        st.write(
            "Enter free-text keywords to retrieve relevant clothing items ranked by cosine similarity."
        )

        query = st.text_input(
            "Search Query",
            placeholder="enter query here",
            key="free_text_query",
        )

        st.caption("Suggested queries: `cotton shirt`, `stretch denim`, `festive kurta`, `winter jacket`, `black t-shirt`")

        if st.button("Search", type="primary", key="btn_free_text"):
            if not query.strip():
                st.warning("Please enter a query to search.")
            else:
                raw_results = search(
                    query=query,
                    inverted_index=inverted_index,
                    document_lengths=document_lengths,
                    documents=documents,
                    top_k=10,
                )

                # Apply optional metadata filtering
                if allowed_doc_ids is not None:
                    results = [r for r in raw_results if r["docid"] in allowed_doc_ids]
                else:
                    results = raw_results

                st.markdown(f"### Results ({len(results)} documents found)")

                if not results:
                    st.info("No matching products found. Try different search terms.")
                else:
                    for rank, result in enumerate(results, start=1):
                        with st.container(border=True):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(
                                    f"#### #{rank} - [{result['docid']}] {result['title']}"
                                )
                                st.markdown(
                                    f"**Category:** `{result['category']}` | "
                                    f"**Cosine Similarity Score:** `{result['score']:.4f}`"
                                )
                            with col2:
                                st.metric("Cosine Score", f"{result['score']:.4f}")

                            # Show full description in expander
                            doc_meta = next(
                                (d for d in documents if d["docid"] == result["docid"]),
                                None,
                            )
                            if doc_meta and doc_meta.get("text"):
                                with st.expander("View product description"):
                                    st.write(doc_meta["text"])

                            # Optional novelty feature: similar products
                            render_similar_products(
                                result["docid"],
                                documents,
                                inverted_index,
                                document_lengths,
                                allowed_doc_ids,
                            )

    # ---------------- Mode 2: Phrase Search ----------------
    elif search_mode == "Phrase search":
        st.subheader("Exact Phrase Search")
        st.write(
            "Search for an exact sequence of words occurring consecutively in the product text."
        )

        phrase = st.text_input(
            "Exact Phrase",
            placeholder="enter query here",
            key="phrase_query",
        )

        st.caption("Suggested phrases: `cotton shirt`, `stretch denim`, `winter wear`, `regular fit`, `breathable fabric`")

        if st.button("Search Phrase", type="primary", key="btn_phrase"):
            if not phrase.strip():
                st.warning("Please enter a phrase to search.")
            else:
                raw_results = phrase_search(
                    phrase=phrase,
                    positional_index=positional_index,
                    documents=documents,
                )

                # Apply optional metadata filtering
                if allowed_doc_ids is not None:
                    results = [r for r in raw_results if r["docid"] in allowed_doc_ids]
                else:
                    results = raw_results

                st.markdown(f"### Results ({len(results)} documents found)")

                if not results:
                    st.info("No exact phrase matches found in the corpus.")
                else:
                    for rank, result in enumerate(results, start=1):
                        with st.container(border=True):
                            st.markdown(
                                f"#### #{rank} - [{result['docid']}] {result['title']}"
                            )
                            st.markdown(f"**Category:** `{result['category']}`")
                            st.success(
                                f"**Matching Phrase Start Positions (token index):** `{result['positions']}`"
                            )

                            doc_meta = next(
                                (d for d in documents if d["docid"] == result["docid"]),
                                None,
                            )
                            if doc_meta and doc_meta.get("text"):
                                with st.expander("View product description"):
                                    st.write(doc_meta["text"])

                            # Optional novelty feature: similar products
                            render_similar_products(
                                result["docid"],
                                documents,
                                inverted_index,
                                document_lengths,
                                allowed_doc_ids,
                            )

    # ---------------- Mode 3: Proximity Search ----------------
    elif search_mode == "Proximity search":
        st.subheader("Ordered Proximity Search")
        st.write(
            "Find documents where **Term 1** occurs before **Term 2** within at most **k** token positions."
        )

        proximity_query = st.text_input(
            "Proximity Query",
            placeholder="enter query here",
            key="prox_query",
        )

        st.caption("Format: `term1 WITHIN/k term2` (Suggested: `cotton WITHIN/3 shirt`, `stretch WITHIN/4 denim`, `winter WITHIN/3 wear`, `festive WITHIN/4 kurta`)")

        if st.button("Search Proximity", type="primary", key="btn_prox"):
            if not proximity_query.strip():
                st.warning("Please enter a proximity query.")
            else:
                parsed = parse_proximity_query(proximity_query)
                if not parsed:
                    st.error(
                        "Invalid proximity query format. Please use: term1 WITHIN/k term2 (e.g. cotton WITHIN/3 shirt)"
                    )
                else:
                    term1, term2, k_val = parsed
                    raw_results = proximity_search(
                        term1=term1,
                        term2=term2,
                        k=k_val,
                        positional_index=positional_index,
                        documents=documents,
                    )

                    # Apply optional metadata filtering
                    if allowed_doc_ids is not None:
                        results = [r for r in raw_results if r["docid"] in allowed_doc_ids]
                    else:
                        results = raw_results

                    # Group matching pairs by document for clear presentation
                    docs_with_matches = {}
                    for r in results:
                        docid = r["docid"]
                        if docid not in docs_with_matches:
                            docs_with_matches[docid] = {
                                "title": r.get("title", ""),
                                "category": r.get("category", ""),
                                "pairs": [],
                            }
                        docs_with_matches[docid]["pairs"].append(
                            (r["term1_position"], r["term2_position"], r["distance"])
                        )

                    st.markdown(f"### Results ({len(docs_with_matches)} documents found)")
                    if results:
                        st.caption(f"{len(results)} matching token pair occurrences across {len(docs_with_matches)} documents")

                    if not docs_with_matches:
                        st.info(
                            f"No occurrences found where '{term1}' appears before '{term2}' within {k_val} token positions."
                        )
                    else:
                        for rank, (docid, info) in enumerate(docs_with_matches.items(), start=1):
                            with st.container(border=True):
                                st.markdown(f"#### #{rank} - [{docid}] {info['title']}")
                                st.markdown(f"**Category:** `{info['category']}`")

                                pairs_str = ", ".join(
                                    f"(pos1={p[0]}, pos2={p[1]}, dist={p[2]})"
                                    for p in info["pairs"]
                                )
                                st.success(f"**Matching Token Pairs:** `{pairs_str}`")

                                doc_meta = next(
                                    (d for d in documents if d["docid"] == docid),
                                    None,
                                )
                                if doc_meta and doc_meta.get("text"):
                                    with st.expander("View product description"):
                                        st.write(doc_meta["text"])

                                # Optional novelty feature: similar products
                                render_similar_products(
                                    docid,
                                    documents,
                                    inverted_index,
                                    document_lengths,
                                    allowed_doc_ids,
                                )


if __name__ == "__main__":
    main()
