# src/corpusloader.py

import re


def load_corpus(file_path):
    """
    Reads the corpus file and extracts:
    DOCID, CATEGORY, TITLE, and TEXT
    """

    with open(file_path, "r", encoding="utf-8") as file:
        corpus = file.read()

    # Find every <DOC>...</DOC> block
    document_blocks = re.findall(
        r"<DOC>(.*?)</DOC>",
        corpus,
        flags=re.DOTALL
    )

    documents = []

    for block in document_blocks:
        docid = extract_tag(block, "DOCID")
        category = extract_tag(block, "CATEGORY")
        title = extract_tag(block, "TITLE")
        text = extract_tag(block, "TEXT")

        document = {
            "docid": docid,
            "category": category,
            "title": title,
            "text": text
        }

        documents.append(document)

    return documents


def extract_tag(text, tag_name):
    """
    Extracts the content inside a tag such as <DOCID>...</DOCID>.
    """

    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, flags=re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


if __name__ == "__main__":
    documents = load_corpus("data/corpus_100.txt")

    print("Number of documents:", len(documents))
    print("\nFirst document:")
    print(documents[0])