from collections import defaultdict
import Calc
from QueryProcess import clean_and_tokenize


# Author: Abigail Pitcairn
# Version: Dec. 2, 2024


def build_inverted_index(docs):
    """
    Build an inverted index with BM25 scores.

    Args:
        docs (list): List of documents where each document is a dictionary
                     with 'ID' and 'Text' keys.

    Returns:
        default dict: Inverted index mapping tokens to documents and their BM25 scores.
    """
    inverted_index = defaultdict(lambda: defaultdict(float))
    df = Calc.dfs(docs)
    avg_dl = Calc.avg_doc_len(docs)
    for doc in docs:
        doc_id = doc['Id']
        unique_tokens = set(clean_and_tokenize(doc['Text']))
        for token in unique_tokens:
            bm25_score = Calc.bm25(token, doc, df, docs, avg_dl)
            inverted_index[token][doc_id] = bm25_score
    return inverted_index