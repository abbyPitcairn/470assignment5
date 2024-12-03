from collections import defaultdict
import Calc
from QueryProcess import clean_and_tokenize
# Author: Abigail Pitcairn
# Version: Dec. 2, 2024


def build_inverted_index_tf_idf(docs):
    """
    Build an inverted index with tf-idf scores.

    Args:
        docs (list): List of documents where each document is a dictionary
                     with 'ID' and 'Text' keys.

    Returns:
        default dict: Inverted index mapping tokens to documents and their BM25 scores.
    """
    inverted_index = defaultdict(lambda: defaultdict(float))
    df = Calc.dfs(docs)
    for doc in docs:
        doc_id = doc['Id']
        unique_tokens = set(clean_and_tokenize(doc['Text']))
        for token in unique_tokens:
            inverted_index[token].update({doc_id: Calc.tf_idf(token, doc, df, docs)})
    return inverted_index


def build_inverted_index_bm25(docs):
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
            inverted_index[token].update({doc_id: Calc.bm25(token, doc, df, docs, avg_dl)})
    return inverted_index


# Build the inverted index
def build_inverted_index(docs):
    inverted_index = defaultdict(set)
    for doc in docs:
        doc_id = doc['Id']
        tokens = set(clean_and_tokenize(doc['Text']))
        for token in tokens:
            inverted_index[token].add(doc_id)
    return inverted_index