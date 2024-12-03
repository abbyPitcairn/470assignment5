import math
from collections import Counter, defaultdict
# This program contains the calculations for TF-IDF and BM25
# Author: Abigail Pitcairn
# Version: 10.11.2024

def tf(term, doc):
    """
    Calculate term frequency (TF) for a given term in a document.

    Args:
        term (str): The term to calculate frequency for.
        doc (list): A list of terms representing the document.

    Returns:
        int: Frequency of the term in the document.
    """
    term_counts = Counter(doc)  # Count occurrences of each term in the document.
    return term_counts[term]  # Return the count of the specified term.


def dfs(docs):
    """
    Calculate document frequencies (DF) for all terms in a collection of documents.

    Args:
        docs (list of list): A collection of documents, where each document
                             is represented as a list of terms.

    Returns:
        defaultdict: A dictionary mapping terms to the number of documents
                     they appear in.
    """
    df = defaultdict(int)
    for doc in docs:
        unique_terms = set(doc)  # Extract unique terms from the document.
        for term in unique_terms:
            df[term] += 1  # Increment the document frequency for each term.
    return df


def idf(term, df, docs):
    """
    Calculate inverse document frequency (IDF) for a given term.

    Args:
        term (str): The term to calculate IDF for.
        df (dict): Precomputed document frequency dictionary mapping terms to their DF.
        docs (list of list): Collection of documents, where each document is a list of terms.

    Returns:
        float: IDF value for the term, computed as log((N + 1) / (DF + 1)) to avoid divide-by-zero.
    """
    n = len(docs) + 1  # Total number of documents, with smoothing.
    return math.log(n / (df[term] + 1))  # Apply smoothing to DF as well.


def tf_idf(term, doc, df, docs):
    """
    Calculate TF-IDF score for a term in a document.

    Args:
        term (str): The term to calculate TF-IDF for.
        doc (list): The document represented as a list of terms.
        df (dict): Precomputed document frequency dictionary.
        docs (list of list): Collection of documents.

    Returns:
        float: TF-IDF score for the term in the document.
    """
    return tf(term, doc) * idf(term, df, docs)  # Product of TF and IDF.


def avg_doc_len(docs):
    """
    Calculate the average document length in a collection of documents.

    Args:
        docs (list of list): A collection of documents, where each document
                             is represented as a list of terms.

    Returns:
        float: The average length of documents in the collection.
    """
    avg_dl = 0
    for doc in docs:
        avg_dl += len(doc)  # Accumulate lengths of all documents.
    return avg_dl / len(docs)  # Divide by the total number of documents.


def bm25(term, doc, df, docs, avg_dl, k1=1.5, b=0.75):
    """
    Calculate the BM25 score for a term in a document.

    Args:
        term (str): The term to calculate BM25 for.
        doc (list): The document represented as a list of terms.
        df (dict): Precomputed document frequency dictionary mapping terms to DF.
        docs (list of list): Collection of documents.
        avg_dl (float): Average document length in the collection.
        k1 (float): BM25 term frequency saturation parameter. Defaults to 1.5.
        b (float): BM25 length normalization parameter. Defaults to 0.75.

    Returns:
        float: BM25 score for the term in the document.
    """
    idf1 = idf(term, df, docs)  # Calculate IDF for the term.
    tf1 = tf(term, doc)  # Calculate term frequency.
    doc_len = len(doc)  # Get the document length.
    normalization = 1 - b + b * (doc_len / avg_dl)  # Length normalization factor.
    return idf1 * ((tf1 * (k1 + 1)) / (tf1 + k1 * normalization))  # BM25 formula.
