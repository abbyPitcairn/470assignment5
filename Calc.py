import math
from collections import Counter, defaultdict


# This program contains the calculations for TF-IDF and BM25
# Author: Abigail Pitcairn
# Version: 10.11.2024

# Calculate term frequency for a term-document pair
# Return how many times term appears in document
def tf(term, doc):
    term_counts = Counter(doc)
    return term_counts[term]


# Calculate document frequencies and store them in a dictionary for quick use
# Return a dictionary of document frequencies by term
def dfs(docs):
    df = defaultdict(int)
    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1
    return df


# Calculate the inverse document frequency for a term and collection
# Return the log of the number of documents in the collection divided by
# the number of documents in the collection containing the term
def idf(term, df, docs):
    n = len(docs) + 1
    return math.log(n / (df[term] + 1))


# Calculate term frequency-inverse document frequency
# Return the product of tf and idf for a term-document pair and a collection
def tf_idf(term, doc, df, docs):
    return tf(term, doc) * idf(term, df, docs)


# Calculate the average document length in a collection
# Return average document length
def avg_doc_len(docs):
    avg_dl = 0
    for doc in docs:
        avg_dl += len(doc)
    return avg_dl / len(docs)


# Calculate the BM25 value
# term - term
# doc - document
# docs - the collection of documents
# avg_dl - average doc length in the collection
# k - constant - 1.5
# b - constant - 0.75
def bm25(term, doc, df, docs, avg_dl, k1=1.5, b=0.75):
    idf1 = idf(term, df, docs)
    tf1 = tf(term, doc)
    doc_len = len(doc)
    return idf1 * ((tf1 * (k1 + 1)) / (tf1 + k1 * (1 - b + b * (doc_len / avg_dl))))