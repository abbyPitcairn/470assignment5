import json
import re
from bs4 import BeautifulSoup
from collections import defaultdict
import Calc

# This program builds the inverted indexes using an input json file.
# The index can be generated using either TF-IDF or BM25 values.
#
# Author: Abigail Pitcairn
# Version: 10.11.2024

# Define stop words to be removed from index
# These are the top 50 most commonly occurring words according to Zipf's law
stop_words = ["the", "of", "and", "to", "a", "in", "is", "that", "was", "it", "for", "on", "with", "he", "be",
              "I", "by", "as", "at", "you", "are", "his", "had", "not", "this", "have", "fom", "but", "which", "she",
              "they", "or", "an", "her", "were", "there", "we", "their", "been", "has", "will", "one", "all",
              "would", "can", "if", "who", "more", "when", "said", "do", "what", "about", "its", "it's", "so", "up",
              "into", "no", "him", "some", "could", "them", "only", "time", "out", "my", "two", "other", "then", "may",
              "over", "also", "new", "like", "these", "me", "after", "first", "your", "did", "now", "any", "people",
              "than", "should", "very", "most", "see", "where", "just", "made", "between", "back", "way", "many",
              "years", "being", "our", "how", "work"]


# Load the JSON file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Clean and tokenize the text and remove stop words
def clean_and_tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    tokens = re.findall(r'\b\w+\b', clean_text.lower())
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    return filtered_words


inverted_index_tfidf = defaultdict(lambda: defaultdict(float))
inverted_index_bm25 = defaultdict(lambda: defaultdict(float))


# Build the inverted index with tf-idf values
def build_inverted_indexes(docs):
    df = Calc.dfs(docs)
    avg_dl = Calc.avg_doc_len(docs)
    for doc in docs:
        doc_id = doc['Id']
        tokens = clean_and_tokenize(doc['Text'])
        for token in set(tokens):
            inverted_index_tfidf[token].update({doc_id: Calc.tf_idf(token, doc, df, docs)})
            inverted_index_bm25[token].update({doc_id: Calc.bm25(token, doc, df, docs, avg_dl)})
