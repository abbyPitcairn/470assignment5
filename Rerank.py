import os
import torch
from sentence_transformers import SentenceTransformer
import QueryProcess
# Author: Abigail Pitcairn
# Version: Dec. 2, 2024

# Load BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WAND_DISABLED"] = "true"
model = SentenceTransformer('all-MiniLM-L6-v2')


def rerank(search_results, queries, docs):
    """
    Rerank search results using cosine similarity calculated by BERT embeddings.

    Args:
        search_results (dict): Search results with query IDs mapping to initial ranked document IDs.
        queries (list): Data already loaded from the JSON file containing query data.
        docs (list): List of documents, each represented as a dictionary with 'ID' and 'Text' fields.

    Returns:
        dict: Reranked results with query IDs mapping to reranked document IDs.
    """
    # Build a dictionary of documents for quick lookup
    doc_embeddings = get_doc_embedding_dict(docs)

    # Initialize a dictionary for reranked results
    reranked_results = {}

    # Iterate over each query
    for query_data in queries:
        # Encode the query using the BERT model
        query_text, query_id = QueryProcess.query_process(query_data, False)
        q_embedding = model.encode(query_text, convert_to_tensor=True)

        # Prepare a list to hold document similarity scores
        doc_scores = []

        # Iterate over the top search results for this query
        for doc_id in search_results.get(query_id, []):
            if doc_id in doc_embeddings:
                d_embedding = doc_embeddings[doc_id]
                # Calculate cosine similarity
                similarity_score = model.cosine_similarity(q_embedding, d_embedding)
                # Append the document ID and its similarity score
                doc_scores.append((doc_id, similarity_score))

    # Sort the documents by similarity score in descending order
    return {k: v for k, v in sorted(reranked_results.items(), key=lambda item: item[1], reverse=True)}


def get_doc_embedding_dict(docs):
    doc_embeddings = {}
    for doc in docs:
        doc_id = doc['Id']
        doc_embeddings[doc_id] = model.encode(doc['Text'])
    return doc_embeddings
