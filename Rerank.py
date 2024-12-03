import os
import torch
from sentence_transformers import SentenceTransformer, util
from QueryProcess import query_process

# Author: Abigail Pitcairn
# Version: Dec. 2, 2024

# Load BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WAND_DISABLED"] = "true"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


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
    print(f"In Rerank.rerank, the 'queries' passed are type {type(queries)}")
    # Build a dictionary of documents for quick lookup
    docs_dict = {doc['Id']: doc['Text'] for doc in docs}

    # Initialize a dictionary for reranked results
    reranked_results = {}

    # Iterate over each query
    for query_data in queries:
        expanded_query_text, query_id = query_process(query_data)  # Expand the query

        # Encode the query using the BERT model
        q_embedding = model.encode(expanded_query_text, convert_to_tensor=True)

        # Prepare a list to hold document similarity scores
        doc_scores = []

        # Iterate over the top search results for this query
        for doc_id in search_results.get(query_id, []):
            if doc_id in docs_dict:
                doc_text = docs_dict[doc_id]
                # Encode the document text
                d_embedding = model.encode(doc_text, convert_to_tensor=True)
                # Calculate cosine similarity
                similarity_score = util.pytorch_cos_sim(q_embedding, d_embedding).item()
                # Append the document ID and its similarity score
                doc_scores.append((doc_id, similarity_score))

        # Sort the documents by similarity score in descending order
        reranked_results[query_id] = [doc_id for doc_id, _ in sorted(doc_scores, key=lambda x: x[1], reverse=True)]

    return reranked_results
