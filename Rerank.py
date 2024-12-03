import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel
import numpy as np
import QueryProcess
# Author: Abigail Pitcairn
# Version: Dec. 2, 2024

# Load BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WAND_DISABLED"] = "true"
basic_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = torch.quantization.quantize_dynamic(
    basic_model, {torch.nn.Linear}, dtype=torch.qint8)


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
    docs_dict = get_docs_dict(docs)

    # Initialize a dictionary for reranked results
    reranked_results = {}

    # Iterate over each query
    for query_data in queries:
        # Encode the query using the BERT model
        query_id = query_data['Id']
        query_text = QueryProcess.clean_and_tokenize(f"{query_data['Title']} {query_data['Body']}")
        q_embedding = model.encode(query_text, batch_size=32, convert_to_tensor=True)

        # Prepare a list to hold document similarity scores
        doc_scores = []

        # Iterate over the top search results for this query
        for doc_id in search_results.get(query_id, []):
            if doc_id in docs_dict:
                d_embedding = docs_dict[doc_id]
                # Calculate cosine similarity
                similarity_score = cosine_similarity(q_embedding, d_embedding)
                # Append the document ID and its similarity score
                doc_scores.append((doc_id, similarity_score))

        # Sort the documents by similarity score in descending order
        reranked_results[query_id] = [doc_id for doc_id, _ in sorted(doc_scores, key=lambda x: x[1], reverse=True)]

    return reranked_results


def get_docs_dict(docs):
    # Convert to NumPy array for efficient processing
    doc_embeddings_array = np.array([model.encode(doc, batch_size=32, convert_to_tensor=True) for doc in docs])
    return doc_embeddings_array
