import InvertedIndex
from QueryProcess import query_process


# Author: Abigail Pitcairn
# Version: Dec. 2, 2024


def search_index(query, inverted_index):
    """
    Perform BM25 search for the given query on the inverted index.

    Args:
        query (str): The input query.
        inverted_index (dict): The inverted index containing document data.

    Returns:
        dict: A dictionary of document IDs and their scores, sorted in descending order of scores.
    """
    result = {}
    terms = InvertedIndex.clean_and_tokenize(query)  # Tokenize the query
    for term in terms:
        if term in inverted_index:  # If the term exists in the index
            for doc_id, score in inverted_index[term].items():
                result[doc_id] = result.get(doc_id, 0.0) + score  # Increment the score for the document
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))  # Sort by score


def search_index_by_query(queries, inverted_index):
    """
    Process multiple queries and perform search for each query.

    Args:
        queries (list): Data already loaded from the JSON file containing query data.
        inverted_index (dict): The inverted index containing document data.

    Returns:
        dict: A dictionary of query IDs and their corresponding ranked results.
    """
    search_results = {}
    for query_data in queries:
        expanded_query_text, query_id = query_process(query_data)
        result_ids = search_index(expanded_query_text, inverted_index)
        search_results[query_id] = result_ids
    return search_results
