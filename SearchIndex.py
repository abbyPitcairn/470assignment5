from QueryProcess import query_process
# Author: Abigail Pitcairn
# Version: Dec. 2, 2024


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


def search_boolean_by_query(queries, inverted_index):
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
        result_ids = search_or(expanded_query_text, inverted_index)
        search_results[query_id] = result_ids
    return search_results


# Return the set() of document ID's for the documents returned from the inverted
# index for the query. The set is ranked by the document's score as determined in
# this method. The set is ordered from the highest score to the lowest score.
def search_index(query, inverted_index):
    result = {}
    for term in query:  # For each term in the query,
        if term in inverted_index:  # If the query term exists in the documents' inverted index,
            list_doc_ids = inverted_index[term]  # Get the list of doc_ids that include this term.
            for doc_id in list_doc_ids:  # For each doc_id:
                if doc_id not in result:
                    result[doc_id] = 0.0  # Initialize score as 0 if not present
                for word, score in inverted_index[doc_id].items():
                    result[doc_id] += score  # Add individual term scores
    # Return set with documents ranked highest to lowest score.
    return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}


# Return the set() of document ID's for the documents returned from the inverted
# index for the query. The set is ranked by the document's score as determined in
# this method. The set is ordered from the highest score to the lowest score.
def search_or(query, inverted_index):
    result = {}
    for term in query:  # For each term in the query,
        if term in inverted_index:  # If the query term exists in the documents' inverted index,
            list_doc_ids = inverted_index[term]  # Get the list of doc_ids that include this term.
            for doc_id in list_doc_ids:  # For each doc_id:
                if doc_id in result:  # If the doc_id has already been added to result,
                    result[doc_id] += 1  # Increase score by one for every occurrence of the term.
                else:  # Else, if the current doc_id is not yet in the result,
                    result[doc_id] = 1  # Add the doc_id to the results and set score to 1.
    # Return set with documents ranked highest to lowest score.
    return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
