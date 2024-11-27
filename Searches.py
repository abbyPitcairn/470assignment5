import InvertedIndex

# This program contains methods for the Boolean information retrival system
# search function.
# This program contains the TF-IDF / BM25 search method.
# This program also contains the "save_to_result_file" method which formats and
# stores the results from the searches into a file with the name specified
# Author: Abigail Pitcairn
# Version: 10.11.2024

# Specify how many documents you would like returned in the result file
total_return_documents = 100


# Return the set() of document ID's for the documents returned from the inverted
# index for the query. The set is ranked by the document's score as determined in
# this method. The set is ordered from the highest score to the lowest score.
def search(query, inverted_index):
    result = {}
    terms = InvertedIndex.clean_and_tokenize(query)  # Tokenize all terms from current query.
    for term in terms:  # For each term in the query,
        if term in inverted_index:  # If the query term exists in the documents' inverted index,
            list_doc_ids = inverted_index[term]  # Get the list of doc_ids that include this term.
            for doc_id in list_doc_ids:  # For each doc_id:
                if doc_id not in result:
                    result[doc_id] = 0.0  # Initialize score as 0 if not present
                for word, score in inverted_index[doc_id].items():
                    result[doc_id] += score  # Add individual term scores
    # Return set with documents ranked highest to lowest score.
    return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}


# Write the input results to an output file with columns for:
# query_id, 0, doc_id, rank, result for doc_id, name of run
def save_to_result_file(results, output_file):
    with open(output_file, 'w') as f:
        for query_id in results:
            dic_result = results[query_id]
            rank = 1  # Initialize rank to 1
            for doc_id in dic_result:
                f.write(f"{query_id} 0 {doc_id} {rank} {dic_result[doc_id]} Run1\n")
                rank += 1  # Increment the rank for each document returned
                if rank > total_return_documents:  # Only return the top 100 results from search
                    break


# Load the queries from the topics file and perform search
# Returns search results from input queries over input collection
def query_load(topics, inverted_index):
    # Load the queries JSON file
    queries_file_path = topics  # Path to the search queries file
    queries = InvertedIndex.load_json_file(queries_file_path)

    # Perform search for each query and store results
    search_results = {}  # Initialize results variable

    # Get the terms and perform search for each query
    for query_data in queries:
        query_id = query_data['Id']
        title = query_data['Title']
        body = query_data['Body']
        query_text = title + " " + body

        # Perform the search
        result_ids = search(query_text, inverted_index)
        search_results[query_id] = result_ids  # Store the result

    return search_results
