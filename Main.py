import Evaluation
import FileProcess
import InvertedIndex
import SearchIndex
import Rerank
import time
import sys
# Author: Abigail Pitcairn
# Version: Dec. 2, 2024

# Main function to execute the search for multiple queries and save results
def main(answers, topics1):
    """
    Executes the search pipeline, including BM25 search, reranking with BERT,
    and saving results to a .tsv file.

    Args:
        answers (str): Path to the JSON file containing the documents.
        topics1 (str): Path to the JSON file containing the queries.
    """
    # Load queries and documents
    documents = FileProcess.load_json_file(answers)
    queries1 = FileProcess.load_json_file(topics1)
    # Name result files
    result1_file_name = 'result_topics1.tsv'

    # Build inverted index
    build_start_time = time.time()
    print("Building inverted index...")
    index = InvertedIndex.build_inverted_index_bm25(documents)
    build_end_time = time.time()
    print(f"Build time: {build_end_time-build_start_time:.2f} seconds.")

    # Perform BM25 search
    print("Searching...")
    search_start = time.time()
    search_result1 = SearchIndex.search_index_by_query(queries1, index)
    search_end = time.time()
    print(f"Search completed in {search_end - search_start:.2f} seconds.")
    # Save and evaluate results
    FileProcess.save_to_result_file(search_result1, result1_file_name)
    Evaluation.evaluate_search_result('qrel_1.tsv', result1_file_name)

    # # Perform reranking with BERT
    # print("Reranking search results with BERT...")
    # rerank_start = time.time()
    # reranked_result1 = Rerank.rerank(search_result1, queries1, documents)
    # rerank_end = time.time()
    # print(f"Reranking completed in {rerank_end - rerank_start:.2f} seconds.")
    # # Save results to the output file
    # print(f"Saving reranked results to {result1_file_name}...")
    # FileProcess.save_to_result_file(reranked_result1, result1_file_name)
    # # Evaluate the reranked results
    # print("Evaluating reranked results...")
    # Evaluation.evaluate_search_result('qrel_1.tsv', result1_file_name)


# # Terminal Command: python Main.py Answers.json topics_1.json topics_2.json
# if __name__ == "__main__":
#     # Ensure two arguments are passed (answers.json and topics.json)
#     if len(sys.argv) != 4:
#         print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json")
#         sys.exit(1)
#
#     # Get file paths from command line arguments
#     answers_file = sys.argv[1]
#     topics1_file = sys.argv[2]
#     topics2_file = sys.argv[3]
#
#     # Call the main function with the file paths
#     main(answers_file, topics1_file, topics2_file)

# Manual run code (remember to comment out the above block)
main("Answers.json", "topics_1.json")
