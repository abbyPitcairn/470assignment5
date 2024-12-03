import Evaluation
import InvertedIndex
import BM25Search
import QueryProcess
import Rerank
import time
import sys


# Main function to execute the search for multiple queries and save results
def main(answers, topics1):
    """
    Executes the search pipeline, including BM25 search, reranking with BERT,
    and saving results to a .tsv file.

    Args:
        answers (str): Path to the JSON file containing the documents.
        topics1 (str): Path to the JSON file containing the queries.
    """
    # Load the documents and build the inverted index
    documents = QueryProcess.load_json_file(answers)
    queries1 = QueryProcess.load_json_file(topics1)

    print("Building inverted index...")
    index = InvertedIndex.build_inverted_index(documents)

    # Output result file names
    result1_file_name = 'result_topics1.tsv'

    # Perform BM25 search
    print("Searching with BM25...")
    search_start = time.time()

    search_result1 = BM25Search.search_index_by_query(queries1, index)

    QueryProcess.save_to_result_file(search_result1, 'bm25_result.tsv')
    Evaluation.evaluate_search_result('qrel_1.tsv', 'bm25_result.tsv')
    search_end = time.time()
    print(f"BM25 Search completed in {search_end - search_start:.2f} seconds.")

    # Perform reranking with BERT
    print("Reranking search results with BERT...")
    rerank_start = time.time()
    reranked_result1 = Rerank.rerank(search_result1, queries1, documents)
    rerank_end = time.time()
    print(f"Reranking completed in {rerank_end - rerank_start:.2f} seconds.")

    # Save results to the output file
    print(f"Saving reranked results to {result1_file_name}...")
    QueryProcess.save_to_result_file(reranked_result1, result1_file_name)

    # Evaluate the reranked results
    print("Evaluating reranked results...")
    Evaluation.evaluate_search_result('qrel_1.tsv', result1_file_name)


# Terminal Command: python Main.py Answers.json topics_1.json topics_2.json
# if __name__ == "__main__":
    # # Ensure two arguments are passed (answers.json and topics.json)
    # if len(sys.argv) != 4:
    #     print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json")
    #     sys.exit(1)
    #
    # # Get file paths from command line arguments
    # answers_file = sys.argv[1]
    # topics1_file = sys.argv[2]
    # topics2_file = sys.argv[3]
    #
    # # Call the main function with the file paths
    # main(answers_file, topics1_file, topics2_file)

# Manual run code (remember to comment out the above block)
main("Answers.json", "topics_1.json")
