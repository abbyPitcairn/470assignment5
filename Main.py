import InvertedIndex
import Searches
import time
import sys

# This program runs the main method for the COS470Assignment1 project.
# When run, the program creates the inverted index from the input file path
# "answers" and then runs all queries from the input file path "topics" over it.
# Results from the queries are formatted and stored in a .tsv file.
# Author: Abigail Pitcairn
# Version: 10.11.2024


# Main function to execute the search for multiple queries and save results
def main(answers, topics):

    # Load the documents and build the inverted index
    documents_file_path = answers  # Path to the database documents file
    documents = InvertedIndex.load_json_file(documents_file_path)
    print("Building inverted indexes...")
    InvertedIndex.build_inverted_indexes(documents)

    search_start = time.time()
    print("Searching...")
    search_results_tfidf = Searches.query_load(topics, InvertedIndex.inverted_index_tfidf)
    end_search_tfidf = time.time()
    search_results_bm25 = Searches.query_load(topics, InvertedIndex.inverted_index_bm25)
    end_search_bm25 = time.time()

    # Save search results to file
    tfidf_output_result_file = 'result_tfidf.tsv'
    bm25_output_result_file = 'result_bm25.tsv'
    Searches.save_to_result_file(search_results_tfidf, tfidf_output_result_file)
    Searches.save_to_result_file(search_results_bm25, bm25_output_result_file)
    print(f"Search results saved to {tfidf_output_result_file} and {bm25_output_result_file}")

    # Print execution times
    print(f"TF_IDF Search time: {end_search_tfidf - search_start} seconds.")
    print(f"BM25 Search time: {end_search_bm25 - search_start} seconds.")


# Terminal Command: python Main.py Answers.json topics_1.json
# OR python Main.py Answers.json topics_2.json
# if __name__ == "__main__":
    # # Ensure two arguments are passed (answers.json and topics.json)
    # if len(sys.argv) != 3:
    #     print("Usage: python main.py <answers.json> <topics.json>")
    #     sys.exit(1)
    #
    # # Get file paths from command line arguments
    # answers_file = sys.argv[1]
    # topics_file = sys.argv[2]
    #
    # # Call the main function with the file paths
    # main(answers_file, topics_file)

# Manual run code (remember to comment out the above block)
main("Answers.json", "topics_1.json")
