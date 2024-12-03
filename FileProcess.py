import json
# Author: Abigail Pitcairn
# Version: Dec. 2, 2024


# Specify how many documents to be returned in the result file
TOTAL_RETURN_DOCUMENTS = 100


def save_to_result_file(results, output_file):
    """
    Save search results to an output file.

    Args:
        results (dict): The search results.
        output_file (str): The output file path.
    """
    with open(output_file, 'w') as f:
        for query_id, dic_result in results.items():
            for rank, (doc_id, score) in enumerate(dic_result.items(), start=1):
                f.write(f"{query_id} 0 {doc_id} {rank} {score:.6f} Run1\n")
                if rank >= TOTAL_RETURN_DOCUMENTS:  # Only return the top results
                    break


def load_json_file(file_path):
    """
    Load JSON file into a Python object.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of document dictionaries.
    """
    with open(file_path, 'r') as file:
        return json.load(file)