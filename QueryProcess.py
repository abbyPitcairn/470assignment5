import json
import re
from bs4 import BeautifulSoup
from nltk.corpus import wordnet

# Specify how many documents you would like returned in the result file
TOTAL_RETURN_DOCUMENTS = 100

# Define stop words to be removed from index
STOP_WORDS = {"the", "of", "to", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with",
              "his", "they", "I", "at", "be", "this", "have", "from", "or", "one", "by", "but", "not", "what", "all",
              "we", "when", "your", "can", "there", "use", "each", "which", "do", "how", "their", "if", "will", "up",
              "other", "about", "out", "many", "then", "them", "these", "so", "some", "would", "into", "more", "no",
              "way", "could", "my", "than", "first", "been", "now", "long", "down", "day", "may", "over", "new", "take",
              "only", "little", "know", "place", "very", "after", "our", "just", "most", "before", "too", "any", "same",
              "tell", "such", "because", "why", "also", "well", "must", "even", "here", "off", "again", "still",
              "should"}


def clean_and_tokenize(text):
    """
    Clean HTML, tokenize text, and remove stop words.

    Args:
        text (str): Raw document text.

    Returns:
        list: List of filtered tokens.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected string for text, but got {type(text)}: {text}")
    clean_text = BeautifulSoup(text, "html.parser").get_text()
    tokens = re.findall(r'\b\w+\b', clean_text.lower())
    return [word for word in tokens if word not in STOP_WORDS]


def expand_query(query):
    """
    Expand the query using WordNet synonyms.

    Args:
        query (str): The input query.

    Returns:
        str: The expanded query string.
    """
    expanded_terms = set()
    terms = clean_and_tokenize(query)
    for term in terms:
        expanded_terms.add(term)  # Include the original term
        for syn in wordnet.synsets(term):
            expanded_terms.update(lemma.name().lower() for lemma in syn.lemmas())  # Add synonyms
    return ' '.join(expanded_terms)  # Return the expanded query as a single string


def query_process(query):
    """
    Process a single query by extracting its ID and expanding its text.

    Args:
        query (dict): A dictionary containing query data.

    Returns:
        tuple: The expanded query text and the query ID.
    """
    query_id = query['Id']
    query_text = f"{query['Title']} {query['Body']}"
    expanded_query_text = expand_query(query_text)
    return expanded_query_text, query_id


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