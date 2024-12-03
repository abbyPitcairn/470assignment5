import re
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
# Author: Abigail Pitcairn
# Version: Dec. 2, 2024


# Define stop words to be removed from index
STOP_WORDS = {"the", "of", "to", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with",
              "his", "they", "I", "at", "be", "this", "have", "from", "or", "one", "by", "but", "not", "what", "all",
              "we", "when", "your", "can", "there", "use", "each", "which", "do", "how", "their", "if", "will", "up",
              "other", "about", "out", "many", "then", "them", "these", "so", "some", "would", "into", "more", "no",
              "way", "could", "my", "than", "first", "been", "now", "long", "down", "day", "may", "over", "new", "take",
              "only", "little", "know", "place", "very", "after", "our", "just", "most", "before", "too", "any", "same",
              "tell", "such", "because", "why", "also", "well", "must", "even", "here", "off", "again", "still",
              "should"}


def query_process(query, expansion):
    """
    Process a single query by extracting its ID and expanding its text.

    Args:
        query (dict): A dictionary containing query data.
        expansion (bool): True if query is to be expanded.

    Returns:
        tuple: The expanded query text and the query ID.
    """
    query_id = query['Id']
    query_text = f"{query['Title']} {query['Body']}"
    if expansion:
        query_text = expand_query(query_text)
    return query_text, query_id


def expand_query(query, max_synonyms=3):
    """
    Expand the query using WordNet synonyms.

    Args:
        query (str): The input query.
        max_synonyms (int): Maximum number of synonyms to add to query.

    Returns:
        str: The expanded query string.
    """
    expanded_terms = set()
    terms = clean_and_tokenize(query)
    for term in terms:
        expanded_terms.add(term)  # Include the original term
        # Add up to `max_synonyms` synonyms for the term
        synonyms_added = 0
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                if lemma.name().lower() != term:  # Avoid adding the original term again
                    expanded_terms.add(lemma.name().lower())
                    synonyms_added += 1
                    if synonyms_added >= max_synonyms:
                        break
            if synonyms_added >= max_synonyms:
                break
    return ' '.join(expanded_terms)  # Return the expanded query as a single string


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
