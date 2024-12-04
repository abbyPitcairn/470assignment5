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
        if expansion:
            tuple: list of cleaned and expanded terms and int query id
        else:
            tuple: str query and int query id
    """
    query_id = query['Id']
    query_text = f"{query['Title']} {query['Body']}"
    if expansion:
        terms = expand_query(list(query_text))
        return terms, query_id
    return query_text.lower(), query_id


def expand_query(query, max_synonyms=2):
    """
    Expand the query using WordNet synonyms.

    Args:
        query (list): The list of terms from input query.
        max_synonyms (int): Maximum number of synonyms to add to query.

    Returns:
        list: The expanded query terms.
    """
    expanded_terms = set()
    for term in query:
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
    return set(expanded_terms)


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
    return {word for word in tokens if word not in STOP_WORDS}
