# Information Retrieval System
This is an assignment for COS470: Intro to Information Retrieval at the University of Southern Maine. It implements a variety of techniques from the course to create an effective information retrieval system.

@Author: Abigail Pitcairn

@Version: December 4, 2024

#### Process:
- Document file is used to build an inverted index using BM25 scores.
- Query text is cleaned and expanded with WordNet.
- Top 100 documents are retrieved per query using the inverted index and preprocessed queries.
- Top 100 results for each query are reranked using BERT model based on cosine similarity scores.
- Final reranked results and saved and evaluated.  


