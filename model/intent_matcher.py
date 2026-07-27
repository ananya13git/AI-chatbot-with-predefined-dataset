"""
SmartAssist Chatbot - ML-based Intent Matcher Module
---------------------------------------------------
This module implements TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization 
and Cosine Similarity to identify user intents. 

Rather than checking for direct word matches (which fail if the user misspells or uses synonyms), 
we convert raw text patterns into numerical vectors in a multi-dimensional semantic space, 
and find which predefined question vector is closest to the user's input vector.

Algorithms Explained:
1. TF-IDF (Term Frequency - Inverse Document Frequency):
   - Term Frequency (TF): Measures how frequently a term appears in a document.
     TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)
   - Inverse Document Frequency (IDF): Measures how important a term is across all documents.
     IDF(t, D) = log(Total number of documents D / Number of documents containing term t)
   - TF-IDF Weight = TF * IDF. High weight is assigned to terms that are common in a specific 
     document but rare across the entire dataset (e.g., words like "python" or "cybersecurity" 
     get high weights, whereas generic words get low weights).

2. Cosine Similarity:
   - Measures the cosine of the angle between two multi-dimensional vectors.
   - It determines how similar two documents are irrespective of their size.
   - Mathematically: Cosine_Similarity(A, B) = (A . B) / (||A|| * ||B||)
   - The resulting score ranges from 0 (completely orthogonal/dissimilar) to 1 (perfectly identical direction).

Author: Senior Python & NLP Engineer
"""

import os  # To handle file paths safely
import json  # To load the JSON intent dataset
import numpy as np  # For array manipulation and mathematical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to TF-IDF matrix
from sklearn.metrics.pairwise import cosine_similarity  # Calculates vector similarity
from model.preprocess import preprocess_text  # Our custom preprocessing function

class IntentMatcher:
    """
    Main class responsible for loading the intent dataset, converting patterns to TF-IDF space, 
    and matching incoming user input queries with the closest trained pattern.
    """
    
    def __init__(self, dataset_path):
        """
        Initializes the IntentMatcher by loading data and training the TF-IDF representation.
        
        Args:
            dataset_path (str): The absolute or relative path to the intents JSON file.
        """
        self.dataset_path = dataset_path
        
        # Initialize the TF-IDF Vectorizer with unigrams and bigrams (ngram_range=(1, 2))
        # sublinear_tf=True scales term frequency logarithmically for smoother similarity scores
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        
        # Lists to store raw intents and parallel mapping arrays
        self.intents_data = []  # Holds the raw intents loaded from JSON
        self.cleaned_patterns = []  # Holds preprocessed versions of all training patterns
        self.pattern_to_intent_map = []  # Maps each pattern index to its corresponding intent tag
        
        # Variables that will hold our TF-IDF model state
        self.tfidf_matrix = None  # Numerical matrix of our training patterns
        
        # Train the model upon class instantiation
        self.train()

    def train(self):
        """
        Loads the JSON dataset, preprocesses all patterns, fits the TF-IDF vectorizer, 
        and builds the comparison matrix.
        """
        # Ensure the file exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Intents file not found at: {self.dataset_path}")
            
        # 1. Load intents from JSON file
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            self.intents_data = data.get("intents", [])

        # 2. Iterate through each intent and extract all training patterns
        for intent in self.intents_data:
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                # Preprocess the pattern text (lowercase, tokenize, clean, lemmatize)
                cleaned = preprocess_text(pattern)
                
                # Append to our training corpus list
                self.cleaned_patterns.append(cleaned)
                
                # Maintain mapping of this training pattern to its intent tag
                self.pattern_to_intent_map.append(tag)

        # 3. Fit the Vectorizer and Transform the cleaned training patterns
        # This converts our list of preprocessed strings into a mathematical sparse matrix
        # where rows = patterns, and columns = unique vocabulary terms.
        if self.cleaned_patterns:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_patterns)
        else:
            raise ValueError("No training patterns were found in the dataset. Training aborted.")

    def match_intent(self, user_query):
        """
        Matches a user's query against the trained pattern vectors using Cosine Similarity.
        
        Args:
            user_query (str): The raw text message entered by the user.
            
        Returns:
            matched_tag (str): The tag of the intent with the highest cosine similarity.
            confidence_score (float): The cosine similarity value (between 0.0 and 1.0).
        """
        # 1. Preprocess the user's query using the exact same NLP pipeline
        cleaned_query = preprocess_text(user_query)
        
        # 2. Check if the preprocessed query is empty (e.g., user entered only punctuation or stopwords)
        if not cleaned_query.strip():
            # Return fallback tag with zero confidence
            return "unknown_intent", 0.0

        # 3. Vectorize the user's query
        # We use transform() instead of fit_transform() because we must use the already-fitted vocabulary.
        query_vector = self.vectorizer.transform([cleaned_query])
        
        # 4. Calculate Cosine Similarity between user query vector and all training pattern vectors
        # This returns an array of similarity scores, one for each pattern in the training set
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 5. Identify the index of the highest similarity score
        best_match_idx = np.argmax(similarity_scores)
        
        # 6. Retrieve the corresponding confidence score (similarity value)
        confidence_score = float(similarity_scores[best_match_idx])
        
        # 7. Retrieve the corresponding intent tag
        matched_tag = self.pattern_to_intent_map[best_match_idx]
        
        return matched_tag, confidence_score
