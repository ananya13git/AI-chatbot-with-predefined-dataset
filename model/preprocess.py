"""
SmartAssist Chatbot - NLP Preprocessing Module
-----------------------------------------------
This file implements the text preprocessing pipeline using the Natural Language Toolkit (NLTK).
Preprocessing is the foundation of Natural Language Processing (NLP) because raw human language 
is noisy, containing grammatical variances, punctuation, and non-essential words.

The pipeline executes the following sequential steps:
1. Lowercasing: Normalizes all text to lowercase to prevent casing mismatches.
2. Tokenization: Splits continuous strings of text into individual words or tokens.
3. Punctuation Removal: Removes non-alphanumeric characters that don't add semantic value.
4. Stopword Removal: Filters out high-frequency functional words (e.g., 'is', 'the', 'and').
5. Lemmatization: Reduces words to their base dictionary form (lemma) based on their Part-of-Speech (POS).

Author: Senior Python & NLP Engineer
"""

import string  # Standard library containing punctuation character sets
import nltk  # Natural Language Toolkit for professional NLP operations

# Import specific modules from NLTK required for our pipeline
from nltk.tokenize import word_tokenize  # For dividing text into list of tokens
from nltk.corpus import stopwords  # For fetching standard list of english stopwords
from nltk.stem import WordNetLemmatizer  # For reducing words to their root/base form
from nltk.corpus import wordnet  # For mapping detailed POS tags to WordNet categories

# ---------------------------------------------------------
# Dynamic NLTK Resource Downloader
# ---------------------------------------------------------
# Production systems must automatically acquire missing lexical resources 
# to run seamlessly on a new environment.
def _initialize_resources():
    """
    Checks if necessary NLTK datasets are downloaded. If not, downloads them silently.
    This prevents missing resource exceptions at runtime.
    """
    required_packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    
    for pkg in required_packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            pass

# Run the initialization on module import
_initialize_resources()

# Initialize the lemmatizer object (used in lemmatize_text function)
lemmatizer = WordNetLemmatizer()

# Fetch the set of English stopwords (words like 'and', 'the', 'a', etc. that we want to filter)
# Using a set is O(1) for lookup time, making it faster than a list.
stop_words = set(stopwords.words('english'))

# ---------------------------------------------------------
# POS Tagging Helper Function
# ---------------------------------------------------------
def get_wordnet_pos(word):
    """
    Maps a word's Penn Treebank Part-of-Speech (POS) tag to its WordNet equivalent.
    Without POS context, a lemmatizer cannot differentiate verbs from nouns (e.g., 'leaves').
    
    Args:
        word (str): The word to analyze.
        
    Returns:
        wordnet_pos (constant): A WordNet POS constant (noun, verb, adjective, or adverb).
    """
    # Use NLTK's pos_tag function to determine the part of speech
    # pos_tag expects a list of words, so we wrap the single word in a list and take the first tag
    tag = nltk.pos_tag([word])[0][1][0].upper()
    
    # Map the first letter of the Penn Treebank tag to the corresponding WordNet categories
    tag_dict = {
        'J': wordnet.ADJ,   # J represents Adjectives (e.g., big, fast)
        'V': wordnet.VERB,  # V represents Verbs (e.g., running, code)
        'N': wordnet.NOUN,  # N represents Nouns (e.g., computer, chatbot)
        'R': wordnet.ADV   # R represents Adverbs (e.g., quickly, nicely)
    }
    
    # Return the mapped category, defaulting to NOUN if the tag is not in the dictionary
    return tag_dict.get(tag, wordnet.NOUN)


# ---------------------------------------------------------
# Core Preprocessing Function
# ---------------------------------------------------------
def preprocess_text(text):
    """
    Runs the complete preprocessing pipeline on a string of raw text.
    
    Args:
        text (str): The raw string input from the user or dataset patterns.
        
    Returns:
        cleaned_text (str): A space-separated string of normalized, lemmatized tokens.
    """
    # 1. Error Handling: Ensure input is a string. If null or empty, return empty string.
    if not isinstance(text, str) or not text.strip():
        return ""

    # 2. Lowercasing: Convert the text to lowercase to ensure consistency (e.g., 'Python' == 'python')
    lowered_text = text.lower()
    
    # 3. Tokenization: Split the continuous text into individual words (tokens)
    # E.g., "Hello, bot!" becomes ["hello", ",", "bot", "!"]
    tokens = word_tokenize(lowered_text)
    
    # 4. Punctuation Removal: Remove tokens that are purely punctuation marks
    # We check if each character in the token is in the string.punctuation set
    no_punctuation_tokens = [
        token for token in tokens 
        if token not in string.punctuation and not all(char in string.punctuation for char in token)
    ]
    
    # 5. Stopword Removal: Filter out high-frequency grammatical words
    filtered_tokens = [
        token for token in no_punctuation_tokens 
        if token not in stop_words
    ]
    
    # Smart Fallback: If stopword removal wipes out all words (e.g. "who are you"),
    # retain the non-punctuation tokens so short conversational queries aren't emptied.
    tokens_to_lemmatize = filtered_tokens if filtered_tokens else no_punctuation_tokens
    
    # 6. Lemmatization with POS Tagging: Reduce words to their dictionary root form
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(token)) 
        for token in tokens_to_lemmatize
    ]
    
    # 7. Reconstruction: Re-assemble the processed tokens back into a single clean string
    return " ".join(lemmatized_tokens)
