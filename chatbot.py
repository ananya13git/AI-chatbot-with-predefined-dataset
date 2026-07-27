"""
SmartAssist Chatbot - Central Engine Orchestrator
------------------------------------------------
This script coordinates our NLP preprocessing module, TF-IDF matcher, and response engine.
It instantiates the ResponseEngine as a global singleton. Doing so ensures that the training dataset 
is loaded and the TF-IDF matrix is computed exactly ONCE during server startup, rather than 
repeatedly on every incoming chat request, which would cause significant latency.

It exposes a single, clean API function: get_bot_response(message).

Author: Senior Python & NLP Engineer
"""

import os  # To fetch path locations dynamically
from model.response_engine import ResponseEngine  # Import the final decision-making component

# 1. Establish the absolute path to the intents database.
# Using os.path.dirname(os.path.abspath(__file__)) ensures that the script correctly locates
# the intents.json file regardless of the working directory from which the application is launched.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

# 2. Instantiate the response engine globally.
# We set a confidence threshold of 0.30. If the similarity between user input and any known 
# pattern is below 30%, the engine will fall back to a polite clarification request.
bot_engine = ResponseEngine(dataset_path=INTENTS_PATH, confidence_threshold=0.30)

# 3. Expose the unified chat interface function
def get_bot_response(message):
    """
    Accepts a raw user message string and returns the chatbot's predicted response,
    the intent tag, and the matching confidence score.
    
    Args:
        message (str): The raw text message typed by the user.
        
    Returns:
        response_data (dict): A dictionary matching the following structure:
            {
                "response": "selected response string...",
                "confidence": 0.8542,
                "intent": "matched_intent_tag"
            }
    """
    # Simply delegate the text to our configured engine and return the resulting dictionary.
    return bot_engine.get_response(message)
