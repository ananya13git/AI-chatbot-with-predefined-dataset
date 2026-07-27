"""
SmartAssist Chatbot - Response Engine Module
-------------------------------------------
This module loads the intents, takes the classification results from the ML matcher, 
and makes the final decision on what response to return to the user.

Key Concepts:
1. Confidence Thresholding: 
   - TF-IDF + Cosine Similarity will always find some matching pattern, even if the user 
     inputs total gibberish (e.g., "qwertyuiop"). It will simply map to the pattern with 
     the highest relative non-zero overlap.
   - To prevent the bot from outputting nonsense, we define a threshold (default: 0.30, or 30%).
   - If the similarity is below this value, we reject the prediction and return a fallback message.

2. Dynamic Random Response:
   - For each intent class, we have a list of responses. We use Python's random choice 
     module to pick one, ensuring the bot doesn't sound robotic by repeating the exact same sentence.

Author: Senior Python & NLP Engineer
"""

import random  # Python's built-in module for pseudo-random operations
from model.intent_matcher import IntentMatcher  # To use the intent matching system

class ResponseEngine:
    """
    Orchestrates response selection based on classification results and confidence thresholding.
    """
    
    def __init__(self, dataset_path, confidence_threshold=0.30):
        """
        Initializes the ResponseEngine.
        
        Args:
            dataset_path (str): Path to the intents.json dataset.
            confidence_threshold (float): Minimum cosine similarity required to accept a match.
        """
        # Instantiate the IntentMatcher
        self.matcher = IntentMatcher(dataset_path)
        
        # Set the threshold parameter
        self.confidence_threshold = confidence_threshold
        
        # Standard polite fallback message for unknown or low-confidence queries
        self.fallback_response = "I'm sorry, I couldn't understand your question. Could you please rephrase it?"

    def get_response(self, user_query):
        """
        Determines the appropriate response and metadata for a given user query.
        
        Args:
            user_query (str): The raw text entered by the user.
            
        Returns:
            result (dict): A dictionary containing the response text, confidence score, and matched tag.
        """
        # 1. Use the IntentMatcher to classify the query and calculate the confidence score
        matched_tag, confidence_score = self.matcher.match_intent(user_query)

        # 2. Check if confidence is below the required threshold
        if confidence_score < self.confidence_threshold:
            return {
                "response": self.fallback_response,
                "confidence": confidence_score,
                "intent": "unknown_intent"
            }

        # 3. If confidence is acceptable, locate the intent matching the tag in our data
        # We search through the loaded intents data inside the matcher
        for intent in self.matcher.intents_data:
            if intent["tag"] == matched_tag:
                # 4. Randomly select one response from the list of responses
                selected_response = random.choice(intent["responses"])
                
                # Return response, confidence score, and intent tag
                return {
                    "response": selected_response,
                    "confidence": round(confidence_score, 4),  # Round to 4 decimal places for readability
                    "intent": matched_tag
                }

        # 5. Fallback in case of code configuration error (should never occur if dataset is consistent)
        return {
            "response": self.fallback_response,
            "confidence": 0.0,
            "intent": "unknown_intent"
        }
