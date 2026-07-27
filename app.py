"""
SmartAssist Chatbot - Flask Web Application Backend
--------------------------------------------------
This script initializes and runs the Flask server, which functions as our HTTP backend.
It serves the frontend interface and provides a JSON API endpoint for chat interactions.

API Design & Security Checklist:
1. Root Route ('/'): GET method, renders the main HTML interface using Flask's render_template.
2. Chat Route ('/chat'): POST method, accepts application/json body and returns a JSON response.
3. Content-Type Validation: Verifies that requests to /chat specify application/json.
4. Input Sanitation & Validation: Rejects empty strings, whitespace-only messages, and malformed JSON.
5. Error Boundary Handler: Implements try/catch blocks to intercept runtime exceptions, 
   returning appropriate HTTP status codes (400 for bad request data, 500 for server errors) 
   instead of causing the application to crash or leaking system stack traces.

Author: Senior Python & NLP Engineer
"""

import logging  # Standard library for generating runtime diagnostic logs
from flask import Flask, render_template, request, jsonify  # Core web-server capabilities
from chatbot import get_bot_response  # Our chatbot engine's unified response function

# 1. Initialize the Flask Application
# By default, Flask looks for templates inside the "templates/" directory 
# and static assets (images, css, js) inside the "static/" directory.
app = Flask(__name__)

# 2. Configure Logging
# We log diagnostic information to the standard output stream (console).
# In production, this helps track errors and trace application activity.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# 3. Define the Web UI Route
@app.route("/", methods=["GET"])
def index():
    """
    Renders and serves the chatbot UI (index.html) to the web browser.
    """
    try:
        # Load index.html from the 'templates/' folder
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Failed to render homepage template: {str(e)}", exc_info=True)
        # Return a simple plain-text message and 500 status if template folder is missing
        return "Critical UI resources are missing. Please verify the installation.", 500


# 4. Define the Chat JSON API Endpoint
@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts JSON messages from the user interface, passes them to the chatbot engine,
    and returns the predicted response with metadata in JSON format.
    
    Expected Request Payload:
        Content-Type: application/json
        Body: { "message": "User query here" }
        
    Expected Response Payload:
        Status Code: 200 OK
        Content-Type: application/json
        Body: { "response": "Bot response text", "confidence": 0.854, "intent": "tag" }
    """
    try:
        # Phase 10: Validate Content-Type header
        if not request.is_json:
            logger.warning("Rejected chat request: Missing 'application/json' Content-Type header.")
            return jsonify({
                "response": "Invalid request format. Content-Type must be 'application/json'.",
                "error": "InvalidContentType"
            }), 400

        # Phase 10: Parse and validate JSON payload existence
        data = request.get_json()
        if data is None:
            logger.warning("Rejected chat request: Request body is empty or malformed.")
            return jsonify({
                "response": "Malformed request. JSON body could not be parsed.",
                "error": "InvalidJSON"
            }), 400

        # Extract user message from JSON body (defaulting to empty string if missing)
        user_message = data.get("message", "")

        # Phase 10: Validate message input content
        if not isinstance(user_message, str) or not user_message.strip():
            logger.warning("Rejected chat request: Received an empty or non-string message.")
            return jsonify({
                "response": "I cannot answer an empty message. Please type something!",
                "error": "EmptyMessage"
            }), 400

        # Log the received message for monitoring (hashing or omitting PII in actual production)
        logger.info(f"Received message of length {len(user_message)} characters.")

        # 5. Process query using chatbot engine
        bot_response_payload = get_bot_response(user_message)

        # 6. Return response payload with 200 OK HTTP code
        return jsonify(bot_response_payload), 200

    except Exception as e:
        # Phase 10: Gracefully catch and log all unhandled application exceptions
        logger.error(f"Unexpected error occurred in /chat handler: {str(e)}", exc_info=True)
        
        # Return a structured fallback response and HTTP 500 Internal Server Error
        return jsonify({
            "response": "An internal server error occurred while processing your request. Please try again later.",
            "error": "InternalServerError"
        }), 500


# 7. Start the application when run as the main program
if __name__ == "__main__":
    logger.info("Initializing SmartAssist Flask Web Server...")
    # debug=True enables auto-reloading when source files change
    # port=5000 is the default local web server port
    app.run(debug=True, port=5000)
