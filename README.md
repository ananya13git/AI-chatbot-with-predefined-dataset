# SmartAssist – Intelligent NLP Chatbot

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?style=for-the-badge&logo=flask&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**SmartAssist** is an enterprise-grade, production-quality AI Chatbot built for an Artificial Intelligence Internship. Unlike beginner chatbots that rely on simple `if-else` string matching, SmartAssist utilizes a **Vector-Space Model** combining **NLTK Preprocessing**, **TF-IDF (Term Frequency - Inverse Document Frequency) Vectorization**, and **Cosine Similarity** to achieve accurate, intent-driven conversational capabilities.

---

## 📐 Architecture & Data Flow

The diagram below illustrates how user messages travel through preprocessing, vector space classification, confidence threshold evaluation, and response generation:

```
[ User Input ]
      │
      ▼
[ NLP Preprocessing Pipeline ]
  ├── 1. Lowercasing
  ├── 2. Tokenization (word_tokenize)
  ├── 3. Punctuation Removal
  ├── 4. Stopword Removal (NLTK Corpora)
  └── 5. POS-Aware Lemmatization (WordNet)
      │
      ▼
[ TF-IDF Vectorizer Transformation ]
  └── Maps tokens into sparse TF-IDF feature space
      │
      ▼
[ Cosine Similarity Computation ]
  └── Calculates dot-product similarity against trained intent matrix
      │
      ▼
[ Confidence Threshold Evaluator (Default >= 30%) ]
  ├── Over Threshold ──► Select Random Intent Response
  └── Below Threshold ─► Trigger Fallback Clarification
      │
      ▼
[ Flask REST API Output (JSON) ]
      │
      ▼
[ Modern Glassmorphism Web Interface ]
```

---

## 📁 Project Directory Structure

```
SmartAssist/
│
├── app.py                     # Flask Web Server & REST API Endpoints
├── chatbot.py                 # Central Orchestrator & Engine Singleton
├── intents.json               # Primary Intent Dataset (20 intents, 200+ patterns)
├── requirements.txt           # Python Dependencies
├── README.md                  # Comprehensive Project Documentation
│
├── model/
│   ├── preprocess.py          # NLTK Preprocessing Pipeline
│   ├── intent_matcher.py      # TF-IDF Vectorizer & Cosine Similarity Classifier
│   └── response_engine.py    # Threshold Evaluator & Response Engine
│
├── templates/
│   └── index.html             # HTML5 Glassmorphism UI Template
│
├── static/
│   ├── style.css              # Custom CSS Design System, Animations & Themes
│   ├── script.js              # Fetch API Handler & Dynamic Typewriter UI
│   └── chatbot.png            # Futuristic 3D AI Assistant Avatar Icon
│
├── dataset/
│   └── training_data.json     # Backup Training Dataset
│
└── screenshots/               # Application UI Screenshots & Demo Recordings
```

---

## ⚙️ Installation & Running Instructions

### Prerequisites
- Python 3.12 or higher installed on your system.
- Git (optional, for cloning).

### Step 1: Navigate to Project Directory
```bash
cd SmartAssist
```

### Step 2: Create & Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Flask Web Application
```bash
python app.py
```

### Step 5: Access the Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🧠 Theoretical & Algorithmic Explanations

### 1. NLP Preprocessing (`model/preprocess.py`)
Human language contains grammatical variations, noise, and non-essential filler words. Preprocessing cleans and standardizes raw text:

1. **Lowercasing**: Converts `"Machine Learning"` to `"machine learning"`, eliminating case-sensitivity mismatches.
2. **Tokenization**: Uses NLTK's `word_tokenize()` to split continuous strings into discrete lexical units (tokens).
3. **Punctuation Removal**: Filters out symbols like `?`, `!`, `,`, and `.` using Python's `string.punctuation` dictionary.
4. **Stopword Removal**: Eliminates high-frequency grammatical words (e.g., `"the"`, `"is"`, `"at"`, `"which"`) that lack domain-specific intent information.
5. **POS-Aware Lemmatization**: Uses `WordNetLemmatizer` mapped with Penn Treebank Part-of-Speech (POS) tags to reduce words to their true dictionary lemma (e.g., `"running"` $\rightarrow$ `"run"`, `"better"` $\rightarrow$ `"good"`).

### 2. TF-IDF Vectorization (`model/intent_matcher.py`)
TF-IDF measures the importance of a word relative to a document in a corpus.

- **Term Frequency (TF)**:
  $$\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in document } d}$$

- **Inverse Document Frequency (IDF)**:
  $$\text{IDF}(t, D) = \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$

- **TF-IDF Weight**:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

High weights are assigned to distinctive terms (e.g., `"python"`, `"cybersecurity"`) while generic terms across documents receive low weights.

### 3. Cosine Similarity Classification (`model/intent_matcher.py`)
To determine how closely a user's input query matches pre-trained intent patterns, we calculate the cosine of the angle between their TF-IDF vectors in multi-dimensional space:

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- **1.0**: Identical semantic direction.
- **0.0**: Completely orthogonal (no shared terms).

### 4. Confidence Thresholding & Fallback (`model/response_engine.py`)
If a user submits out-of-scope queries or gibberish (e.g., `"qwertyuiop"`), vector models still compute a maximum score. SmartAssist enforces a **confidence threshold of 0.30 (30%)**. Queries below this threshold safely trigger a polite clarification response:
> *"I'm sorry, I couldn't understand your question. Could you please rephrase it?"*

---

## 🌐 Flask REST API Documentation

### 1. Root UI Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**: Renders `templates/index.html`.

### 2. Chat JSON Endpoint
- **URL**: `/chat`
- **Method**: `POST`
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "message": "What is Machine Learning?"
  }
  ```
- **Response Body (200 OK)**:
  ```json
  {
    "response": "Machine Learning (ML) is a subset of AI that enables systems to learn and improve from experience...",
    "confidence": 0.8642,
    "intent": "machine_learning"
  }
  ```
- **Error Responses**:
  - `400 Bad Request`: Returned for missing JSON headers, empty messages, or non-string payloads.
  - `500 Internal Server Error`: Returned for unexpected application errors.

---

## 🎨 UI/UX Design System Features

- **Glassmorphism Aesthetic**: Translucent cards featuring `backdrop-filter: blur(20px)`, subtle glowing borders, and floating ambient background light orbs.
- **Dynamic Typewriter Animation**: Bot responses stream smoothly character-by-character.
- **Interactive Quick Prompt Chips**: Allows instant one-click testing of pre-defined queries.
- **Response Metadata Badges**: Displays intent tags and TF-IDF similarity confidence percentages.
- **Mobile Responsive Layout**: Auto-adjusts for desktop, tablet, and mobile screens.

---

## 🛡️ Code Quality & PEP8 Standards
- Fully compliant with **PEP8** Python formatting guidelines.
- Modular architecture separating model logic (`model/`), central orchestration (`chatbot.py`), and HTTP web server (`app.py`).
- 100% documented with Python docstrings (`"""..."""`) and inline comments explaining NLP theory.

---

## 📸 Screenshots

*(Add screenshots of the SmartAssist interface running in your browser inside the `screenshots/` directory)*

---

## 🚀 Future Improvements

1. **Context Memory**: Integrate multi-turn dialogue history tracking.
2. **Deep Learning Intent Classifier**: Expand classifier using Transformer models (e.g., BERT or Sentence-Transformers).
3. **Voice Input/Output**: Integrate Web Speech API for voice commands and speech synthesis.
4. **Live External APIs**: Connect weather and news intent handlers to live third-party REST APIs.
