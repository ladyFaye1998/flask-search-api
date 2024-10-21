from flask import Flask, request, jsonify
import os
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from functools import lru_cache
import logging
import json
import threading
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download the wordnet data if not already available
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# Ensure NLTK data is downloaded only once
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

required_nltk_packages = ['punkt', 'stopwords']
for package in required_nltk_packages:
    try:
        if package == 'punkt':
            nltk.data.find(f'tokenizers/{package}')
        else:
            nltk.data.find(f'corpora/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)

# Initialize Flask app
app = Flask(__name__)

# Initialize the transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Initialize the cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device='cpu')

# Global variables for ANN index and embeddings
ann_index = None
content_embeddings = None

# Load stop words and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to load links from 'All_Links.json'
def load_links_from_json(file_path='All_Links.json'):
    links = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                url = item.get('url', '').strip()
                description = item.get('description', '').strip()
                if url:
                    links.append({'url': url, 'description': description})
        if not links:
            logger.warning(f"No links found in '{file_path}'.")
    except FileNotFoundError:
        logger.error(f"Error: '{file_path}' file not found.")
    except Exception as e:
        logger.error(f"Error reading '{file_path}': {e}")
    return links

# Load links from JSON file
links = load_links_from_json('All_Links.json')
logger.info(f"Loaded {len(links)} links from 'All_Links.json'.")

# Helper function: Text preprocessing
def preprocess_text(text):
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text, flags=re.MULTILINE)
    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    # Remove stop words and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Helper function: Fetch and process link content with caching
@lru_cache(maxsize=1000)
def fetch_and_process_link_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        session = requests.Session()
        session.max_redirects = 5  # Limit redirects to prevent loops
        response = session.get(url, timeout=10, headers=headers, allow_redirects=True)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        # Extract text
        text = soup.get_text(separator=' ')
        processed_text = preprocess_text(text)
        return processed_text

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return ''

# Fetch and preprocess content for all links
def fetch_all_contents(links):
    contents = []
    for idx, link in enumerate(links):
        logger.info(f"Fetching content for link {idx+1}/{len(links)}: {link['url']}")
        content = fetch_and_process_link_content(link['url'])
        if content:
            combined_text = f"{link['description']} {content}"
        else:
            combined_text = link['description']
        link['combined_text'] = combined_text
        contents.append(combined_text)
    return contents

logger.info("Fetching and processing content from all links. This may take some time...")

# Run content fetching in a separate thread to avoid blocking
def precompute_contents():
    global content_embeddings
    contents = fetch_all_contents(links)
    logger.info("Computing embeddings for all contents...")
    content_embeddings = model.encode(contents, convert_to_numpy=True, normalize_embeddings=True)
    logger.info("Embeddings computed.")
    # Build ANN index after embeddings are computed
    build_ann_index(content_embeddings)

threading.Thread(target=precompute_contents).start()

# Helper function: Build ANN index using FAISS
def build_ann_index(embeddings):
    global ann_index
    embeddings_np = embeddings.astype('float32')
    dimension = embeddings_np.shape[1]
    ann_index = faiss.IndexFlatIP(dimension)  # Inner Product (for normalized embeddings)
    ann_index.add(embeddings_np)
    logger.info("Built ANN index.")

# Sigmoid function for score normalization
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Function to search for the most relevant links based on query
def find_relevant_links(query, top_n=3):
    if not links or content_embeddings is None or ann_index is None:
        logger.warning("Data is not ready for searching.")
        return []
    
    # Step 1: Preprocess the query
    query_processed = preprocess_text(query)

    # Step 2: Compute query embedding
    query_embedding = model.encode([query_processed], convert_to_numpy=True, normalize_embeddings=True)

    # Step 3: Perform ANN search
    num_candidates = min(len(links), 50)
    distances, indices = ann_index.search(query_embedding, num_candidates)
    indices = indices[0]

    # Step 4: Collect candidate links
    candidate_links = []
    for idx in indices:
        link = links[idx]
        candidate_links.append(link)

    # Step 5: Re-rank candidates using Cross-Encoder
    cross_encoder_inputs = [
        (query, candidate['combined_text'])
        for candidate in candidate_links
    ]

    cross_scores = cross_encoder.predict(cross_encoder_inputs)

    # Step 6: Normalize scores using sigmoid function
    for candidate, score in zip(candidate_links, cross_scores):
        candidate['raw_score'] = float(score)
        normalized_score = sigmoid(score)  # Scale between 0 and 1
        candidate['normalized_score'] = round(normalized_score, 2)  # Round to 2 decimal places

    # Step 7: Sort candidates by normalized score
    ranked_candidates = sorted(candidate_links, key=lambda x: x['normalized_score'], reverse=True)

    # Ensure there are always 3 links
    top_links = []
    for link in ranked_candidates[:top_n]:
        top_links.append({
            'url': link['url'],
            'description': link['description'],
            'score': link['normalized_score'],
        })

    # If less than 3 links are found, fill the rest with the next best candidates
    if len(top_links) < top_n:
        remaining_links = [link for link in candidate_links if link not in ranked_candidates]
        remaining_links = sorted(remaining_links, key=lambda x: x['normalized_score'], reverse=True)
        while len(top_links) < top_n and remaining_links:
            link = remaining_links.pop(0)
            top_links.append({
                'url': link['url'],
                'description': link['description'],
                'score': link['normalized_score'],
            })

    return top_links

# API endpoint to search for links
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = find_relevant_links(query)
    if results:
        return jsonify(results)
    else:
        # This should not happen with the updated code, but included for completeness
        return jsonify({"message": "No relevant links found"}), 404

# Add a route for the root URL
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Enhanced Link Search API!",
        "endpoints": {
            "/search": "Search for relevant links. Use '?query=' parameter with your search term."
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
