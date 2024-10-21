#!/usr/bin/env bash

set -o errexit  # Exit on error

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader wordnet punkt averaged_perceptron_tagger stopwords
