@echo off
REM Exit on error
setlocal enabledelayedexpansion
set "errorlevel="

REM Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

REM Download NLTK data
python -m nltk.downloader wordnet punkt averaged_perceptron_tagger stopwords
