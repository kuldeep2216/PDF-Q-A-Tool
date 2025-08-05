# utils/cleaner.py

import re
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def clean_highlighted_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def extract_sentences(text):
    return sent_tokenize(text)
