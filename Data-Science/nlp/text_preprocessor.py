# Text Preprocessor
import re
import sys

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_keywords(text, min_length=3):
    words = text.split()
    keywords = [word for word in words if len(word) >= min_length]
    return keywords