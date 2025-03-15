import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def preprocess_text(text):
    sentences = sent_tokenize(text)  # Split text into sentences
    sentences = [sent.strip() for sent in sentences]  # Remove leading/trailing whitespace
    return sentences
