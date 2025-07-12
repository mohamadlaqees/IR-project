import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import traceback
import time

stop_words = None
lemmatizer = None
punct_table = None

def initialize_text_processor():
    global stop_words, lemmatizer, punct_table
    start_time = time.time()
    try:
        if stop_words is None:
            stop_words = set(stopwords.words("english"))
        if lemmatizer is None:
            lemmatizer = WordNetLemmatizer()
        if punct_table is None:
            punct_table = str.maketrans("", "", string.punctuation)

        print(f"Time to initialize text processor: {time.time() - start_time:.4f} seconds.")
    except Exception as e:
        print(f"ERROR (text_processing_service): Failed to initialize text processor: {e}")
        traceback.print_exc()
        raise

def clean_text(text):
    if stop_words is None or lemmatizer is None or punct_table is None:
        initialize_text_processor()

    if not isinstance(text, str):
        return ""
    
    start_total_clean = time.time()
    
    text = text.lower()
    text = text.translate(punct_table)
    text = re.sub(r"\d+", "", text)
    
    tokens = word_tokenize(text)
    
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(word, pos="n")) 
            
    final_cleaned_text = " ".join(cleaned_tokens)

    print(f"Time for text cleaning: {time.time() - start_total_clean:.4f} seconds")
    return final_cleaned_text

def custom_tokenizer(text):
    if not isinstance(text, str):
        return []
    return text.split()


