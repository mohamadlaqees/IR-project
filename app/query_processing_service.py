import time
import json
from sentence_transformers import SentenceTransformer
import app.text_processing_service as text_processing_service

# --- Use a dictionary for the loaded model cache ---
_model_cache = {
    "sentence_transformer": None
}
SENTENCE_TRANSFORMER_MODEL_NAME = 'all-MiniLM-L6-v2'

def _load_model_if_needed():
    """Checks if the SentenceTransformer is loaded and loads it if not."""
    if _model_cache["sentence_transformer"] is None:
        start_time = time.time()
        try:
            _model_cache["sentence_transformer"] = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME, local_files_only=True)
            print(f"Time to load SentenceTransformer model: {time.time() - start_time:.4f} seconds")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to lazy load SentenceTransformer model: {e}")
            raise

def initialize_query_processor():
    """Initializes dependencies but does NOT load the heavy model."""
    # print("DEBUG (query_processing_service): Query processor initialized (model will be lazy-loaded).")
    text_processing_service.initialize_text_processor()

def process_query(raw_query_text):
    """
    Processes a raw query: cleans text and generates its embedding.
    """
    start_total_time = time.time()
    _load_model_if_needed()
    
    if not raw_query_text:
        return {"original_query": raw_query_text, "preprocessed_text": "", "query_embedding": []}

    start_time_clean = time.time()
    preprocessed_text = text_processing_service.clean_text(raw_query_text)
    print(f"Time for query text cleaning: {time.time() - start_time_clean:.4f} seconds")
    
    start_time_embed = time.time()
    query_embedding = _model_cache["sentence_transformer"].encode([raw_query_text], convert_to_tensor=True).cpu().numpy().tolist()
    print(f"Time for query embedding generation: {time.time() - start_time_embed:.4f} seconds")

    print(f"Total query processing time: {time.time() - start_total_time:.4f} seconds")
    return {
        "original_query": raw_query_text,
        "preprocessed_text": preprocessed_text,
        "query_embedding": query_embedding
    }


