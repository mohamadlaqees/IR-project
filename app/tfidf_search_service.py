import os
import sys
import joblib
import time
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_script_dir, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Path Setup ---

# Import the service modules directly
import app.document_service as document_service
import app.text_processing_service as text_processing_service

# --- Configuration Constants ---
TFIDF_OUTPUT_BASE_DIR = 'TF_IDF' 

# --- Global storage for loaded models/data ---
global_loaded_data = {
    "antique": {
        "vectorizer": None, "tfidf_matrix": None
    },
    "webis": {
        "vectorizer": None, "tfidf_matrix": None
    }
}

# --- Data Loading Functions ---
def _load_tfidf_data_internal(dataset_name):
    start_time = time.time()
    
    vectorizer_path = os.path.join(PROJECT_ROOT, TFIDF_OUTPUT_BASE_DIR, dataset_name, f'{dataset_name}_tfidf_vectorizer.joblib')
    matrix_path = os.path.join(PROJECT_ROOT, TFIDF_OUTPUT_BASE_DIR, dataset_name, f'{dataset_name}_tfidf_matrix.npz') 
    
    vectorizer = joblib.load(vectorizer_path)
    
    loaded_npz = np.load(matrix_path)
    tfidf_matrix = csr_matrix((loaded_npz['data'], loaded_npz['indices'], loaded_npz['indptr']), shape=loaded_npz['shape'])
    
    print(f"Time to load TF-IDF data for {dataset_name}: {time.time() - start_time:.4f} seconds")
    return vectorizer, tfidf_matrix

# --- Initialization Function (Public) ---
def initialize_tfidf_models(project_root_path):
    global PROJECT_ROOT
    PROJECT_ROOT = project_root_path

# --- TF-IDF Search Function (Public) ---
def tfidf_search(query_text, dataset_name, top_n=10, top_k_inverted_index=None):
    start_total_time = time.time()

    dataset_data = global_loaded_data.get(dataset_name)
    
    if not dataset_data or dataset_data["vectorizer"] is None:
        try:
            vectorizer, tfidf_matrix = _load_tfidf_data_internal(dataset_name)
            global_loaded_data[dataset_name]["vectorizer"] = vectorizer
            global_loaded_data[dataset_name]["tfidf_matrix"] = tfidf_matrix
        except Exception as e:
            print(f"ERROR (tfidf_search_service): Failed to load TF-IDF models for {dataset_name.upper()} on demand: {e}")
            traceback.print_exc()
            raise RuntimeError(f"TF-IDF models for {dataset_name} could not be loaded.")
        dataset_data = global_loaded_data.get(dataset_name)

    if not dataset_data or dataset_data["vectorizer"] is None: # Double check after attempted load
        raise RuntimeError(f"TF-IDF data for {dataset_name} not initialized. Check service startup logs.")

    vectorizer = dataset_data["vectorizer"]
    tfidf_matrix = dataset_data["tfidf_matrix"]
    
    doc_ids = document_service.load_all_doc_ids_from_db(dataset_name)
    if not doc_ids:
        raise RuntimeError(f"Could not load document IDs for {dataset_name} from database.")

    # Step 1: Preprocess query
    start_time_query_proc = time.time()
    processed_query = text_processing_service.clean_text(query_text)
    print(f"Time for query preprocessing (TF-IDF search): {time.time() - start_time_query_proc:.4f} seconds")

    if not processed_query:
        return []

    # Step 2: Transform query into TF-IDF vector
    start_time_transform = time.time()
    query_vector = vectorizer.transform([processed_query])
    print(f"Time for query transformation (TF-IDF search): {time.time() - start_time_transform:.4f} seconds")

    # Step 3: Calculate cosine similarity
    start_time_cosine_sim = time.time()
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    print(f"Time for cosine similarity calculation (TF-IDF search): {time.time() - start_time_cosine_sim:.4f} seconds")

    # Step 4: Get top N results
    start_time_top_n = time.time()
    if top_k_inverted_index and top_k_inverted_index < len(similarities):
        top_candidate_indices = np.argpartition(similarities, -top_k_inverted_index)[-top_k_inverted_index:]
        candidate_similarities = similarities[top_candidate_indices]
        candidate_doc_ids = [doc_ids[i] for i in top_candidate_indices] 
        
        sorted_candidate_indices = np.argsort(candidate_similarities)[::-1]
        final_indices = sorted_candidate_indices[:top_n]
        
        scored_docs = []
        for idx in final_indices:
            scored_docs.append((candidate_doc_ids[idx], candidate_similarities[idx]))
    else:
        top_indices = similarities.argsort()[::-1][:top_n]
        scored_docs = []
        for idx in top_indices:
            scored_docs.append((doc_ids[idx], similarities[idx]))

    print(f"Time for top-N selection (TF-IDF search): {time.time() - start_time_top_n:.4f} seconds")

    # Step 5: Retrieve snippets using document_service
    start_time_snippets = time.time()
    doc_ids_for_snippets = [doc_id for doc_id, score in scored_docs]
    
    snippets_map = {}
    if doc_ids_for_snippets:
        try:
            snippets_map = document_service.get_document_contents_batch(dataset_name, doc_ids_for_snippets)
        except Exception as e:
            print(f"WARNING: Could not fetch snippets from document service in batch: {e}")
            traceback.print_exc()
    print(f"Time for batch snippet retrieval (TF-IDF search): {time.time() - start_time_snippets:.4f} seconds")

    final_results = []
    for doc_id, score in scored_docs:
        doc_content = snippets_map.get(doc_id, "")
        snippet = doc_content[:150] + "..." if len(doc_content) > 150 else doc_content
        
        final_results.append({
            "doc_id": doc_id,
            "score": float(score),
            "snippet": snippet
        })
    
    end_total_time = time.time()
    print(f"Total TF-IDF search execution time: {end_total_time - start_total_time:.4f} seconds")

    return final_results


