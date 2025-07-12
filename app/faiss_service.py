import os
import sys
import numpy as np
import faiss
import pickle
import time
import traceback
from typing import List, Dict

import app.text_processing_service as text_processing_service
import app.document_service as document_service

loaded_faiss_data = {
    "indices": {},
    "metadata": {},
    "sentence_transformer": None
}

FAISS_OUTPUT_BASE_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'FAISS_Indices')

def _load_faiss_dependencies_if_needed():
    if loaded_faiss_data["sentence_transformer"] is None:
        start_time = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            model_name = 'all-MiniLM-L6-v2'
            loaded_faiss_data["sentence_transformer"] = SentenceTransformer(model_name, local_files_only=True)
            print(f"Time to load SentenceTransformer model for FAISS: {time.time() - start_time:.4f} seconds")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to lazy load SentenceTransformer for FAISS: {e}")
            raise

def _load_index_if_needed(dataset: str):
    if dataset not in loaded_faiss_data["indices"]:
        start_time = time.time()
        index_path = os.path.join(FAISS_OUTPUT_BASE_DIR, f'{dataset}_faiss.index')
        metadata_path = os.path.join(FAISS_OUTPUT_BASE_DIR, f'{dataset}_metadata.pkl')
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                loaded_faiss_data["indices"][dataset] = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    loaded_faiss_data["metadata"][dataset] = pickle.load(f)
                print(f"Time to load FAISS index and metadata for '{dataset}': {time.time() - start_time:.4f} seconds")
            except Exception as e:
                print(f"ERROR: Failed to lazy load FAISS index for {dataset}: {e}")
                loaded_faiss_data["indices"][dataset] = None
                loaded_faiss_data["metadata"][dataset] = None
        else:
            print(f"ERROR: FAISS index files not found for dataset {dataset}. Please build them first.")
            loaded_faiss_data["indices"][dataset] = None
            loaded_faiss_data["metadata"][dataset] = None

def initialize_faiss_service(project_root_path):
    os.makedirs(FAISS_OUTPUT_BASE_DIR, exist_ok=True)

def search_faiss(query: str, dataset: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
    start_total_time = time.time()
    _load_faiss_dependencies_if_needed()
    _load_index_if_needed(dataset)
    
    index = loaded_faiss_data["indices"].get(dataset)
    metadata = loaded_faiss_data["metadata"].get(dataset)

    if not index or not metadata:
        print(f"ERROR (faiss_service): Cannot perform search, index or metadata for '{dataset}' not available.")
        return []

    try:
        start_time_clean = time.time()
        cleaned_query = text_processing_service.clean_text(query)
        print(f"Time for query cleaning (FAISS search): {time.time() - start_time_clean:.4f} seconds")
        if not cleaned_query: return []
        
        start_time_encode = time.time()
        query_embedding = loaded_faiss_data["sentence_transformer"].encode([cleaned_query])
        faiss.normalize_L2(query_embedding)
        print(f"Time for query embedding (FAISS search): {time.time() - start_time_encode:.4f} seconds")
        
        start_time_search = time.time()
        scores, indices = index.search(query_embedding, top_k)
        print(f"Time for FAISS index search: {time.time() - start_time_search:.4f} seconds")
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and score >= similarity_threshold:
                chunk_metadata = metadata[idx].copy()
                chunk_metadata['score'] = float(score)
                results.append(chunk_metadata)
        
        print(f"Total FAISS search execution time: {time.time() - start_total_time:.4f} seconds")
        return results
    except Exception as e:
        print(f"ERROR (faiss_service): Failed to search FAISS index: {e}")
        traceback.print_exc()
        return []


