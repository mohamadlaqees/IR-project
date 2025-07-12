import os
import sys
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import heapq
import time

import app.query_processing_service as query_processing_service
import app.document_service as document_service

PROJECT_ROOT = None

EMBEDDING_OUTPUT_BASE_DIR = 'Embedding'

global_loaded_data = {
    "antique": {
        "embeddings": None, "doc_ids": None
    },
    "webis": {
        "embeddings": None, "doc_ids": None
    }
}

def _load_embedding_data_internal(dataset_name):
    start_time = time.time()
    embeddings_path = os.path.join(PROJECT_ROOT, EMBEDDING_OUTPUT_BASE_DIR, dataset_name, f'{dataset_name}_embeddings.joblib')
    ids_path = os.path.join(PROJECT_ROOT, EMBEDDING_OUTPUT_BASE_DIR, dataset_name, f'{dataset_name}_ids.joblib')
    
    embeddings = joblib.load(embeddings_path)
    doc_ids = joblib.load(ids_path)
    print(f"Time to load embedding data for {dataset_name}: {time.time() - start_time:.4f} seconds")
    return embeddings, doc_ids

def initialize_embedding_models(project_root_path):
    global PROJECT_ROOT
    PROJECT_ROOT = project_root_path

def embedding_search(query_text, dataset_name, top_n=10):
    start_total_time = time.time()

    dataset_data = global_loaded_data.get(dataset_name)
    
    if not dataset_data or dataset_data["embeddings"] is None:
        try:
            embeddings, doc_ids = _load_embedding_data_internal(dataset_name)
            global_loaded_data[dataset_name]["embeddings"] = embeddings
            global_loaded_data[dataset_name]["doc_ids"] = doc_ids
        except Exception as e:
            print(f"ERROR (embedding_search_service): Failed to load embedding data for {dataset_name.upper()} on demand: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Embedding data for {dataset_name} could not be loaded.")
        dataset_data = global_loaded_data.get(dataset_name)

    if not dataset_data or dataset_data["embeddings"] is None:
        raise RuntimeError(f"Embedding data for {dataset_name} not initialized. Check service startup logs.")

    embeddings = dataset_data["embeddings"]
    doc_ids = dataset_data["doc_ids"]

    start_time_query_proc = time.time()
    processed_query_info = query_processing_service.process_query(query_text)
    query_embedding = np.array(processed_query_info["query_embedding"][0])
    print(f"Time for query processing (embedding search): {time.time() - start_time_query_proc:.4f} seconds")

    start_time_cosine_sim = time.time()
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()
    print(f"Time for cosine similarity calculation (embedding search): {time.time() - start_time_cosine_sim:.4f} seconds")
    
    results_with_scores = []
    for i, score in enumerate(similarities):
        results_with_scores.append((doc_ids[i], score))
    
    start_time_heapq = time.time()
    top_n_scored_docs = heapq.nlargest(top_n, results_with_scores, key=lambda x: x[1])
    print(f"Time for top-N selection (embedding search): {time.time() - start_time_heapq:.4f} seconds")
    
    start_time_snippets = time.time()
    doc_ids_for_snippets = [doc_id for doc_id, score in top_n_scored_docs]
    
    snippets_map = {}
    if doc_ids_for_snippets:
        try:
            snippets_map = document_service.get_document_contents_batch(dataset_name, doc_ids_for_snippets)
        except Exception as e:
            print(f"WARNING: Could not fetch snippets from document service in batch: {e}")
            traceback.print_exc()
    print(f"Time for batch snippet retrieval (embedding search): {time.time() - start_time_snippets:.4f} seconds")

    final_results = []
    for doc_id, score in top_n_scored_docs:
        doc_content = snippets_map.get(doc_id, "")
        snippet = doc_content[:150] + "..." if len(doc_content) > 150 else doc_content
        
        final_results.append({
            "doc_id": doc_id,
            "score": float(score),
            "snippet": snippet
        })
    
    end_total_time = time.time()
    print(f"Total Embedding search execution time: {end_total_time - start_total_time:.4f} seconds")

    return final_results


