import os
import sys
import traceback
import heapq
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import app.tfidf_search_service as tfidf_search_service
import app.embedding_search_service as embedding_search_service
import app.query_processing_service as query_processing_service
import app.document_service as document_service

def hybrid_search(query_text, dataset_name, top_n=10, top_k_sparse=1000):
    start_total_time = time.time()

    start_time_tfidf = time.time()
    try:
        tfidf_results = tfidf_search_service.tfidf_search(query_text, dataset_name, top_n=top_k_sparse)
    except Exception as e:
        print(f"ERROR (hybrid_search_service): Error during TF-IDF retrieval: {e}")
        traceback.print_exc()
        raise
    print(f"Time for TF-IDF retrieval (hybrid search): {time.time() - start_time_tfidf:.4f} seconds. Found {len(tfidf_results)} candidates.")

    if not tfidf_results:
        return []
    
    candidate_doc_ids = [result["doc_id"] for result in tfidf_results]

    start_time_query_embedding = time.time()
    try:
        processed_query_info = query_processing_service.process_query(query_text)
        query_embedding = np.array(processed_query_info["query_embedding"][0])
    except Exception as e:
        print(f"ERROR (hybrid_search_service): Error getting query embedding: {e}")
        traceback.print_exc()
        raise
    print(f"Time for query embedding (hybrid search): {time.time() - start_time_query_embedding:.4f} seconds")

    if embedding_search_service.global_loaded_data[dataset_name]["embeddings"] is None:
        embedding_search_service.embedding_search("dummy query", dataset_name, top_n=1) 

    dataset_embeddings = embedding_search_service.global_loaded_data[dataset_name]["embeddings"]
    dataset_doc_ids = embedding_search_service.global_loaded_data[dataset_name]["doc_ids"]
    
    doc_id_to_embedding_idx = {doc_id: i for i, doc_id in enumerate(dataset_doc_ids)}
    
    candidate_embeddings_list = []
    candidate_original_doc_ids = []
    
    for doc_id in candidate_doc_ids:
        idx = doc_id_to_embedding_idx.get(doc_id)
        if idx is not None:
            candidate_embeddings_list.append(dataset_embeddings[idx])
            candidate_original_doc_ids.append(doc_id)
    
    if not candidate_embeddings_list:
        return []

    candidate_embeddings = np.array(candidate_embeddings_list)

    start_time_reranking = time.time()
    reranked_scores = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
    print(f"Time for re-ranking (hybrid search): {time.time() - start_time_reranking:.4f} seconds")

    # FIX: Define reranked_results_with_scores before using it
    reranked_results_with_scores = []
    for i, score in enumerate(reranked_scores):
        reranked_results_with_scores.append((candidate_original_doc_ids[i], score))

    start_time_sort_top_n = time.time()
    final_top_n_scored_docs = heapq.nlargest(top_n, reranked_results_with_scores, key=lambda x: x[1])
    print(f"Time for final sort and top N (hybrid search): {time.time() - start_time_sort_top_n:.4f} seconds")

    start_time_snippets = time.time()
    doc_ids_for_snippets = [doc_id for doc_id, score in final_top_n_scored_docs]
    
    snippets_map = {}
    if doc_ids_for_snippets:
        try:
            snippets_map = document_service.get_document_contents_batch(dataset_name, doc_ids_for_snippets)
        except Exception as e:
            print(f"WARNING: Could not fetch snippets from document service in batch: {e}")
            traceback.print_exc()
    print(f"Time for batch snippet retrieval (hybrid search): {time.time() - start_time_snippets:.4f} seconds")

    final_results = []
    for doc_id, score in final_top_n_scored_docs:
        doc_content = snippets_map.get(doc_id, "")
        snippet = doc_content[:150] + "..." if len(doc_content) > 150 else doc_content
        
        final_results.append({
            "doc_id": doc_id,
            "score": float(score),
            "snippet": snippet
        })
    
    end_total_time = time.time()
    print(f"Total Hybrid search execution time: {end_total_time - start_total_time:.4f} seconds")

    return final_results


