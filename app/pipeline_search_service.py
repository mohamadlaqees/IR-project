import time
import numpy as np
import faiss
from typing import List, Dict

import app.hybrid_search_service as hybrid_search_service
import app.faiss_service as faiss_service
import app.document_service as document_service
import app.rag_service as rag_service

def get_basic_pipeline_candidates(query_text: str, dataset_name: str, top_n: int = 10) -> List[Dict]:
    start_time = time.time()
    candidate_docs = hybrid_search_service.hybrid_search(query_text, dataset_name, top_n=top_n)
    print(f"Time for basic pipeline (Hybrid Search) candidate retrieval: {time.time() - start_time:.4f} seconds")
    return candidate_docs

def faiss_with_basics(query_text: str, dataset_name: str, top_k_chunks: int = 10) -> List[Dict]:
    start_total_time = time.time()

    start_time_candidates = time.time()
    candidate_docs = get_basic_pipeline_candidates(query_text, dataset_name, top_n=50)
    print(f"Time to get candidate documents for faiss_with_basics: {time.time() - start_time_candidates:.4f} seconds")
    if not candidate_docs:
        return []
    
    candidate_doc_ids = {doc["doc_id"] for doc in candidate_docs}

    start_time_faiss_load = time.time()
    faiss_service._load_index_if_needed(dataset_name)
    faiss_service._load_faiss_dependencies_if_needed()
    print(f"Time to load FAISS dependencies and index for faiss_with_basics: {time.time() - start_time_faiss_load:.4f} seconds")
    
    full_index = faiss_service.loaded_faiss_data["indices"].get(dataset_name)
    full_metadata = faiss_service.loaded_faiss_data["metadata"].get(dataset_name)

    if not full_index or not full_metadata:
        print(f"ERROR: FAISS index for {dataset_name} not loaded.")
        return []

    start_time_filter_chunks = time.time()
    candidate_chunk_indices = [i for i, meta in enumerate(full_metadata) if meta["doc_id"] in candidate_doc_ids]
    print(f"Time to filter chunks for faiss_with_basics: {time.time() - start_time_filter_chunks:.4f} seconds. Found {len(candidate_chunk_indices)} chunks.")
    
    if not candidate_chunk_indices:
        return []
        
    start_time_reconstruct = time.time()
    candidate_vectors_np = np.array([full_index.reconstruct(int(i)) for i in candidate_chunk_indices]).astype("float32")
    print(f"Time to reconstruct candidate vectors for faiss_with_basics: {time.time() - start_time_reconstruct:.4f} seconds")

    start_time_filtered_index = time.time()
    filtered_index = faiss.IndexFlatIP(full_index.d)
    filtered_index.add(candidate_vectors_np)
    print(f"Time to build filtered FAISS index for faiss_with_basics: {time.time() - start_time_filtered_index:.4f} seconds")

    start_time_query_embedding = time.time()
    query_embedding = faiss_service.loaded_faiss_data["sentence_transformer"].encode([query_text])
    faiss.normalize_L2(query_embedding)
    print(f"Time for query embedding (faiss_with_basics): {time.time() - start_time_query_embedding:.4f} seconds")
    
    start_time_search = time.time()
    scores, indices = filtered_index.search(query_embedding, top_k_chunks)
    print(f"Time for FAISS search on filtered index (faiss_with_basics): {time.time() - start_time_search:.4f} seconds")

    final_results = []
    for i, score in zip(indices[0], scores[0]):
        if i != -1:
            original_metadata_index = candidate_chunk_indices[i]
            chunk_metadata = full_metadata[original_metadata_index]
            final_results.append({
                "doc_id": chunk_metadata["doc_id"],
                "score": float(score),
                "snippet": chunk_metadata["text"]
            })

    print(f"Total faiss_with_basics Service execution time: {time.time() - start_total_time:.4f} seconds")
    return final_results

def rag_with_basics(query_text: str, dataset_name: str, top_k_context: int = 3, instruction_key: str = "default") -> Dict:
    start_total_time = time.time()
    start_time_candidates = time.time()
    context_docs = get_basic_pipeline_candidates(query_text, dataset_name, top_n=top_k_context)
    print(f"Time to get candidate documents for rag_with_basics: {time.time() - start_time_candidates:.4f} seconds")
    if not context_docs:
        return {"generated_response": "Could not find any relevant documents to generate an answer.", "retrieved_context": []}
    
    start_time_doc_content = time.time()
    context_doc_ids = [doc["doc_id"] for doc in context_docs]
    documents_map = document_service.get_document_contents_batch(dataset_name, context_doc_ids)
    print(f"Time to retrieve full document contents for rag_with_basics: {time.time() - start_time_doc_content:.4f} seconds")
    
    retrieved_context_for_prompt = [{
        "doc_id": doc["doc_id"],
        "text": documents_map.get(doc["doc_id"], "Content not found."),
        "score": doc["score"]
    } for doc in context_docs]
    
    start_time_rag_load = time.time()
    rag_service._load_rag_model_if_needed()
    print(f"Time to load RAG model for rag_with_basics: {time.time() - start_time_rag_load:.4f} seconds")
    
    start_time_format_prompt = time.time()
    prompt = rag_service.format_context_for_llm(
        retrieved_context_for_prompt,
        query_text,
        rag_service.rag_models["tokenizer"],
        rag_service.rag_models["max_context_length"],
        instruction_key=instruction_key
    )
    print(f"Time to format prompt for rag_with_basics: {time.time() - start_time_format_prompt:.4f} seconds")
    
    start_time_generate = time.time()
    generated_response = rag_service.generate_response(prompt)
    print(f"Time to generate response for rag_with_basics: {time.time() - start_time_generate:.4f} seconds")
    
    print(f"Total RAG_with_basics Service execution time: {time.time() - start_total_time:.4f} seconds")
    return {
        "query": query_text,
        "generated_response": generated_response,
        "retrieved_context": retrieved_context_for_prompt,
    }


