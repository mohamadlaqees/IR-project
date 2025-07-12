import os
import sys
import traceback
import time
import json
from typing import List, Dict, Optional
import torch

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import app.faiss_service as faiss_service

# --- Use a dictionary for the loaded model cache ---
rag_models = {
    "llm_model": None,
    "tokenizer": None,
    "model_name": "google/flan-t5-large",
    "max_context_length": 512,
    "max_response_length": 150,
}

SYSTEM_INSTRUCTIONS = {
    "default": "You are a helpful assistant. Based on the provided context, answer the question. If the context does not contain enough information, state that the documents did not contain the answer. Summarize the relevant points from the context to form your answer.",
    "comparative": "You are a highly analytical assistant. Your primary goal is to compare and contrast concepts from the provided context. Structure your answer to clearly highlight the similarities and differences.",
    "bullet_points": "You are a concise assistant. Your task is to answer the question by extracting key information from the context and presenting it as a bulleted list. Keep each point brief and to the point.",
    "beginner_friendly": "You are a friendly and patient teacher. Explain the answer to the user's question in simple, easy-to-understand terms. Use analogies if they help clarify complex topics from the context.",
}

def _load_rag_model_if_needed():
    """Checks if the LLM is loaded and loads it if not."""
    if rag_models["llm_model"] is None:
        start_time = time.time()
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            model_name = rag_models["model_name"]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # print(f"INFO (rag_service): Loading {model_name} on device: {device}")

            rag_models["tokenizer"] = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            rag_models["llm_model"] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                local_files_only=True
            ).to(device)
            rag_models["llm_model"].eval()
            
            print(f"Time to load RAG Language Model: {time.time() - start_time:.4f} seconds.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to lazy load RAG Language Model: {e}")
            raise

def format_context_for_llm(retrieved_chunks: List[Dict], query: str, tokenizer, max_tokens_for_context: int, instruction_key: str = "default"):
    system_instruction = SYSTEM_INSTRUCTIONS.get(instruction_key, SYSTEM_INSTRUCTIONS["default"])
    base_prompt_estimate = len(tokenizer.encode(system_instruction + "\n\nQuestion: " + query + "\nAnswer:", add_special_tokens=False))
    available_context_tokens = max_tokens_for_context - base_prompt_estimate - 50
    context_parts = ["Context:"]
    current_context_tokens = 0
    for chunk in sorted(retrieved_chunks, key=lambda x: x.get('score', 0), reverse=True):
        chunk_text = f"Document (ID: {chunk.get('doc_id', 'N/A')}): {chunk.get('text', '')}"
        chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
        if current_context_tokens + len(chunk_tokens) <= available_context_tokens:
            context_parts.append(chunk_text)
            current_context_tokens += len(chunk_tokens)
        else:
            break
    full_context = "\n\n".join(context_parts)
    return f"{system_instruction}\n\n{full_context}\n\nQuestion: {query}\nAnswer:"

def generate_response(context: str) -> str:
    _load_rag_model_if_needed()
    try:
        inputs = rag_models["tokenizer"](context, return_tensors="pt", max_length=rag_models["max_context_length"], truncation=True, padding=True)
        device = rag_models["llm_model"].device
        with torch.no_grad():
            outputs = rag_models["llm_model"].generate(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                max_new_tokens=rag_models["max_response_length"],
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        return rag_models["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"ERROR (rag_service): Failed to generate response: {e}")
        return "Error generating response."

def rag_query(query: str, dataset: str, top_k: int = 5, similarity_threshold: float = 0.7, instruction_key: str = "default") -> Dict:
    start_total_time = time.time()
    _load_rag_model_if_needed()
    
    start_time_retrieval = time.time()
    retrieved_chunks = faiss_service.search_faiss(query=query, dataset=dataset, top_k=top_k, similarity_threshold=similarity_threshold)
    print(f"Time for RAG retrieval (FAISS search): {time.time() - start_time_retrieval:.4f} seconds")

    start_time_format_context = time.time()
    context = format_context_for_llm(retrieved_chunks, query, rag_models["tokenizer"], rag_models["max_context_length"], instruction_key=instruction_key)
    print(f"Time for RAG context formatting: {time.time() - start_time_format_context:.4f} seconds")

    start_time_generate_response = time.time()
    response = generate_response(context)
    print(f"Time for RAG response generation: {time.time() - start_time_generate_response:.4f} seconds")

    print(f"Total RAG query execution time: {time.time() - start_total_time:.4f} seconds")
    return {
        "query": query,
        "dataset": dataset,
        "retrieved_chunks": retrieved_chunks,
        "generated_response": response,
        "metadata": {"processing_time": time.time() - start_total_time}
    }


