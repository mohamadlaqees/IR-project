import os
import sys
import traceback
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import all services ---
import app.document_service as document_service
import app.text_processing_service as text_processing_service
import app.query_processing_service as query_processing_service
import app.tfidf_search_service as tfidf_search_service
import app.embedding_search_service as embedding_search_service
import app.hybrid_search_service as hybrid_search_service
import app.faiss_service as faiss_service
import app.rag_service as rag_service
import app.pipeline_search_service as pipeline_search_service

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Health Check Endpoint ---
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "unified_ir_system"})

# --- Unified Query Endpoint ---
@app.route("/unified_query", methods=["POST", "OPTIONS"])
def unified_query_route():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        query_text = data.get("query")
        dataset_name = data.get("dataset")
        method = data.get("method")

        if not all([query_text, dataset_name, method]):
            return jsonify({"error": "Missing query, dataset, or method"}), 400

        instruction_key = data.get("instruction_key", "default")
        result = {}

        start_query_time = time.time()

        # --- Dispatch to the correct service based on the method ---
        if method == "tfidf":
            results = tfidf_search_service.tfidf_search(query_text, dataset_name)
            result = {"results": results}
        elif method == "embedding_brute_force":
            results = embedding_search_service.embedding_search(query_text, dataset_name)
            result = {"results": results}
        elif method == "embedding_faiss":
            faiss_res = faiss_service.search_faiss(query_text, dataset_name)
            results = [{"doc_id": r.get('doc_id'), "score": r.get('score', 0.0), "snippet": r.get('text', '')[:150] + "..."} for r in faiss_res]
            result = {"results": results}
        elif method == "hybrid":
            results = hybrid_search_service.hybrid_search(query_text, dataset_name)
            result = {"results": results}
        elif method == "rag_with_faiss":
            result = rag_service.rag_query(query_text, dataset_name, instruction_key=instruction_key)
        elif method == "faiss_with_basics":
            results = pipeline_search_service.faiss_with_basics(query_text, dataset_name)
            result = {"results": results}
        elif method == "rag_with_basics":
            result = pipeline_search_service.rag_with_basics(query_text, dataset_name, instruction_key=instruction_key)
        else:
            return jsonify({"error": f"Invalid method: {method}"}), 400
        
        end_query_time = time.time()
        total_query_duration = end_query_time - start_query_time
        print(f"Total query execution time for method '{method}' on dataset '{dataset_name}': {total_query_duration:.4f} seconds")

        result['query'] = query_text
        result['dataset'] = dataset_name
        result['method'] = method
        result['total_query_duration'] = total_query_duration # Add duration to result
        
        return jsonify(result)

    except Exception as e:
        print(f"ERROR (main.py): Error in /unified_query route: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# --- Get Document Endpoint ---
@app.route("/get_document", methods=["POST", "OPTIONS"])
def get_document_route():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        doc_id = data.get("doc_id")
        dataset_name = data.get("dataset")
        if not all([doc_id, dataset_name]):
            return jsonify({"error": "Missing doc_id or dataset"}), 400
        content = document_service.get_document_content(doc_id, dataset_name)
        if content is None:
            return jsonify({"error": "Document not found"}), 404
        return jsonify({"doc_id": doc_id, "content": content})
    except Exception as e:
        print(f"ERROR (main.py): Error in /get_document route: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# --- Main Initialization ---
def initialize_all_services():
    start_time = time.time()
    try:
        document_service.initialize_document_data(project_root)
        text_processing_service.initialize_text_processor()
        query_processing_service.initialize_query_processor()
        tfidf_search_service.initialize_tfidf_models(project_root)
        embedding_search_service.initialize_embedding_models(project_root)
        faiss_service.initialize_faiss_service(project_root)
        end_time = time.time()
        print(f"All services initialized successfully in {end_time - start_time:.4f} seconds.")
    except Exception as e:
        print(f"CRITICAL ERROR (main.py): Failed to initialize one or more services: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    initialize_all_services()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


