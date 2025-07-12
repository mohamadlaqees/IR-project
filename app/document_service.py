import os
import sqlite3
import pandas as pd
import traceback
import time
from typing import Optional, List, Dict

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_script_dir, '..'))

DATA_BASE_DIR = os.path.join(PROJECT_ROOT, 'data')

ANTIQUE_QUERIES_PATH = os.path.join(DATA_BASE_DIR, 'antique', 'test', 'queries.txt')
WEBIS_QUERIES_PATH = os.path.join(DATA_BASE_DIR, 'webis-touche2020', 'queries.jsonl')

ANTIQUE_QRELS_PATH = os.path.join(DATA_BASE_DIR, 'antique', 'test', 'qrels.txt')
WEBIS_QRELS_PATH = os.path.join(DATA_BASE_DIR, 'webis-touche2020', 'qrels', 'test.tsv')

DB_PATH = None

def _load_data_from_db(db_path, table_name, id_column_name):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT {id_column_name}, text FROM {table_name}"
        start_time_sql = time.time()
        df = pd.read_sql_query(query, conn)
        # print(f"DEBUG (document_service): SQL query for {table_name} took: {time.time() - start_time_sql:.4f} seconds")
        df.rename(columns={id_column_name: 'id', 'text': 'text'}, inplace=True)
        data_dict = pd.Series(df.text.values, index=df.id).to_dict()
        # print(f"DEBUG (document_service): ✅ Data loaded from DB table '{table_name}'")
        return data_dict
    except Exception as e:
        print(f"Error loading data from DB table '{table_name}': {e}")
        traceback.print_exc()
        return {}
    finally:
        if conn:
            conn.close()

def initialize_document_data(project_root_path):
    global DB_PATH
    DB_PATH = os.path.join(project_root_path, 'IR_project.db')
    # print(f"DEBUG (document_service): Initializing document data. DB_PATH resolved to: {DB_PATH}")
    # print("DEBUG (document_service): Document texts will be loaded on-demand.")

def get_document_content(doc_id: str, dataset_name: str) -> Optional[str]:
    if DB_PATH is None:
        raise RuntimeError("Document service not initialized. Call initialize_document_data first.")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        table_name = f'cleaned_{dataset_name}'
        id_column = 'id' if dataset_name == 'antique' else '_id'
        query = f"SELECT text FROM {table_name} WHERE {id_column} = ?"
        
        cursor = conn.execute(query, (doc_id,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        return None
    except Exception as e:
        print(f"Error loading document content for {doc_id} from DB table '{table_name}': {e}")
        traceback.print_exc()
        return None
    finally:
        if conn:
            conn.close()

def get_document_contents_batch(dataset_name: str, doc_ids: List[str]) -> Dict[str, str]:
    start_total_time = time.time()
    if DB_PATH is None:
        raise RuntimeError("Document service not initialized. Call initialize_document_data first.")

    documents_map = {}
    if not doc_ids:
        return documents_map

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        table_name = f'cleaned_{dataset_name}'
        id_column = 'id' if dataset_name == 'antique' else '_id'
        
        placeholders = ','.join('?' for _ in doc_ids)
        query = f"SELECT {id_column}, text FROM {table_name} WHERE {id_column} IN ({placeholders})"
        
        cursor = conn.execute(query, doc_ids)
        for row in cursor:
            documents_map[row[0]] = row[1]
        
        # print(f"DEBUG (document_service): Retrieved {len(documents_map)} documents for {dataset_name} out of {len(doc_ids)} requested.")
        print(f"Time to retrieve {len(documents_map)} documents for {dataset_name} in batch: {time.time() - start_total_time:.4f} seconds")
        return documents_map
    except Exception as e:
        print(f"Error loading document contents batch from DB table '{table_name}': {e}")
        traceback.print_exc()
        return {}
    finally:
        if conn:
            conn.close()

def get_all_documents_for_faiss(dataset_name: str) -> Dict[str, str]:
    start_time = time.time()
    conn = None
    documents_map = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        table_name = f'cleaned_{dataset_name}'
        id_column = 'id' if dataset_name == 'antique' else '_id'
        query = f"SELECT {id_column}, text FROM {table_name}"
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        for row in cursor.fetchall():
            documents_map[str(row[0])] = row[1]
        
        print(f"Time to load all documents for FAISS build from '{table_name}': {time.time() - start_time:.4f} seconds. Count: {len(documents_map)}")
        return documents_map
    except Exception as e:
        print(f"Error loading all documents for FAISS build from DB table '{table_name}': {e}")
        traceback.print_exc()
        return {}
    finally:
        if conn:
            conn.close()

def load_all_doc_ids_from_db(dataset_name: str) -> List[str]:
    start_time = time.time()
    if DB_PATH is None:
        raise RuntimeError("Document service not initialized. Call initialize_document_data first.")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        table_name = f'cleaned_{dataset_name}'
        id_column = 'id' if dataset_name == 'antique' else '_id'
        query = f"SELECT {id_column} FROM {table_name}"
        
        df = pd.read_sql_query(query, conn)
        print(f"Time to load all doc IDs for {dataset_name} from DB: {time.time() - start_time:.4f} seconds")
        return df[id_column].tolist()
    except Exception as e:
        print(f"Error loading document IDs from DB table '{table_name}': {e}")
        traceback.print_exc()
        return []
    finally:
        if conn:
            conn.close()


