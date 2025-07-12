import os
import pandas as pd
import sqlite3
import json
import time

# Assuming this utility file is in the 'app' directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Project root is the directory above 'app'
PROJECT_ROOT = os.path.abspath(os.path.join(current_script_dir, '..'))

# Define paths for queries and qrels relative to PROJECT_ROOT
# Adjust these paths if your data is located elsewhere
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, 'data')

# Query Paths
ANTIQUE_QUERIES_PATH = os.path.join(DATA_BASE_DIR, 'antique', 'test', 'queries.txt')
WEBIS_QUERIES_PATH = os.path.join(DATA_BASE_DIR, 'webis-touche2020', 'queries.jsonl')

# Qrels Paths
ANTIQUE_QRELS_PATH = os.path.join(DATA_BASE_DIR, 'antique', 'test', 'qrels.txt')
WEBIS_QRELS_PATH = os.path.join(DATA_BASE_DIR, 'webis-touche2020', 'qrels', 'test.tsv')

DB_PATH = os.path.join(PROJECT_ROOT, 'IR_project.db')

def load_queries_txt(filepath):
    """Loads queries from a plain text file (qid\tquery_text)."""
    start_time = time.time()
    queries = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    qid, query_text = parts
                    queries[qid] = query_text
                # else:
                #     print(f"Warning: Skipping malformed query line in {filepath}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Query file not found at {filepath}")
    except Exception as e:
        print(f"Error loading queries from {filepath}: {e}")
    print(f"Time to load queries from {os.path.basename(filepath)}: {time.time() - start_time:.4f} seconds")
    return queries

def load_queries_jsonl(filepath):
    """Loads queries from a JSON Lines file (each line is a JSON object with '_id' and 'text')."""
    start_time = time.time()
    queries = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if '_id' in data and 'text' in data:
                        queries[str(data['_id'])] = data['text']
                    # else:
                    #     print(f"Warning: Skipping malformed JSONL line (missing '_id' or 'text') in {filepath}: {line.strip()}")
                except json.JSONDecodeError:
                    pass # print(f"Warning: Skipping invalid JSONL line in {filepath}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Query file not found at {filepath}")
    except Exception as e:
        print(f"Error loading queries from {filepath}: {e}")
    print(f"Time to load queries from {os.path.basename(filepath)}: {time.time() - start_time:.4f} seconds")
    return queries

def load_qrels(filepath, delimiter=' ', skip_header=False):
    """Loads qrels from a text file with specified delimiter.
       Assumes format: qid [Q0] docid relevance
       For Webis, it's qid \t docid \t relevance (3 fields)
       For Antique, it's qid Q0 docid relevance (4 fields)
    """
    start_time = time.time()
    qrels = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if skip_header:
                next(f) # Skip the header line
            for line in f:
                parts = line.strip().split(delimiter)
                
                qid = None
                doc_id = None
                relevance = None

                if delimiter == '\t': # Webis format: qid \t docid \t relevance
                    if len(parts) >= 3:
                        qid = parts[0]
                        doc_id = parts[1]
                        relevance = parts[2]
                elif delimiter == ' ':
                    if len(parts) >= 4: # Antique format: qid Q0 docid relevance
                        qid = parts[0]
                        doc_id = parts[2] # docid is the 3rd field (index 2)
                        relevance = parts[3] # relevance is the 4th field (index 3)
                
                if qid is not None and doc_id is not None and relevance is not None:
                    try:
                        qrels.setdefault(qid, {})[doc_id] = int(relevance)
                    except ValueError:
                        pass # print(f"Warning: Skipping qrels line with non-integer relevance in {filepath}: {line.strip()}")
                # else:
                #     print(f"Warning: Skipping malformed qrels line (could not parse fields) in {filepath}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Qrels file not found at {filepath}")
    except Exception as e:
        print(f"Error loading qrels from {filepath}: {e}")
    print(f"Time to load qrels from {os.path.basename(filepath)}: {time.time() - start_time:.4f} seconds")
    return qrels

def load_antique_queries():
    return load_queries_txt(ANTIQUE_QUERIES_PATH)

def load_webis_queries():
    return load_queries_jsonl(WEBIS_QUERIES_PATH)

def load_antique_qrels():
    return load_qrels(ANTIQUE_QRELS_PATH, delimiter=' ', skip_header=False)

def load_webis_qrels():
    return load_qrels(WEBIS_QRELS_PATH, delimiter='\t', skip_header=True)

# You might also need a function to load document IDs if not already handled by search services
def load_all_doc_ids_from_db(dataset_name):
    start_time = time.time()
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
        return []
    finally:
        if conn:
            conn.close()



