AI-Powered Information Retrieval (IR) System







🌟 Project Overview

This project presents an advanced Information Retrieval (IR) system designed to deliver accurate and relevant search results across large datasets. The system combines traditional and modern search methods, including TF-IDF, Embeddings-based search, FAISS, and Hybrid Search, in addition to integrating Retrieval-Augmented Generation (RAG) techniques to provide comprehensive and direct answers. The system is built with a modular architecture based on the Service-Oriented Architecture (SOA) principle to ensure flexibility, scalability, and ease of maintenance.

✨ Key Features

•
Multiple Search Methods: Supports TF-IDF, Semantic Search (Brute Force and FAISS), and Hybrid Search.

•
Retrieval-Augmented Generation (RAG): Generates natural language answers based on retrieved document context.

•
Advanced Pipeline Search: Combines multiple search methods to improve accuracy and efficiency.

•
Modular Architecture: Service-oriented design for easy development and maintenance.

•
Flexible API: Allows for easy interaction with the system.

•
Interactive User Interface (Frontend): For a seamless search and chat experience.

•
Dynamic RAG Instruction Control: Ability to customize the behavior of the Large Language Model from the frontend.

🚀 Quick Start

To set up and run the project locally, follow these steps:

Prerequisites

Ensure you have the following installed on your system:

•
Python 3.9+ (recommended)

•
pip (Python package manager)

•
Node.js and npm (to run the frontend)

•
Git (to clone the repository)

1. Clone the Repository

Bash


git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


2. Backend Setup

Bash


cd project/app
pip install -r requirements.txt


Note: You might need to download NLTK resources for the first time:

Python


import nltk
nltk.download(\'punkt\')
nltk.download(\'stopwords\')
nltk.download(\'wordnet\')
nltk.download(\'omw-1.4\')


3. Database Setup and Data Loading

The project expects an SQLite database named IR_project.db in the project/app folder. You can load the antique and webis datasets using data_loader_utils.py.

Bash


python data_loader_utils.py


This will create the necessary tables and load the data. Ensure you have the raw data files in the correct paths or modify data_loader_utils.py to suit your paths.

4. Build FAISS Indexes (Optional but Recommended)

To improve semantic search performance, you can build FAISS indexes:

Bash


python faiss_service.py build_index antique
python faiss_service.py build_index webis


5. Run the Backend Server

Bash


python main.py


The server will run on http://127.0.0.1:8000 (or another port if specified).

6. Frontend Setup and Run

Bash


cd ../../frontend
npm install
npm start


The frontend will typically run on http://localhost:3000.

📂 Project Structure

The project consists of the following main components:

Plain Text


. # Repository root
├── project/
│   ├── app/ # Backend code
│   │   ├── data_loader_utils.py
│   │   ├── document_service.py
│   │   ├── embedding_search_service.py
│   │   ├── faiss_service.py
│   │   ├── hybrid_search_service.py
│   │   ├── main.py
│   │   ├── pipeline_search_service.py
│   │   ├── query_processing_service.py
│   │   ├── rag_service.py
│   │   ├── text_processing_service.py
│   │   ├── tfidf_search_service.py
│   │   └── IR_project.db (Database - will be created)
│   └── requirements.txt
└── frontend/
    ├── public/
    ├── src/
    │   ├── App.jsx
    │   ├── ChatInterface.jsx
    │   └── ... (other React components)
    └── package.json


🛠️ Core Services

•
main.py: Main entry point, server and API endpoint initialization.

•
document_service.py: Manages interaction with the document database.

•
text_processing_service.py: Text processing and cleaning (conversion, stop-word removal, lemmatization).

•
query_processing_service.py: Processes user queries and generates embeddings.

•
tfidf_search_service.py: Implements TF-IDF search.

•
embedding_search_service.py: Implements brute-force semantic search.

•
faiss_service.py: Builds, indexes, and searches FAISS.

•
hybrid_search_service.py: Orchestrates hybrid search (TF-IDF + semantic re-ranking).

•
rag_service.py: Implements the RAG pipeline for answer generation.

•
pipeline_search_service.py: Provides advanced search pipelines combining multiple methods.

📊 Evaluation Results

The performance of different search methods was evaluated on the Antique and Webis-Touche2020 datasets using standard information retrieval evaluation metrics. The following results show the average performance at K=10.

Datasets Used

•
Antique: Contains 403,457 documents.

•
Webis-Touche2020: Contains 382,545 documents.

Performance on Antique Dataset

Search MethodP@10Recall@10nDCG@10MAP@10MRR@10Hybrid0.41550.13690.36580.10880.7506Embedding (Brute Force)0.36950.12100.34120.09530.7145TF-IDF0.28950.09960.20570.07530.6114Embedding (FAISS)0.22900.07430.26650.06870.8261RAG (Retrieval)0.22900.07430.26650.06870.8261

Analysis: The Hybrid search method shows the best overall performance, while embedding methods (FAISS and RAG) excel at quickly finding the first relevant document (MRR@10).

Performance on Webis-Touche2020 Dataset

Search MethodP@10Recall@10nDCG@10MAP@10MRR@10TF-IDF0.24290.05120.12720.02680.4722Embedding (FAISS)0.22440.04650.17710.02980.5311RAG (Retrieval)0.22440.04650.17710.02980.5311Hybrid0.16120.03360.08920.01770.3694Embedding (Brute Force)0.14080.02960.06990.01440.3127

Analysis: Embedding methods (FAISS and RAG) outperform in nDCG, MAP, and MRR metrics on this dataset, indicating their better contextual understanding for argumentative queries.

Execution Times (Antique Dataset - Example)

Search MethodTotal Execution Time (seconds)TF-IDF17.77Embedding (Brute Force)4.49Embedding (FAISS)0.08Hybrid20.95RAG (with FAISS)4.74FAISS with Basics (Pipeline)2.70RAG with Basics (Pipeline)4.63

Analysis: FAISS is significantly faster after initial loading, while methods involving large language models (RAG) or large data loading (TF-IDF) show longer initial response times.

💡 Advanced Features

Dynamic Control of SYSTEM_INSTRUCTIONS for RAG Service

This feature allows users to dynamically customize the behavior of the Large Language Model (LLM) in rag_service.py directly from the frontend. Different instruction templates (e.g., default, comparative, bullet_points, beginner_friendly) can be selected to guide the LLM to generate answers with a specific tone or purpose. The instruction_key is passed from the frontend to the RAG service, providing great flexibility in answer phrasing.

Advanced Pipeline Search Services

pipeline_search_service.py allows combining different search methods into complex workflows:

•
faiss_with_basics: Combines basic search (TF-IDF and Hybrid) with FAISS efficiency. It uses basic search to identify initial candidates, then FAISS for precise searching within those candidates.

•
rag_with_basics: Uses the basic search pipeline to identify the best contextual documents, then generates a RAG answer based on the full content of those documents, ensuring high-quality context for the RAG model.

🤝 Contribution

Contributions to this project are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

