import hashlib
import sqlite3

import chromadb
from llama_index.core import StorageContext, Settings, set_global_handler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

set_global_handler("simple")  # side effect

Settings.embed_model = HuggingFaceEmbedding("avsolatorio/GIST-large-Embedding-v0")
Settings.llm = Ollama(model="mistral:instruct", temperature=0.1, request_timeout=60.0)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
chroma_collection = chroma_client.get_or_create_collection("high_life_3")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


def hex_id(hash_value):
    h = hashlib.new('sha256')
    h.update(hash_value.encode())
    return h.hexdigest()


def logging_startup():
    conn = sqlite3.connect('logs.sqlite3')  # Replace with your database name
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            logging_id TEXT PRIMARY KEY,
            query TEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_retrieval (
            logging_id TEXT,
            document TEXT,
            FOREIGN KEY(logging_id) REFERENCES queries(logging_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filtering_responses (
            logging_id TEXT,
            input TEXT,
            result TEXT,
            FOREIGN KEY(logging_id) REFERENCES queries(logging_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS final_responses (
            logging_id TEXT PRIMARY KEY,
            input TEXT,
            result TEXT,
            FOREIGN KEY(logging_id) REFERENCES queries(logging_id)
        )
    """)

###
#  metadata_versions
#  1. adding categories
# 2. adding NER
