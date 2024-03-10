import chromadb
from llama_index.core import StorageContext, Settings, set_global_handler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

set_global_handler("simple")  # side effect

Settings.embed_model = HuggingFaceEmbedding("avsolatorio/GIST-large-Embedding-v0")
Settings.llm = Ollama(model="mistral:instruct", temperature=0.1, request_timeout=60.0)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
chroma_collection = chroma_client.get_or_create_collection("high_life")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
