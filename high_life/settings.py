import chromadb
from llama_index.core import StorageContext, Settings, set_global_handler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

set_global_handler("simple")  # side effect
chunking_agent_llm = Ollama(model="mistral", temperature=0.5)
question_answering_llm = Ollama(model="mistral", temperature=0.1)
rerank_llm = Ollama(model="mistral")

embedding_model_revision = None  # Replace with the specific revision to ensure reproducibility in  case the model is updated.
Settings.embed_model = HuggingFaceEmbedding("avsolatorio/GIST-Embedding-v0")

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
chroma_collection = chroma_client.get_or_create_collection("high_life")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# def embed(texts: list[str]) -> list[list[float]]:
#     model = Settings.embed_model
#     return model.encode(texts).tolist()[:len(texts)]
