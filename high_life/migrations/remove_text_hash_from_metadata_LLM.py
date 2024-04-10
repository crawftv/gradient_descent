from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from high_life.migrations.add_NER_to_metadata import get_NER
from high_life.search import hyde_vector_retriever
from high_life.settings import chroma_collection, hex_id, chroma_client

if __name__ == "__main__":
    # chroma_client.delete_collection("high_life_4")
    new_collection = chroma_client.get_or_create_collection("high_life_4")
    new_vector_store = ChromaVectorStore(chroma_collection=new_collection, ssl=False, stores_text=True)
    new_storage_context = StorageContext.from_defaults(vector_store=new_vector_store)
    new_index = VectorStoreIndex.from_vector_store(new_vector_store, storage_context=new_storage_context)
    for i in tqdm(range(chroma_collection.count())):
        # result = chroma_collection.get(limit=1, offset=i)
        node = hyde_vector_retriever.retrieve("Afghan Anar, Zürich")
        result = chroma_collection.get(ids=[node.node_id])
        # create Node
        node: TextNode = metadata_dict_to_node(result["metadatas"][0])
        document = result['documents'][0]
        text_hash = hex_id(document)
        if len(document) > 1 and len(new_collection.get(where={"text_hash": text_hash})["documents"]) == 0:
            if not node.metadata.get("Named-Entities"):
                ner = get_NER(document)
                node.set_content(document)
                ner = {"Named-Entities": ner}
                node.metadata.update(ner)
            node.excluded_llm_metadata_keys = node.excluded_llm_metadata_keys + ["url", "Named-Entities", "text_hash"]
            node.excluded_embed_metadata_keys = ["url", "text_hash"]
            new_index.insert_nodes([node])
