from llama_index.core import PromptTemplate, StorageContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from high_life.settings import chroma_client, chroma_collection, hex_id


def get_NER(query_str):
    labels = [
        "person",  # people, including fictional characters
        "fac",  # buildings, airports, highways, bridges
        "org",  # organizations, companies, agencies, institutions
        "gpe",  # geopolitical entities like countries, cities, states
        "loc",  # non-gpe locations
        "product",  # vehicles, foods, appareal, appliances, software, toys
        "event",  # named sports, scientific milestones, historical events
        "work_of_art",  # titles of books, songs, movies
    ]
    prompt_template = PromptTemplate("""You are an expert in Natural Language Processing. Your task is to identify common Named Entities (NER) in a given text.
    The possible common Named Entities (NER) types are exclusively: ({labels}).
    If you include a city, add the country. For wine products, include the type if possible.
    EXAMPLE:
    Text: 'In Germany, in 1440, goldsmith Johannes Gutenberg invented the movable-type printing press. His work led to an information revolution and the unprecedented mass-spread / 
    of literature throughout Europe. Modelled on the design of the existing screw presses, a single Renaissance movable-type printing press could produce up to 3,600 pages per workday.'
    {{
        "gpe": ["Germany", "Europe"],
        "date": ["1440"],
        "person": ["Johannes Gutenberg"],
        "product": ["movable-type printing press"],
        "event": ["Renaissance"],
    TASK:
        Text: {query_str}
    }}
    MORE INSTRUCTIONS: DO NOT provide an explanation. Give just the Named Entities only.
    My job depends on it.
    """)

    prompt_template = prompt_template.partial_format(labels=", ".join(labels))
    model = Ollama(model="mistral", temperature=0.1, timeout=500)
    prompt = prompt_template.format(query_str=query_str)
    return model.complete(prompt).text


if __name__ == "__main__":
    metadata_version = 2
    chroma_client.delete_collection("high_life_3")
    new_collection = chroma_client.get_or_create_collection("high_life_3")
    new_vector_store = ChromaVectorStore(chroma_collection=new_collection, ssl=False, stores_text=True)
    new_storage_context = StorageContext.from_defaults(vector_store=new_vector_store)
    new_index = VectorStoreIndex.from_vector_store(new_vector_store, storage_context=new_storage_context)
    for i in tqdm(range(chroma_collection.count())):
        result = chroma_collection.get(limit=1, offset=i)

        # create Node
        node = metadata_dict_to_node(result["metadatas"][0])
        document = result['documents'][0]
        text_hash = hex_id(document)
        if len(new_collection.get(where={"text_hash": text_hash})["documents"]) == 0:
            ner = get_NER(document)
            node.set_content(document)
            node.excluded_llm_metadata_keys = node.excluded_llm_metadata_keys + ["url", "Named Entities"]
            node.excluded_embed_metadata_keys = ["url"]

            ner = {"Named-Entities": ner}
            # print(ner)

            node.metadata.update(ner)
            node.metadata.update({"text_hash": text_hash})
            new_index.insert_nodes([node])
