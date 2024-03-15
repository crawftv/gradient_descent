from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from tqdm import tqdm

from high_life.settings import chroma_collection


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
    The possible common Named Entities (NER) types are exclusively: ({labels}) 
    EXAMPLE:
    Text: 'In Germany, in 1440, goldsmith Johannes Gutenberg invented the movable-type printing press. His work led to an information revolution and the unprecedented mass-spread / 
    of literature throughout Europe. Modelled on the design of the existing screw presses, a single Renaissance movable-type printing press could produce up to 3,600 pages per workday.'
    {{
        "gpe": ["Germany", "Europe"],
        "date": ["1440"],
        "person": ["Johannes Gutenberg"],
        "product": ["movable-type printing press"],
        "event": ["Renaissance"],
        "quantity": ["3,600 pages"],
        "time": ["workday"]
    TASK:
        Text: {query_str}
    }}""")
    prompt_template = prompt_template.partial_format(labels=", ".join(labels))
    model = Ollama(model="mistral", temperature=0.1)
    prompt = prompt_template.format(query_str=query_str)
    return model.complete(prompt).text


if __name__ == "__main__":
    metadata_version = 1
    for i in tqdm(range(chroma_collection.count())):
        res = chroma_collection.get(limit=1, offset=i)
        ner = get_NER(res['documents'][0])
        print(ner)
        #     new_metadata = res["metadatas"][0]
        #     new_metadata["category"] = Category(category).name
        #     chroma_collection.update(ids=res["ids"], metadatas=[new_metadata])
