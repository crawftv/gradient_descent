import contextlib
import re
from enum import Enum

from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from tqdm import tqdm

from high_life.settings import chroma_collection


class Category(Enum):
    RESTAURANT = 1
    HOTEL_RESORT = 2
    DESTINATION_REVIEW = 3
    PODCAST_DESCRIPTION = 4
    MARKETING = 5
    WINE = 6
    RECIPE = 7
    INTERVIEW = 8
    STORE = 9
    STYLE = 10
    PRODUCT_PROMOTION = 11
    EMAIL_SUMMARY = 12
    OTHER = 13


def get_category(query_str):
    prompt = PromptTemplate("""You are the world's best AI assistant trained to categorize short articles or pieces of articles into predefined categories. 
        Your goal is to analyze each piece of text to assign the most relevant category.
          
        Predefined Categories:  
          
        1. Restaurant
            - Focused on a single restaurant
            - a background of the chef or founder
        2. Hotel/Resort
            - Focused on a single place to stay in one city 
        3. Destination Review
            - Focus on a city or country
        4. Podcast description
            - description or ad copy for a podcast
        5. marketing
            - asking to subscribe or unsubscribe to newsletter
            - buying a copy of a magazine
        6. Wine 
            - expert recommendation
            - description
        7. Recipe
            - a recipe featuring a list of ingredients 
            - could be a single ingredient
        8. Interview
            - An question/answer article between 2 people.
        9. Store
            - a feature on a place to shop or its founder/creator.
        10. Style
            - an article an article of clothing or clothing style in general
        11. Product Promotion
            - ad copy for a single item of clothing or accessory
        12. Email summary
            - a summary of a full article referencing many different topics.
        13. Other
           - Does not fit into any other category
            
        ----------------
        ARTICLE: {query_str}
        ----------------
        RESPONSE FORMAT: 'Since the text mentions [thing], the best category is [return one category number that best categorizes the article.]'
        """)

    model = Ollama(model="mistral", temperature=0.1)

    model_resp = model.complete(prompt.format(query_str=query_str), max_output=40).text
    with contextlib.suppress(ValueError, IndexError):
        resp = int(re.findall(r'\d{1,2}', model_resp)[-1])
        if resp in range(1, 13 + 1):
            return resp
    return 13


if __name__ == "__main__":
    metadata_version = 1
    for i in tqdm(range(chroma_collection.count())):
        res = chroma_collection.get(limit=1, offset=i)
        if "category" not in res['metadatas'][0].keys():
            category = get_category(f"{res['metadatas'][0]}\n{res['documents'][0]}")
            new_metadata = res["metadatas"][0]
            new_metadata["category"] = Category(category).name
            chroma_collection.update(ids=res["ids"], metadatas=[new_metadata])
    # 100%|██████████| 6335/6335 [8:57:37<00:00,  5.09s/it]
