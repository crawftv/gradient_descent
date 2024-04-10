import os
import re
import sqlite3
import uuid
from urllib.parse import urlparse, urlunparse

from anthropic import Anthropic
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from agent_splitter import get_nodes
from scraper import scrape_website
from search import log_retrieval, retrieve_documents, gather_nodes_recursively
from settings import logging_startup

logging_startup()
app = FastAPI()
app.mount("/src", StaticFiles(directory="src"), name="src")

origins = [
    "http://localhost:8002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("src/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/scrape")
def scrape_many(parent_url, save_links: bool = False):
    if save_links:
        parsed_url = urlparse(parent_url)
        base_url = urlunparse((parsed_url.scheme, parsed_url.netloc, '', '', '', ''))
        other_parts = urlunparse(('', '', parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))
        scrape_website(parent_url, save_links=True)


@app.post("/add_one")
def scrape_one(url: str):
    get_nodes(url)
    return {"message": "Done"}


class Input(BaseModel):
    text: str


def log_query(logging_id: str, query_str: str):
    with sqlite3.connect('logs.sqlite3') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO queries (logging_id,query) VALUES (?,?)", (logging_id, query_str))


@app.post("/query")
async def query(input: Input) -> dict[str, str]:
    logging_id: str = uuid.uuid4().hex
    return {"text": master_query(input.text, logging_id)}


@app.post("/vector-query")
def vector_query(input: Input):
    return retrieve_documents(query_str=input.text)


prompt = PromptTemplate(""" You are the world's best recommendation bot designed to answer user questions about food, travel, wine, etc.
    We have provided you a document to answer a query from an search embedding search.
    Your job is to determine whether the subject of document answers the query given the proper keywords or topics.
    if some asks for a red wine rec, only affirm red wines.
    if some asks about a city make sure the city is correct in the given document.
    QUERY: {query_str}
    ------------ Document ------------
    {doc}
    ----------------------------------
    ------------Output format------------
     Before answering the question, please think about it step-by-step within <thinking></thinking> tags.
     Then, provide your final answer within <answer>Yes/No</answer> tags.
    """)

accumulated_prompt = PromptTemplate("""You are an expert Q&A assistant, specializing in extracting and synthesizing the most relevant information from provided knowledge bases to answer user queries accurately and concisely.

Your role is to carefully analyze the given documents, identify the passages that directly address the user's question, and present that information in a clear, coherent response. If the knowledge base contains a complete answer, quote or paraphrase the relevant text. If it only partially answers the query, summarize the key relevant points. And if the provided documents do not contain a suitable answer, simply inform the user that an adequate answer could not be found in the given knowledge base.

Focus on extracting information solely from the provided documents, without adding any external knowledge or personal opinions. Aim to provide a response that fully addresses the query using only the most pertinent information from the knowledge base.
    -----------------
    KNOWLEDGE_BASE: {docs}
    -----------------
    QUERY: {query_str}
    -----------------
    INSTRUCTIONS:If the documents do not contain a suitable answer, simply respond: "i could not find a suitable answer".
    Do NOT suggest a solution from your own knowledge.
    Do NOT include phrases like 'Based on the provided documents, or 'According to the documents'. or  'based on the documents provided.'
    Include the url from the document metadata.
    ------------------
    Before answering the question, please think about it step-by-step within a <thinking></thinking> xml tag.
    Then, provide your final answer within <answer></answer> xml tag
    In the thinking steps, discuss the relevance of the content without mentioning document numbers.
    If you want to cite a source, consider using the author's name or the platform instead of the document number. For example: "According to @claire_rose on Instagram,..."
    EXAMPLES of  THOUGHT PROCESS:
    <examples>
    <example 1>
        Query: seafood restaurant in Paris?
        <thinking> The query asks for seafood restaurant recommendations in Paris. The provided information includes several relevant suggestions, such as a highly-regarded fish restaurant where everything is prepared exquisitely, and a top seafood spot in the city. Other details discuss wine pairings with seafood in general, which is less directly relevant to the specific query. 
        </thinking>
        <answer>
        [X] is a restaurant in Paris that features [a,b,c].
        </answer>
    </example 1>
    <example 2>
     Query: restaurant in Hong kong?
     <thinking>
     The query asks for any restaurants or dining spots in hong kong, mentioned in the documents,
     The provided documents include a recipe from a chef, John Woo, who has a restaurant in hong kong called the Hangout. The Hangout specializes in Hong Kong cuisine.
     The document also ...
     </thinking>
     <answer>
     1. The Hangout. The Hangout run by John Woo specializes in Hong Kong cuisine.
     2 ...
     </answer>
    </example 2>
    </examples>
    """)


def log_filtering_responses(logging_id, query_str: str, model_resp: str):
    with sqlite3.connect('logs.sqlite3') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO filtering_responses (logging_id,input,result) VALUES (?,?,?)",
                       (logging_id, query_str, model_resp))


def log_final_responses(logging_id, query_str: str, model_resp: str):
    with sqlite3.connect('logs.sqlite3') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO final_responses (logging_id,input,result) VALUES (?,?,?)",
                       (logging_id, query_str, model_resp))


def master_query(query_str: str, logging_id) -> str:
    # sourcery skip: inline-immediately-returned-variable
    log_query(logging_id, query_str=query_str)
    docs: list[NodeWithScore] = retrieve_documents(query_str=query_str)  # + vector_retrieved_docs2
    log_retrieval(logging_id, docs)
    texts = gather_nodes_recursively(docs)
    _answers = []
    model = Ollama(model="mistral", temperature=0, request_timeout=500, )
    for nodes in texts.values():
        doc = " ".join([node.get_content(metadata_mode=MetadataMode.LLM) for node in nodes])
        resp = model.complete(prompt.format(query_str=query_str, doc=doc))
        log_filtering_responses(logging_id, doc, resp.text)
        _answers.extend(
            nodes
            for i in re.finditer(
                r"<answer>(?P<answer>.*)</answer>", resp.text.lower()
            )
            if "yes" in i.groupdict()["answer"]
        )
    if not _answers:
        return "I could not find a suitable answer"
    answer_content = "<documents>"
    for index, node_list in enumerate(_answers):
        _answer_content = f"\n<document {index}:\n"
        for node in node_list:
            _answer_content += f" {node.get_content(metadata_mode=MetadataMode.LLM)}"

        answer_content += f"{_answer_content}\n</document {index}>"

    answer_content += "\n</documents>"
    final_prompt = accumulated_prompt.format(
        query_str=query_str,
        docs=answer_content
    )
    final_response = Ollama(model="vicuna:7b-16k", request_timeout=500).complete(final_prompt).text
    # final_response = Llamafile(temperature=0.1, seed=0).complete(final_prompt).text

    # final_response = anthropic_call(final_prompt)
    log_final_responses(logging_id, final_prompt, final_response)
    xml = BeautifulSoup(f"<response>{final_response}</response>", 'lxml-xml')
    answer = xml.find("answer").text
    return answer


def anthropic_call(text) -> str:
    client = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    messages = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ],
    ).content
    return " ".join([i.text for i in messages])


if __name__ == "__main__":
    master_query("ACtivities in Rome", uuid.uuid4().hex)
