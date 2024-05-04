import os
import sqlite3
import uuid
from urllib.parse import urlparse, urlunparse

from anthropic import Anthropic
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from agent_splitter import get_nodes
from prompts import accumulated_prompt
from scraper import scrape_website
from search import log_retrieval, retrieve_documents, gather_nodes_recursively, hyde_vector_retriever
from settings import logging_startup, vector_store, storage_context

logging_startup()
app = FastAPI()
app.mount("/src", StaticFiles(directory="src"), name="src")
llm = Ollama(model="llama3", request_timeout=500, additional_kwargs={"num_ctx": 256_000}, temperature=0.1)
simple_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

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
    docs = hyde_vector_retriever.retrieve(query_str)
    log_retrieval(logging_id, docs)
    texts = gather_nodes_recursively(docs)
    answer_content = "<documents>"
    for index, node_list in enumerate(texts.values()):
        _answer_content = f"\n<document {index}:\n"
        for node in node_list:
            _answer_content += f" {node.get_content(metadata_mode=MetadataMode.NONE)}"

        answer_content += f"{_answer_content}\n</document {index}>"
    answer_content += "\n</documents>"
    final_prompt = accumulated_prompt.format(
        query_str=query_str,
        docs=answer_content
    )
    final_response = llm.complete(final_prompt).text
    # final_response = anthropic_call(final_prompt)
    log_final_responses(logging_id, final_prompt, final_response)
    xml = BeautifulSoup(f"<response>{final_response}</response>", 'lxml-xml')
    return resp.text if (resp := xml.find("answer")) else "I could not find a suitable answer"


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
