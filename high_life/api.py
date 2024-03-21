import logging
import sqlite3
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from websockets.sync.client import connect

from search import retrieve_documents, gather_nodes_recursively, log_retrieval
from settings import logging_startup
from websocket_server import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
manager


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("src/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


class Input(BaseModel):
    text: str


def log_query(logging_id: str, query_str: str):
    with sqlite3.connect('logs.sqlite3') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO queries (logging_id,query) VALUES (?,?)", (logging_id, query_str))


@app.post("/query/{client_id}")
async def query(input: Input, client_id: str, ) -> dict[str, str]:
    query_str = input.text
    logging_id: str = uuid.uuid4().hex
    log_query(logging_id, query_str=query_str)
    docs: list[NodeWithScore] = retrieve_documents(query_str=query_str)
    with connect(f"ws://localhost:8003/ws/{client_id}") as websocket:
        websocket.send(f"Found {len(docs)} potential answers", )
    log_retrieval(logging_id, docs)
    texts = gather_nodes_recursively(docs)
    _answers = []
    model = Ollama(model="mistral", temperature=0, request_timeout=500, )
    for nodes in list(texts.values())[:2]:  # TODO Remove debugging slice
        doc = " ".join([node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes])
        resp = model.complete(prompt.format(query_str=query_str, doc=doc))
        log_filtering_responses(logging_id, doc, resp.text)
        if "resolves query: yes" in resp.text.lower():
            _answers.append(nodes)
    answer_content = ""
    for index, node_list in enumerate(_answers):
        _answer_content = f"Document {index}:\n"
        for node in node_list:
            _answer_content += f" {node.get_content(metadata_mode=MetadataMode.NONE)}"
        answer_content += _answer_content
    if not answer_content:
        final_response = "I could not find a suitable answer"
    else:
        final_prompt = accumulated_prompt.format(
            query_str=query_str,
            docs=answer_content
        )
        final_response = Ollama(model="mistral", temperature=0.1, request_timeout=500).complete(final_prompt).text
        log_final_responses(logging_id, final_prompt, final_response)
    return {"text": final_response}


prompt = PromptTemplate(""" You are the world's best recommendation bot designed to answer user questions about food, travel, wine, etc.
    We have provided you a document to answer a query from an search embedding search.
    Your job is to determine whether the document answers the query using the relevant keywords.
    if user asks for a red wine, The document resolves the query if a red wine variety is provided.
    if user asks about a city make sure the city is correct in the given document.
    if user asks about an ingredient, the document must mention that ingredient to answer the question.
    QUERY: {query_str}
    ------------ Document ------------
    {doc}
    ------------Output format------------
    Summary: Give a summary of the document with respect to the query.
    Resolves Query: No/Yes 
    """)

accumulated_prompt = PromptTemplate("""You are the worlds best Q&A bot, specializing in synthesizing correct answers to questions.
    You will be given a knowledge base that has been proven to answer the user question.
    You're job is to extract the text that best answers the question and give it to the user.
    -----------------
    KNOWLEDGE_BASE: {docs}
    -----------------
    QUERY: {query_str}
    -----------------
    <<<RESPONSE INSTRUCTIONS: If the documents do not contain a suitable answer, simply respond: "i could not find a suitable answer".
    Do NOT suggest a solution from your own knowledge.
    Do NOT include phrases like 'Based on the provided documents, or 'According to the documents'. or  'based on the documents provided.'
    Make sure to include the url from the metadata for each document in the respected answer.>>>>
    -----------------
    Restatement of QUERY: {query_str}
    Response: [Put your response here and include your reasoning]
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

# async def master_query(query_str: str, logging_id, client_id) -> str:
#     log_query(logging_id, query_str=query_str)
#     docs: list[NodeWithScore] = retrieve_documents(query_str=query_str)
#     logger.info("found docs")
#
#     await manager.send_personal_message(f"Found {len(docs)} potential answers", client_id)
#     log_retrieval(logging_id, docs)
#     texts = gather_nodes_recursively(docs)
#     _answers = []
#     model = Ollama(model="mistral", temperature=0, request_timeout=500, )
#     for nodes in texts.values():
#         doc = " ".join([node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes])
#         resp = model.complete(prompt.format(query_str=query_str, doc=doc))
#         log_filtering_responses(logging_id, doc, resp.text)
#         if "resolves query: yes" in resp.text.lower():
#             _answers.append(nodes)
#     answer_content = ""
#     for index, node_list in enumerate(_answers):
#         _answer_content = f"Document {index}:\n"
#         for node in node_list:
#             _answer_content += f" {node.get_content(metadata_mode=MetadataMode.NONE)}"
#         answer_content += _answer_content
#     if not answer_content:
#         final_response = "I could not find a suitable answer"
#     else:
#         final_prompt = accumulated_prompt.format(
#             query_str=query_str,
#             docs=answer_content
#         )
#         final_response = Ollama(model="mistral", temperature=0.1, request_timeout=500).complete(final_prompt).text
#         log_final_responses(logging_id, final_prompt, final_response)
#     return final_response
