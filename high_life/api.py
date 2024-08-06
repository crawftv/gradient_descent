import logging
import os
import sqlite3
import uuid
from datetime import datetime

from anthropic import Anthropic
from bs4 import BeautifulSoup
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
from llama_index.llms.groq import Groq
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler, Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette import status
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from prompts import claude_prompt, accumulated_prompt
from search import log_retrieval, retrieve_documents, gather_nodes_recursively, hyde_vector_retriever
from settings import logging_startup, vector_store, storage_context

logging_startup()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()
app.mount("/src", StaticFiles(directory="src"), name="src")
simple_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
logger = logging.getLogger("uvicorn.out")

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

security = HTTPBearer()

ADMIN_TOKEN = os.getenv("HIGH_LIFE_ADMIN_TOKEN")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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


@app.post("/query")
@limiter.limit("5/minute")
async def query(request: Request, input: Input) -> dict[str, str]:
    logging_id: str = uuid.uuid4().hex
    docs = docs_accumulation(query_str=input.text, logging_id=logging_id)
    answer = anthropic_call(docs, input.text, logging_id)
    logger.info(f"{logging_id}:{datetime.utcnow()} - Query finalized")

    return {"text": answer}


@limiter.limit("5/minute")
@app.post("/vector-query")
def vector_query(request: Request, input: Input):
    return retrieve_documents(query_str=input.text)


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


def docs_accumulation(query_str: str, logging_id) -> str:
    # sourcery skip: inline-immediately-returned-variable
    log_query(logging_id, query_str=query_str)
    logger.info(f"{logging_id}:{datetime.utcnow()} - Starting Query")
    docs = hyde_vector_retriever.retrieve(query_str)
    logger.info(f"{logging_id}:{datetime.utcnow()} - documents retrieved")
    log_retrieval(logging_id, docs)
    texts = gather_nodes_recursively(docs)
    logger.info(f"{logging_id}:{datetime.utcnow()} - nodes gathered")
    answer_content = "<documents>"
    for index, node_list in enumerate(texts.values()):
        _answer_content = f"\n<document {index}:\n"
        for node in node_list:
            _answer_content += f" {node.get_content(metadata_mode=MetadataMode.NONE)}"

        answer_content += f"{_answer_content}\n</document {index}>"
    answer_content += "\n</documents>"
    return answer_content


def anthropic_call(accumulated_docs, user_query, logging_id) -> str:
    client = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    messages = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        temperature=0.1,
        system=claude_prompt.format(docs=accumulated_docs),
        messages=[
            {"role": "user", "content": [{"type": "text", "text": user_query}]}
        ],
    ).content
    # final_response = llm.complete(final_prompt).text

    resp = " ".join([i.text for i in messages])
    log_final_responses(logging_id, f"{accumulated_docs}\n{user_query}", resp)
    xml = BeautifulSoup(f"<response>{resp}</response>", 'lxml-xml')
    return resp.text if (resp := xml.find("answer")) else "I could not find a suitable answer"


def call_model(accumulated_docs, user_query, logging_id) -> str:
    groq = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
    # Call the complete method with a query
    resp = groq.complete(accumulated_prompt.format(docs=accumulated_docs, query_str=user_query)).text

    # from groq import Groq
    #
    # client = Groq(
    #     api_key=os.environ.get("GROQ_API_KEY"),
    # )
    #
    # chat_completion = client.chat.completions.create(
    #     model="llama3-70b-8192",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": claude_prompt.format(docs=accumulated_docs),
    #         },
    #         {"role": "user", "content": [{"type": "text", "text": user_query}]}
    #     ],
    # ).content

    # resp = chat_completion.choices[0].message.content

    # Call the complete method with a query
    log_final_responses(logging_id, f"{accumulated_docs}\n{user_query}", resp)
    xml = BeautifulSoup(f"<response>{resp}</response>", 'lxml-xml')
    return resp.text if (resp := xml.find("answer")) else "I could not find a suitable answer"


class ScrapeInput(BaseModel):
    url: str


@limiter.limit("5/minute")
@app.post("/add_one")
def scrape_one(request: Request, input: ScrapeInput, token: str = Depends(verify_token), ):
    """add an url to be scraped later"""
    conn = sqlite3.connect("scraping.db")
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS urls
            (url TEXT PRIMARY KEY, complete BOOLEAN)
        ''')

    # Insert the new URL with complete set to False
    cursor.execute('''
            INSERT OR IGNORE INTO urls (url, complete)
            VALUES (?, ?)
        ''', (input.url, False))

    conn.commit()
    conn.close()
    return {"message": "success"}
