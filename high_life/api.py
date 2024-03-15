from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from high_life_agent import index
from high_life_prompt import master_query

templates = Jinja2Templates(directory="src")

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

query_engine = index.as_query_engine(similarity_top_k=10, )


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("src/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


class Input(BaseModel):
    text: str


@app.post("/query")
async def read_item(input: Input):
    return {"text": master_query(input.text)}
