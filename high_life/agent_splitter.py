import functools
import hashlib
import re

import requests
from bs4 import BeautifulSoup, Tag
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, Document
from tqdm import tqdm

from high_life.data import konfekt_newsletters
from high_life.high_life_agent import index
from settings import Settings, chroma_collection

Settings


class Chunk:
    def __init__(self, chunk):
        self.chunk: Tag = chunk

    @property
    @functools.cache
    def summary(self):
        return Settings.llm.complete(f"Provide a one sentence summary of the text below:\n{self.chunk.text}").text


from llama_index.core import PromptTemplate, Settings

qa_prompt_tmpl_str = """All of the summaries below come from the same newsletter issue. I am trying to determine similarity for recall purposes.
chunk text is below.
---------------------
summary A: {middle_chunk}
--------------------- 
summary B: {previous_chunk}
---------------------

Determine whether these summaries share the same general context or topics and people of interest,. Start your answer with 'yes' or 'no'
"""

prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)


def _fmt_promt(index, parents: list[Chunk]):
    return prompt_tmpl.format(
        previous_chunk=parents[index - 1].summary,
        middle_chunk=parents[index].summary,
    )


strip_text_pattern = re.compile(r"(\s+\s+)", )
strip_text = lambda x: re.sub(strip_text_pattern, " ", x).strip()


def hex_id(hash_value):
    h = hashlib.new('sha256')
    h.update(hash_value.encode())
    return h.hexdigest()


def nodes_from_html(request: requests.Response, chunking_agent_llm=None):
    text = request.text  # simple_json_from_html_string(request.text, use_readability=True)["content"]
    doc = BeautifulSoup(text, 'html.parser')
    parents: list[Tag | None] = []
    for p in doc.find_all('p'):
        if p.parent not in parents:
            parents.append(p.parent)
    parents: list[Chunk] = [Chunk(i) for i in parents]
    parent_documents = []
    for parent_index in range(len(parents)):
        summary = parents[parent_index].summary
        primary_text = strip_text(parents[parent_index].chunk.text)
        sub_nodes = []
        parent_id = hex_id(primary_text)
        parent_document = Document(
            id=parent_id,
            metadata={
                "h1": strip_text(parents[parent_index].chunk.h1.text) if parents[parent_index].chunk.h1 else None,
                "h2": strip_text(parents[parent_index].chunk.h2.text) if parents[parent_index].chunk.h2 else None,
                "h3": strip_text(parents[parent_index].chunk.h3.text) if parents[parent_index].chunk.h3 else None,
                "url": request.url, },
            text=f"Summary of the Document: {summary}\nDocument: {primary_text}"
        )
        _result = Settings.llm.complete(_fmt_promt(parent_index, parents)).text.strip()
        related = _result.startswith("Yes")
        if related and parent_documents:
            parent_document.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=parent_documents[-1].node_id
            )
            parent_documents[-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=parent_document.node_id)

        # Sub texts
        sub_texts = parents[parent_index].chunk.find_all("p")
        if len(sub_texts) > 1:
            sub_nodes: list[TextNode] = []
            for i, sub_text in enumerate(sub_texts):
                sub_text_text = strip_text(sub_text.text)
                new_node = TextNode(
                    text=f"Summary of the Parent Document: {summary}\nDocument: {sub_text_text}",
                    id=hex_id(sub_text_text),
                    metadata={
                        "h1": strip_text(parents[parent_index].chunk.h1.text) if parents[
                            parent_index].chunk.h1 else None,
                        "h2": strip_text(parents[parent_index].chunk.h2.text) if parents[
                            parent_index].chunk.h2 else None,
                        "h3": strip_text(parents[parent_index].chunk.h3.text) if parents[
                            parent_index].chunk.h3 else None,
                        "url": request.url,
                    },
                )
                new_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_document.node_id)
                parent_document.relationships[NodeRelationship.CHILD] = RelatedNodeInfo(node_id=new_node.node_id)
                if i > 0:
                    previous_node = sub_nodes[-1]
                    previous_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=new_node.node_id)
                    new_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=previous_node.node_id)
                sub_nodes.append(new_node)
            index.insert_nodes(sub_nodes)
        parent_documents.append(parent_document)
    index.insert_nodes(parent_documents)
    return


if __name__ == "__main__":
    for url in tqdm(konfekt_newsletters.konfekt_news_letters):
        existing_documents = chroma_collection.get(where={"url": url})["documents"]
        if len(existing_documents) == 0:
            response = requests.get(url)

            nodes_from_html(response)
        else:
            print(f"Documents exist for {url}")

    print(f"collection_count:{chroma_collection.count()}")
    import datetime

    day_of_interest = datetime.date.today()
    while True:
        year = day_of_interest.year
        month = day_of_interest.month
        day = day_of_interest.day
        url = f"https://monocle.com/minute/{year}/{month}/{day}/"
        day_of_interest -= datetime.timedelta(days=1)
        existing_documents = chroma_collection.get(where={"url": url})["documents"]

        if len(existing_documents) == 0:
            response = requests.get(url)

            nodes_from_html(response)
        else:
            print(f"Documents exist for {url}")
