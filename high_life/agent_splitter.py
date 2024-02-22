import functools
import re
from uuid import uuid4

import requests
from bs4 import BeautifulSoup, Tag
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from tqdm import tqdm

from high_life.data.konfekt_newsletters import konfekt_news_letters
from high_life.high_life_agent import index
from high_life.settings import chunking_agent_llm


class Chunk:
    def __init__(self, chunk, llm):
        self.chunk: Tag = chunk
        self.llm = llm

    @property
    @functools.cache
    def summary(self):
        return self.llm.complete(f"Provide a one sentence summary of the text below:\n{self.chunk.text}").text


from llama_index.core import PromptTemplate

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


def nodes_from_html(request: requests.Response):
    doc = BeautifulSoup(request.text, 'html.parser')
    parents: list[Tag | None] = []
    for p in doc.find_all('p'):
        if p.parent not in parents:
            parents.append(p.parent)
    parents: list[Chunk] = [Chunk(i, chunking_agent_llm) for i in parents]
    nodes = []
    for index in range(len(parents)):
        _result = chunking_agent_llm.complete(_fmt_promt(index, parents)).text.strip()
        related = _result.startswith("Yes")
        primary_text = strip_text(parents[index].chunk.p.text)
        breakpoint()
        new_node = TextNode(text=primary_text, id=uuid4(),
                            metadata={
                                "h1": strip_text(parents[index].chunk.h1.text) if parents[index].chunk.h1 else None,
                                "h2": strip_text(parents[index].chunk.h2.text) if parents[index].chunk.h2 else None,
                                "h3": strip_text(parents[index].chunk.h3.text) if parents[index].chunk.h3 else None,
                                "url": request.url,
                            })
        if related:
            new_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[-1].node_id
            )
            nodes[-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=new_node.node_id)
        nodes.append(new_node)
    return nodes


if __name__ == "__main__":
    nodes: list[TextNode] = []
    for url in tqdm(konfekt_news_letters):
        # print(url)
        response = requests.get(url)
        _nodes = nodes_from_html(response)
        # print(f"Nodes: {len(nodes)}")
        nodes += _nodes
    index.insert_nodes(nodes=nodes)
    # print(f"collection_count:{chroma_collection.count()}")
