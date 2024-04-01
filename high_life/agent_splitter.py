import contextlib
import re

import requests
import urllib3.util
from bs4 import BeautifulSoup, Tag
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, Document
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from high_life_agent import index
from propisitional_splitter import propositional_splitter
from settings import Settings, chroma_collection, hex_id

Settings

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


def _fmt_promt(*, previous_chunk_summary, middle_chunk_summary):
    return prompt_tmpl.format(
        previous_chunk=previous_chunk_summary,
        middle_chunk=middle_chunk_summary,
    )


strip_text_pattern = re.compile(r"(\s+\s+)", )
strip_text = lambda x: re.sub(strip_text_pattern, " ", x).strip()


def passes_filter(text):
    """Filter out bad or uninformative text."""
    text_filter_list = ["+44 (0) 20 7725 4349 (view hours)", "", "Aton\nPadded coat", 'Forgotten password?',
                        "Frequently asked questions",
                        "It looks like your browser has JavaScript turned off. JavaScript is required for this feature to work.",
                        "Please enter a valid email address", 'Your email address Please enter a valid email address',
                        'Please enter a valid password', 'Your Monocle password Please enter a valid password',
                        "Terms and conditions", "To stop receiving all Konfekt newsletters, please click here.",
                        "Tiptoe\nLou stool", "© 2024 Monocle Back to top", "Unsubscribe from Konfekt Kompakt.",
                        "new to monocle?", "Sign up to our daily newsletters",
                        "Sign up to our weekly Saturday newsletter",
                        "This email is from Konfekt whose registered office is at Dufourstrasse 90, Zürich. You have received this email because you have previously provided us with your email address and subscribed to Konfekt newsletters. © 2023 Konfekt.",
                        "Tokyo\n+81 (0)3 6407 0845", "Unsubscribe from Konfekt Kompakt.",
                        "Collect.Studio\nWaan Nozo bowl", "Comme des Garçons\nCandle One: Hinoki",
                        "Comme des Garçons\nScent Three: Sugi", "Farmers’\nWelsh Lavender Foot Cream",
                        "Leuchtturm1917\nWeekly diary 2024", "London\n+44 207 486 8770",
                        "Monocle\nThe Monocle Book of Japan", "Monocle\nCotton twill cap",
                        "Monocle magazine March 2024", "OLO\nCalavria roll-on fragrance", "Rier\nFleece overshirt",
                        "Porter\nBackpack with detachable pouch", "Sargadelos\nMeigallo cobalt vase",
                        "Zürich\n+41 44 368 70 01", "Email e Sign up Invalid email",
                        "For the best experience with monocle.com, please ensure that your browser has Javascript enabled.",
                        "Invalid email", "Laperruque\nNeck pouch", "Monocle email newsletters",
                        "Sign up to Monocle’s email newsletters to stay on top of news and opinion, plus the latest from the magazine, radio, film and shop.",
                        "This email is from Konfekt whose registered office is at Dufourstrasse 90, Zürich. You have received this email because you have previously provided us with your email address and subscribed to Konfekt newsletters. © 2022 Konfekt.",
                        "Want more stories like these in your inbox?\nSign up to Monocle’s email newsletters to stay on top of news and opinion, plus the latest from the magazine, radio, film and shop. Email e Sign up Invalid email",
                        "Comme des Garçons\nThis internationally recognised brand, with the dual design influences of Tokyo and Paris.",
                        "Japan Collection\ncollection", "Konfekt - Issue ",
                        "Founded by Bilbao-born Gonzalo Fonseca in 2007, Steve Mono creates shoes and bags that draw on Spanish culture, reinventing everything from traditional artisan sandals to canvas totes with a functional, modern spirit.",
                        "Founded by Bilbao-born Gonzalo Fonseca in 2007, Steve Mono creates shoes and bags that draw on Spanish culture, reinventing everything from traditional artisan sandals to canvas totes with a functional, modern spirit.",
                        "Arpenteur\nContour vest", "Aton\nSafari jacket", "Burel\nLarge weekender bag",
                        "Konfekt has teamed up with a pair of like-minded brands to create two pieces – a bag and a soft cotton kaftan – to complete any sunny look.",
                        "Leuchtturm1917\n",
                        "Porter\nTokyo-made, our selection of bags have been made with the seasoned traveller in mind.",
                        "Proteca\nEquinox Light U Carry-on suitcase 34 L", "Qwstion\nHoldall bag",
                        "Getty Images"
                        ]

    if (" " not in text or text in text_filter_list) and "Subscriptions start from" in text:
        return False
    return True


def check_node_exists(text_hash):
    existing_documents = chroma_collection.get(where={"text_hash": text_hash})
    if len(existing_documents["documents"]) > 0:
        node = metadata_dict_to_node(existing_documents["metadatas"][0])
        node.set_content(existing_documents["documents"][0])
        node.get_content()
        return node


def white_space_cleaner(text):
    text = strip_text(text)
    return re.sub(r"(\r|\n){2,}", "\n", text)


def get_nodes(url):
    if len(chroma_collection.get(where={"url": url})["documents"]) > 0:
        return
    if urllib3.util.parse_url(url).hostname == "www.instagram.com":
        return nodes_from_instagram(url)
    return nodes_from_html(url)


def nodes_from_instagram(url):
    text = requests.get(url).text
    xml = BeautifulSoup(text, 'lxml-xml')

    contents = [i.attrs["content"] for i in xml.find_all("meta") if i.attrs.get("property") == "og:title"]
    nodes: list[TextNode] = []
    parent_document = TextNode(
        text=" ".join(contents),
        metadata={
            "text_hash": hex_id(contents[0]),
            "url": url
        })
    for content in contents:
        result = list(re.finditer(r"(?P<Title> (?P<author>.*) on Instagram): ", content))[0]
        title = result.groupdict()["Title"]
        author = result.groupdict()["author"]
        propositions = propositional_splitter(content)
        for proposition in propositions:
            text_hash = hex_id(proposition)
            new_node = TextNode(
                text=proposition,
                metadata={
                    "text_hash": text_hash,
                    "title": title,
                    "author": author,
                    "url": url
                })
            new_node.excluded_llm_metadata_keys = ["url", "text_hash"]
            new_node.excluded_embed_metadata_keys = ["url", "text_hash"]
            new_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_document.node_id)
            parent_document.relationships[NodeRelationship.CHILD] = RelatedNodeInfo(node_id=new_node.node_id)
            nodes.append(new_node)
    nodes.append(parent_document)
    index.insert_nodes(nodes)


def nodes_from_html(url, chunking_agent_llm=None):
    request = requests.get(url)
    text = request.text  # simple_json_from_html_string(request.text, use_readability=True)["content"]
    doc = BeautifulSoup(text, 'html.parser')
    parents: list[Tag | None] = []
    for p in doc.find_all('p'):
        if p.parent not in parents:
            parents.append(p.parent)

    parent_documents: list[Document] = []
    previous_chunk_summary = "N/A"
    for parent in parents:
        primary_text = parent.text
        primary_text = white_space_cleaner(primary_text)
        if " " not in primary_text:
            continue
        summary = Settings.llm.complete(f"Provide a one sentence summary of the text below:\n{primary_text}").text
        text_hash = hex_id(primary_text)
        parent_document = check_node_exists(text_hash)
        if parent_document is None:
            parent_document = Document(
                metadata={
                    "text_hash": text_hash,
                    "h1": strip_text(parent.h1.text)
                    if parent.h1
                    else None,
                    "h2": strip_text(parent.h2.text)
                    if parent.h2
                    else None,
                    "h3": strip_text(parent.h3.text)
                    if parent.h3
                    else None,
                    "url": request.url,
                    "Summary": summary,
                },
                text=primary_text,
            )
            parent_document.excluded_llm_metadata_keys = ["Summary", "url", "text_hash"]
            _result = Settings.llm.complete(
                _fmt_promt(previous_chunk_summary=previous_chunk_summary, middle_chunk_summary=summary)
            ).text.strip()
            related = _result.startswith("Yes")
            if related and parent_documents:
                parent_document.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=parent_documents[-1].node_id
                )
                parent_documents[-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=parent_document.node_id)
            previous_chunk_summary = summary

        # Sub texts
        sub_texts = parent.find_all("p")
        if len(sub_texts) > 1:
            sub_nodes: list[TextNode] = []
            for i, sub_text in enumerate(sub_texts):
                sub_text_text = sub_text.text
                sub_text_text = white_space_cleaner(sub_text_text)
                if " " not in sub_text_text:
                    continue
                text_hash = hex_id(sub_text_text)
                new_node = check_node_exists(text_hash)
                if new_node is None:
                    new_node = TextNode(
                        text=sub_text_text,
                        metadata={
                            "text_hash": text_hash,
                            "h1": strip_text(parent.chunk.h1.text)
                            if parent.chunk.h1
                            else None,
                            "h2": strip_text(parent.chunk.h2.text)
                            if parent.chunk.h2
                            else None,
                            "h3": strip_text(parent.chunk.h3.text)
                            if parent.chunk.h3
                            else None,
                            "url": request.url,
                            "Parent Document Summary": summary,
                        },
                    )
                    new_node.excluded_llm_metadata_keys = ["Parent Document Summary", "url", "text_hash"]
                    new_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_document.node_id)
                    parent_document.relationships[NodeRelationship.CHILD] = RelatedNodeInfo(node_id=new_node.node_id)
                    if i > 0:
                        with contextlib.suppress(IndexError):
                            previous_node = sub_nodes[-1]
                            previous_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                                node_id=new_node.node_id)
                            new_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                                node_id=previous_node.node_id)
                    sub_nodes.append(new_node)
            index.insert_nodes(sub_nodes)
        parent_documents.append(parent_document)
    index.insert_nodes(parent_documents)
    return


if __name__ == "__main__":
    nodes = nodes_from_instagram("https://www.instagram.com/p/C2zergot2im/?img_index=1")
    nodes
    # for url in tqdm(konfekt_newsletters.konfekt_news_letters):
    #     nodes_from_html(url)
    #
    # print(f"collection_count:{chroma_collection.count()}")
    # import datetime
    #
    # day_of_interest = datetime.date.today()
    #
    #
    # def generator():
    #     while True:
    #         yield
    #
    #
    # for _ in tqdm(generator()):
    #     year = day_of_interest.year
    #     month = day_of_interest.month
    #     day = day_of_interest.day
    #     url = f"https://monocle.com/minute/{year}/{month}/{day}/"
    #     day_of_interest -= datetime.timedelta(days=1)
    #     existing_documents = chroma_collection.get(where={"url": url})["documents"]
    #
    #     if len(existing_documents) == 0:
    #         response = requests.get(url)
    #
    #         nodes_from_html(response)
    #     else:
    #         print(f"Documents exist for {url}")
