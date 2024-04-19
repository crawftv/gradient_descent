import os
import re
import time

import requests
from bs4 import BeautifulSoup
from llama_index.core import PromptTemplate
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, Document
from llama_index.llms.ollama import Ollama
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from propisitional_splitter import propositional_splitter
from search import index
from settings import hex_id


def get_instagram_text(url: str):
    # Set up the webdriver (make sure you have the appropriate driver installed)
    driver = webdriver.Chrome()  # For Chrome, you can use webdriver.Firefox() for Firefox, etc.

    # Navigate to the Instagram login page
    driver.get("https://www.instagram.com/accounts/login/")
    # Wait for the login form to be present
    username_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "username"))
    )

    # Enter your Instagram credentials
    username_field.send_keys(os.getenv("INSTAGRAM_USERNAME"))
    password_field = driver.find_element(By.NAME, "password")
    password_field.send_keys(os.getenv("INSTAGRAM_PASSWORD"))

    # Submit the login form
    login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
    login_button.click()

    # Wait for the login process to complete and the feed to load
    time.sleep(5)  # Adjust the delay as needed

    # Navigate to the desired webpage
    driver.get(url)

    # Wait for the page to load completely
    time.sleep(5)  # Adjust the delay as needed
    page = driver.page_source
    driver.quit()
    # Get the full page source
    return page


def filter_instagram_by_url(url):
    profile_pattern = r'^https?://(?:www\.)?instagram\.com/([a-zA-Z0-9_\.]+)/?$'
    post_pattern = r'^https?://(?:www\.)?instagram\.com/p/([a-zA-Z0-9_\-]+)/?'

    profile_match = re.match(profile_pattern, url)
    post_match = re.match(post_pattern, url)

    if profile_match:
        return profile_nodes_from_html_file(url)
    elif post_match:
        return nodes_from_instagram(url)
    else:
        return "The URL is not a valid Instagram profile or post URL."


def profile_nodes_from_html_file(url):
    text = get_instagram_text(url)
    html = BeautifulSoup(text, 'html.parser')

    stack = list(html.find("header").children)
    bio_texts = []
    while stack:
        x = stack.pop(0)
        try:
            children = list(x.children)
            stack = children + stack
        except:
            bio_texts.append(x.text)
    prompt = PromptTemplate(
        "the text provided comes from an instagram bio. Please reconstruct a biographical sentence to be included with other pieces of information as part of a data application:\n{bio_text}")
    bio = Ollama(model="mistral", temperature=0, request_timeout=500, ).complete(prompt.format(bio_text=bio_texts)).text

    body = html.find("header").next_sibling
    new_nodes = []
    texts: list[str] = []
    while body:
        image_tags = body.find_all("img")
        for i in image_tags:
            text = i.get("alt") or ""
            texts.append(text)
            current_tag = i
            post_url = ""
            while current_tag.parent is not None:
                if current_tag.get("href"):
                    post_url = current_tag.get("href")
                    break
                else:
                    current_tag = current_tag.parent

            text_hash = hex_id(text)
            new_node = TextNode(
                text=text,
                metadata={
                    "text_hash": text_hash,
                    "url": f"{url}{post_url.lstrip('/')}",
                    "bio": bio
                })
            new_node.excluded_llm_metadata_keys = ["url", "text_hash"]
            new_node.excluded_embed_metadata_keys = ["url", "text_hash"]
            new_nodes.append(new_node)
        body = body.next_sibling
    # build parent doc
    parent_texts = "\n".join(texts)
    parent_document = Document(
        metadata={
            "url": url,
            "bio": bio,
            "text_hash": hex_id(parent_texts)
        },
        text=parent_texts
    )
    parent_document.excluded_llm_metadata_keys = ["url", "text_hash"]
    for new_node in new_nodes:
        parent_document.relationships[NodeRelationship.CHILD] = RelatedNodeInfo(node_id=new_node.node_id)
        new_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_document.node_id)

    new_nodes.append(parent_document)
    for node in new_nodes:
        index.insert_nodes([node])
    return "Success"


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


if __name__ == "__main__":
    resp = filter_instagram_by_url("https://www.instagram.com/holybellycafe/")
