import functools
import sqlite3
from collections import OrderedDict

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import BaseNode, NodeWithScore, NodeRelationship
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from settings import vector_store, storage_context, chroma_collection

index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
hyde_vector_retriever = index.as_retriever(similarity_top_k=20,
                                           node_postprocessors=[
                                               LLMRerank(
                                                   choice_batch_size=5,
                                                   top_n=2,
                                               ),
                                           ],
                                           vector_store_query_mode="hybrid",
                                           query_transform=HyDEQueryTransform(
                                               include_original=True, )
                                           )


@functools.cache
def retrieve_node(node_id):
    result = chroma_collection.get(ids=node_id)
    try:
        node = metadata_dict_to_node(result["metadatas"][0])
    except IndexError:
        return None
    node.set_content(result["documents"][0])
    return node


def log_retrieval(logging_id, nodes: list[NodeWithScore]):
    # Connect to or create the database
    with sqlite3.connect('logs.sqlite3') as conn:  # Replace with your database name
        cursor = conn.cursor()
        return cursor.executemany(
            "INSERT INTO vector_retrieval (logging_id, document) VALUES (?, ?)",
            [(logging_id, i.json()) for i in nodes]
        )


def retrieve_documents(query_str: str):
    return hyde_vector_retriever.retrieve(query_str)


def traverse_nodes(start_node, relationship):
    """Helper function to traverse nodes in a given direction."""
    nodes = []
    current_node = start_node
    while current_node.relationships.get(relationship):
        next_node_id = current_node.relationships.get(relationship).node_id
        next_node = retrieve_node(next_node_id)
        if not next_node:
            break
        nodes.append(next_node)
        current_node = next_node
    return nodes


def gather_nodes_recursively(docs: list[NodeWithScore]) -> dict[str, list[BaseNode]]:
    contents = OrderedDict()
    for node in docs:
        new_node = node.node
        if parent_node := new_node.relationships.get(NodeRelationship.PARENT):
            if retrieved_parent_node := retrieve_node(parent_node.node_id):
                contents[parent_node.node_id] = [retrieved_parent_node]
                continue
        if previous_nodes := traverse_nodes(
                new_node, NodeRelationship.PREVIOUS
        ):
            new_node = previous_nodes[-1]
        _contents = [new_node] + traverse_nodes(new_node, NodeRelationship.NEXT)
        contents[_contents[0].node_id] = _contents

    return contents

# ANSWERS = list[tuple[tuple[str, list[NodeWithScore]], CompletionResponse]]
