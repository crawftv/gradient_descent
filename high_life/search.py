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
                                           query_transform=HyDEQueryTransform(include_original=True, )
                                           )


def retrieve_node(node_id):
    result = chroma_collection.get(ids=node_id)
    node = metadata_dict_to_node(result["metadatas"][0])
    node.set_content(result["documents"][0])
    node.get_content()
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


def gather_nodes_recursively(docs: list[NodeWithScore]) -> dict[str, list[BaseNode]]:
    """
    provides recommendations on travel and wine, etc.
    """

    contents = OrderedDict()
    for node in docs:
        new_node = node.node
        # if there is a parent node add that.
        if parent_node := new_node.relationships.get(NodeRelationship.PARENT):
            contents[parent_node.node_id] = [retrieve_node(parent_node.node_id)]
        else:
            # go backwards
            while new_node.relationships.get(NodeRelationship.PREVIOUS):
                next_node_id = new_node.relationships.get(NodeRelationship.PREVIOUS).node_id
                new_node = retrieve_node(next_node_id)
            _contents = [new_node]
            # go forward
            while new_node.relationships.get(NodeRelationship.NEXT):
                next_node_id = new_node.relationships.get(NodeRelationship.NEXT).node_id
                new_node = retrieve_node(next_node_id)
                _contents.append(new_node)
            contents[_contents[0].node_id] = _contents

    return contents

# ANSWERS = list[tuple[tuple[str, list[NodeWithScore]], CompletionResponse]]
