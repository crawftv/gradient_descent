from llama_index.core.agent import ReActAgent
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore, NodeRelationship
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from high_life.settings import rerank_llm, vector_store, storage_context, question_answering_llm, chroma_collection

index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

retriever = index.as_retriever()

query_engine = index.as_query_engine(similarity_top_k=10, llm=rerank_llm,
                                     node_postprocessors=[
                                         LLMRerank(
                                             llm=rerank_llm,
                                             choice_batch_size=5,
                                             top_n=5,
                                         )
                                     ],

                                     )
vector_retriever = index.as_retriever(similarity_top_k=10, llm=rerank_llm,
                                      node_postprocessors=[
                                          LLMRerank(
                                              llm=rerank_llm,
                                              choice_batch_size=5,
                                              top_n=5,
                                          )
                                      ],
                                      )
hyde = HyDEQueryTransform(include_original=True, llm=question_answering_llm)
vector_retriever_2 = index.as_retriever(similarity_top_k=10, llm=rerank_llm,
                                        node_postprocessors=[
                                            LLMRerank(
                                                llm=rerank_llm,
                                                choice_batch_size=5,
                                                top_n=5,
                                            )
                                        ],
                                        query_transform=hyde
                                        )


# hyde_query_engine = TransformQueryEngine(query_engine, query_transform=hyde)

def retrieve_node(node_id):
    result = chroma_collection.get(ids=node_id)
    node = metadata_dict_to_node(result["metadatas"][0])
    node.set_content(result["documents"][0])
    return node


def combined_high_life_search(query_str: str) -> str:
    """
    provides recommendations on travel and wine
    """
    vector_retrieved_docs = vector_retriever.retrieve(query_str)
    vector_retrieved_docs2 = vector_retriever_2.retrieve(query_str)
    _combined_docs: list[NodeWithScore] = vector_retrieved_docs + vector_retrieved_docs2
    combined_docs = {i.node_id: i for i in _combined_docs}
    _combined_docs[4].node.relationships.items()
    contents: list[str] = []
    for node in combined_docs.values():
        _contents = ""
        # go backwards
        new_node = node.node
        while new_node.relationships.get(NodeRelationship.PREVIOUS):
            next_node_id = new_node.relationships.get(NodeRelationship.PREVIOUS).node_id
            new_node = retrieve_node(next_node_id)
        _contents += new_node.get_content()
        # go forward
        while new_node.relationships.get(NodeRelationship.NEXT):
            next_node_id = new_node.relationships.get(NodeRelationship.NEXT).node_id
            new_node = retrieve_node(next_node_id)
            _contents += new_node.get_content()
        contents.append(_contents)

    return contents


def get_all_related_nodes(node: NodeWithScore):
    if new_node := node.node.relationships.get(NodeRelationship.PREVIOUS):
        return get_all_related_nodes(new_node.node_id)


from llama_index.core.tools import FunctionTool

function_tool = FunctionTool.from_defaults(fn=combined_high_life_search)

if __name__ == "__main__":
    query_str = "What can i eat in Barcelona?"
    # retrieved_documents = retriever.retrieve(query_str)
    #
    # response = query_engine.query(query_str)
    # print(response)

    # hyde_response = hyde_query_engine.query(query_str)
    # print(hyde_response)
    # print(vector_retriever.retrieve(query_str))
    #
    # docs = combined_high_life_search(query_str)
    tools = [function_tool]

    agent = ReActAgent.from_tools(tools, llm=question_answering_llm, verbose=True)
    agent.chat_repl()
    #
    # qa_prompt_tmpl_str = (
    #     "Context information is below.\n"
    #     "---------------------\n"
    #     "{context_str}\n"
    #     "---------------------\n"
    #     "Given the context information and not prior knowledge return the answers that best answer the question.\n"
    #     "Give the list in bullet points.\n"
    #     "---------------------\n"
    #     "Query: {query_str}\n"
    #     "Answer: "
    # )
    # qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    #
    # query_engine.update_prompts(
    #     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    # )
    # print(query_engine.query(query_str))
    # print(hyde_query_engine.query(query_str))
