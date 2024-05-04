import datetime

from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.llms.ollama import Ollama

from prompts import accumulated_prompt
from search import hyde_vector_retriever

query_str = "i need some restaurants or places to eat in Switzerland?"

llm = Ollama(model="llama3-gradient", request_timeout=5500, additional_kwargs={"num_ctx": 32_000}, temperature=0.1)

# simple_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
# llm = Ollama(model="mistral", temperature=0.1, request_timeout=500)
# query_engine = index.as_query_engine(similarity_top_k=20, llm=llm, )

# docs = retrieve_documents(query_str)
docs: list[NodeWithScore] = hyde_vector_retriever.retrieve(query_str)
answer_content = "<documents>"
for index, node in enumerate(docs):
    _answer_content = f"\n<document {index}:\n>"
    _answer_content += f" {node.get_content(metadata_mode=MetadataMode.LLM)}"
    answer_content += f"{_answer_content}\n</document {index}>"
answer_content += "\n</documents>"
final_prompt = accumulated_prompt.format(
    query_str=query_str,
    docs=answer_content
)
if __name__ == "__main__":
    # query_engine.query("I need a bulleted list of recommended central european wines with a description", )
    start = datetime.datetime.now()
    r = (llm.complete(final_prompt))
    # r = query_engine.query("Summer wine recommendations?", )
    end = datetime.datetime.now() - start
    print(r)
    print(end)

# from datetime import datetime
# start = datetime.now()
# r = query_engine.query("I need a bulleted list of recommended central european wines with a description",stream=True).response
# end = datetime.now() -start
# print(end,r)
