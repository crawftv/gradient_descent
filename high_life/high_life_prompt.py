from llama_index.core import PromptTemplate, SelectorPromptTemplate, ChatPromptTemplate, Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole, CompletionResponse
from llama_index.core.prompts import PromptType
from llama_index.core.prompts.utils import is_chat_model
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.llms.ollama import Ollama

from high_life_agent import search, hyde_vector_retriever

high_life_prompt = PromptTemplate("""
You are a qa bot designed to answer to help with travel, food, and wine recommendations using the provided tool. 
Use only the context provided below to answer the question.
-----------------
{context_str}
-----------------
"Given this information, please answer the question: {query_str}\n"

""")
MODIFIED_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, add the new answer to  the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the just  original answer.\n"
    "The out will be the form of \n"
    "Refined Answer: "
)
# the same general context or topics and people of interest
MODIFIED_REFINE_PROMPT = PromptTemplate(
    MODIFIED_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)

MODIFIED_TEXT_QA_PROMPT_TMPL = (
    "A potential answer to a query is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context or topic and people of interest in the query and not prior knowledge, "
    "determine if the query can be answered, with respect to same general context or topic and people of interest. "
    "If the context answers the question, return the answer, if not simply return 'Empty Response'\n"
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or "
    "Query: {query_str}\n"
    "Answer: ")
MODIFIED_QA_PROMPT = ChatMessage(content=MODIFIED_TEXT_QA_PROMPT_TMPL, role=MessageRole.SYSTEM, )

MODIFIED_TEXT_QA_PROMPT = PromptTemplate(MODIFIED_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)
MODIFIED_TEXT_TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    role=MessageRole.SYSTEM,
)
TEXT_QA_PROMPT_TMPL_MSGS = [MODIFIED_TEXT_TEXT_QA_SYSTEM_PROMPT,
                            ChatMessage(content=MODIFIED_TEXT_QA_PROMPT_TMPL, role=MessageRole.USER, ), ]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

MODIFIED_TEXT_QA_SYSTEM_PROMPT_SEL = SelectorPromptTemplate(
    default_template=MODIFIED_TEXT_QA_PROMPT,
    conditionals=[(is_chat_model, CHAT_TEXT_QA_PROMPT)],
)

prompt = PromptTemplate(""" You are the world's best recommendation bot designed to answer user questions about food, travel, wine, etc.
    We have provided you a document to answer a query from an search embedding search.
    Your job is to determine whether the subject of document answers the query given the proper keywords or topics.
    if some asks for a red wine rec, only affirm red wines.
    if some asks about a city make sure the city is correct in the given document.
    QUERY: {query_str}
    ------------ Document ------------
    {doc}
    ----------------------------------
    ------------Output format------------
    Summary: Give a summary of the document with respect to the query.
    Resolves Query: True/False 
    """)

ANSWERS = list[tuple[tuple[str, list[NodeWithScore]], CompletionResponse]]


def filter_answer(answer: tuple[tuple[str, list[NodeWithScore]], CompletionResponse]):
    if "resolves query: true" in answer[1].text.strip().lower():
        return answer[0][0]


def master_query(query_str: str) -> str:
    texts = search(query_str)
    answers: ANSWERS = []
    for nodes in texts.values():
        resp = Settings.llm.complete(
            prompt.format(query_str=query_str,
                          doc=" ".join([node.get_content(metadata_mode=MetadataMode.LLM) for node in nodes])))
        answers.append(resp)
    answers = list(zip(texts.items(), answers))

    final_answers: ANSWERS = list(filter(lambda x: filter_answer(x), answers))

    p = PromptTemplate("""You are the worlds best Q&A bot, specializing in synthesizing correct answers to questions.
    You will be given a knowledge base that has been proven to answer the user query.
    You're job is to extract the text that best answers the query and give it to the user.
    -----------------
    KNOWLEDGE_BASE: {docs}
    -----------------
    QUERY: {query_str}
    -----------------
    <<<RESPONSE INSTRUCTIONS: If the documents do not contain a suitable answer, simply respond: "i could not find a suitable answer".
    Do NOT suggest a solution from your own knowledge.
    Do NOT include phrases like 'Based on the provided documents, or 'According to the documents'. or  'based on the documents provided.'
    DO NOT
    Make sure to include the url from the metadata for each document in the respected answer.>>>>
    -----------------
    Response: [Put your response here and include your reasoning]
    """)

    answer_content = "".join(
        "".join(
            [
                node.get_content(metadata_mode=MetadataMode.NONE)
                for node in node_list
            ]
        )
        for (id, node_list), completion_response in final_answers
    )
    p = p.format(
        query_str=query_str,
        docs=answer_content
    )
    resp = Ollama(model="mistral", temperature=0, request_timeout=500).complete(p)
    return resp.text


if __name__ == "__main__":
    query_str = (
        "Can you give me a recommendation for a wine to eat with Salmon. I would prefer one that is not italian or french.")
    # query_str = "Can you give me a recommendation for a place to stay in Antartica?"
    v = hyde_vector_retriever.retrieve(query_str)
    texts = search(query_str)
    prompt = high_life_prompt.partial_format(context_str=search(query_str))
    resp = Settings.llm.complete(prompt.format(query_str=query_str))
    # print(resp)
    # summarizer = get_response_synthesizer(
    #     verbose=True,
    #     structured_answer_filtering=True,
    #     response_mode=ResponseMode.REFINE,
    #     refine_template=MODIFIED_REFINE_PROMPT,
    #     text_qa_template=CHAT_TEXT_QA_PROMPT,
    # )
    # resp = summarizer.get_response(
    #     query_str,
    #     text_chunks=[
    #         f"Document {index}: {text}"
    #         for index, text in enumerate(texts.values())])
    # print("ANSWER" * 20)
    # print(resp)
    # from llama_index.core.response_synthesizers import Refine
    #
    # summarizer = Refine(verbose=True)
    # print(summarizer.get_response(query_str, texts))
    # texts = "\n".join(texts.values())
