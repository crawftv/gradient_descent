from llama_index.core import PromptTemplate, SelectorPromptTemplate, ChatPromptTemplate, Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import PromptType
from llama_index.core.prompts.utils import is_chat_model

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
