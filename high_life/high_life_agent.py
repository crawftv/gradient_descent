from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent, ReActChatFormatter
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank

from high_life.settings import vector_store, storage_context

index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

retriever = index.as_retriever()

query_engine = index.as_query_engine(similarity_top_k=10,
                                     node_postprocessors=[LLMRerank(choice_batch_size=5, top_n=5, )], )
vector_retriever = index.as_retriever(similarity_top_k=10,
                                      node_postprocessors=[
                                          LLMRerank(
                                              choice_batch_size=5,
                                              top_n=5,
                                          )
                                      ],
                                      )

# hyde_query_engine = TransformQueryEngine(query_engine, query_transform=hyde)


# function_tool = FunctionTool.from_defaults(fn=search)

_high_life_prompt = """

You are designed to help with travel, food, and wine recommendations using the provided tool. 
## Tools
You have access to one tool. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require retrying the query multiple times to find the right answer

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.
Please Always use the tool.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information to answer the question without using any more tools.
At that point, you MUST respond in the one of the following three formats:


```
Thought: I can answer without using any more tools. I need to check each possible answer for relevance.
Answer: [provide a detailed summary explaining your answer. include any relevant web links included in the text. use bullet points. ]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""
high_life_prompt = PromptTemplate(_high_life_prompt)

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

    agent = ReActAgent.from_tools(tools,
                                  verbose=True,
                                  react_chat_formatter=ReActChatFormatter(system_header=_high_life_prompt)
                                  )
    agent.update_prompts({"agent_worker:system_prompt": high_life_prompt})

    resp = agent.chat("do you have recommendations on where to stay in the south of france?")
    print(resp)
    # agent.chat_repl()
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
