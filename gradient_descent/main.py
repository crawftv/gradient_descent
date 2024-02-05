from llama_index import PromptTemplate
from llama_index.agent import ReActAgent
from llama_index.llms import Ollama
from llama_index.tools import FunctionTool


def _state(*args, **kwargs):
    return {
        "current room": "The players are currently in room 22A.",
        "action state": "investigate",
    }


def _help(*args, **kwargs):
    """if the player asks for help use this tool."""
    return """
    if the action state is 'investigate':
    Possible options.
     1. investigate an item in the room
     2. attack a player in the room 
     3. check inventory
     """


state = FunctionTool.from_defaults(fn=_state)
help = FunctionTool.from_defaults(fn=_help)
tools = [state, help]
llm = Ollama(model="mistral", request_timeout=30.00)

agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
)
game_warden_prompt = """
You are designed to be a game warden for the rpg Gradient descent. The vibe is industrial sci-fi future/horror. 
The main setting is 'the Deep'. an abandoned factory/spaceship like the deathstar from star wars.
 You give characters descriptions of the room they are in and respond to their action requests.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}
1. Use the state tool to determine where the players are
2. look up the room tool by name. then return that description.

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Additional Rules
- The answer MUST contain a sequence of bullet points that explain how you arrived at the answer. This can include aspects of the previous conversation history.
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""

agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(game_warden_prompt)})
if __name__ == "__main__":
    # resp = agent.chat("Where am i?")
    # print(resp)
    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("gradient_descent/the_deep").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("what is room 22a like?")
    print(response)
