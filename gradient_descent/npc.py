from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from pydantic import BaseModel


class NPC(BaseModel):
    prompt: str
    tools: list[FunctionTool]

    def make_agent(self, llm):
        return ReActAgent.from_tools(
            tools=[],
            llm=llm,
            verbose=True,
        )
