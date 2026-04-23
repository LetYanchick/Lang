from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
#from langchain_openai import ChatOpenAI
#from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

#os.environ["DEEPSEEK_API_KEY"] = "sk-d40b0663d53f47ca9a909ce8721b2bae"
#"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" open router
#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
#llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))

llm = ChatOllama(model="llama3.2:1b")

def process(state:AgentState)->AgentState:
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != 'exit':
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")