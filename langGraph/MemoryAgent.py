from typing import TypedDict, List, Union
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
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="llama3.2:1b")

def process(state:AgentState)-> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f'AI: {response.content}')
    #print("CURRENT STATE: ", state["messages"])
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    #print(result["messages"])
    user_input = input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("My conversation Log: \n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Me: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation")
print("Conversation saved to logging.txt")