from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """This is an addition function that adds two numbers"""
    return a + b

@tool
def substract(a:int, b:int):
    """This is a substraction function"""
    return a - b

@tool
def multiply(a:int, b:int):
    """This is a multiplication function"""
    return a * b

tools = [add, substract, multiply]

model = ChatOllama(model="llama3.2:1b").bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    response = model.invoke(["You are my AI assistant, please try your best."] + state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState)->bool:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return False
    else:
        return True
    
graph = StateGraph(AgentState)
graph.add_node("TheAgent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("TheAgent")
graph.add_conditional_edges("TheAgent", should_continue, {True: "tools", False: END})
graph.add_edge("tools", "TheAgent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 then multiply 12 * 12 and tell me both results")]}
print_stream(app.stream(inputs, stream_mode="values"))