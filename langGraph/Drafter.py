from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content:str)->str:
    """Updates the document with the provided content"""
    global document_content
    document_content = content
    return f'Document has been updated successfully! The current content is: \n{document_content}'

@tool
def save(filename:str)->str:
    """Saves the current document to a text file and finishes the process
    Args: 
        filename: Name for the text file.
    """
    global document_content
    if not filename.endswith(".txt"):
        filename = f'{filename}.txt'

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f'\nDocument has been saved to {filename}')
        return f'Document has been saved successfully to {filename}.'
    except Exception as e:
        return f'Error saving document: {str(e)}'
    
tools=[update, save]

model = ChatOllama(model="llama3.2:1b").bind_tools(tools)

def the_agent(state:AgentState)->AgentState:
    if not state["messages"]:
        user_input = input("I'm ready to help you to update a document. What would you like to create?")
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"\nUser: {user_input}")
        user_message = HumanMessage(content=user_input) 
    system_prompt = SystemMessage(content=f"""You are a Drafter, a helpfull writing assistant. You are going to update and modify documents."
    If the user wants to upddate or modify content, use 'update' tool with the complete updated content."
    If the user wants to save and finish, you need to use the 'save' tool."
    Always show the current document state after modifications."
    The current document content is {document_content}""")

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f'\n AI: {response.content}')
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f'USING TOOLS: {[tc["name"] for tc in response.tool_calls]}')

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state:AgentState)->bool:
    """Determines if we should continue or end the conversation."""
    messages = state["messages"]
    if not messages:
        return True
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content:
            return False
    return True

graph = StateGraph(AgentState)
graph.add_node("agent_node", the_agent)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("agent_node")
graph.add_edge("agent_node", "tool_node")
graph.add_conditional_edges("tool_node", should_continue, {True: "agent_node", False: END})
app = graph.compile()

def print_messages(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\nTOOL RESULT: {message.content}")

def run_document_agent():
    print("\n ===== DRAFTER =====")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()

