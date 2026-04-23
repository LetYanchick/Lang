from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
#from langgraph.graph.message import add_messages
from operator import add as add_messages
from langgraph.prebuilt import ToolNode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import os

load_dotenv()

llm = ChatOllama(model="llama3.2:1b", temperature=0)
embeddings = OllamaEmbeddings(model="all-minilm:22m")

pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {str(e)}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"C:\Users\Вход\Desktop\питончики\langGraph"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(documents=pages_split, embedding=embeddings, persist_directory=persist_directory, collection_name=collection_name)
    print(f"Created ChromaDB vector store!")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

@tool
def retriver_tool(query:str)->str:
    """This tool searches and returns the information from the Stock Market Performance 2024 document"""
    docs = retriever.invoke(query)
    if not docs:
        return f"I found no relevant information in the Stock Market Performance 2024 document"
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")
    return "\n\n".join(results)

tools = [retriver_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state:AgentState)->bool:
    """Checks if the last message contains tool calls"""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """You are an intellegent AI assistant who answers questions about Stock Market Performance 2024 bases on the PDF document loaded into your knowledge base.
Use the reriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dictionary = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state:AgentState)->AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}

def take_action(state:AgentState)->AgentState:
    """Executes tool calls from LLM's response."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling tool: {t["name"]} with query: {t['args'].get('query', 'No query provided')}")
        if not t['name'] in tools_dictionary:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect tool name, please retry and select tool from list of available tools."
        else:
            result = tools_dictionary[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Tools execution complete. Back to the model!")
    return {'messages':results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriver_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriver_agent", False: END})
graph.add_edge("retriver_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n  ===== RAG AGENT =====")
    while True:
        user_input = input("\nWhat is your questiion: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n  ===== ANSWER =====")
        print(result['messages'][-1].content)

running_agent()