import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Dict, Any
load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    model="zai-org/GLM-5.2:cheapest", #google/gemma-4-31B-it:
    temperature=0.1,
    top_p=0.3
)

class Subtopic(BaseModel):
    title: str = Field(description="Technical title of reseach subtopic")
    description: str = Field(description="1-2 sentence explanation")

class SubtopicList(BaseModel):
    subtopics: List[Subtopic]

class State(Dict[str, Any]):
    topic: str
    subtopics: List[Subtopic]
    approval: str

parser = PydanticOutputParser(pydantic_object=SubtopicList)

def generate_subtopics(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PromptTemplate(template=('''{format_instructions}\n
                            You are a senior researcher.\n 
                            Given topic {topic} provide exactly 3 narrow, technical subtopics.\n
                            Output JSON object: {{\"subtopics\": [{{\"title\":\"...\", \"description\":\"}}, ...]}}\n
                            '''),
                            input_variables=["topic"],
                            partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | model | parser
    out: SubtopicList = chain.invoke({'topic': state["topic"]})
    return {"subtopics": out.subtopics}


def ask_approval(state:Dict[str, Any]) -> Dict[str, Any]:
    print("Possible subtopics:\n")
    for (i, st) in enumerate(state["subtopics"], 1):
        print(f'{i}. {st.title} - {st.description}\n')
    ans = input("Are you fine with these? (yes/no)\n> ")
    return {"approval": ans.strip().lower()}

builder = StateGraph(State)
builder.add_node("generate", generate_subtopics)
builder.add_node("approve", ask_approval)

builder.add_edge(START, "generate")
builder.add_edge("generate", "approve")

def route(state:Dict[str, Any]) -> Dict[str, Any]:
    return "SUCCESS" if state.get("approval", '').startswith('y') else "GenerateAgain"

builder.add_conditional_edges("approve", route, {"GenerateAgain": "generate", "SUCCESS": END})

graph = builder.compile()
graph_png = graph.get_graph(xray=True)
png_bytes = graph_png.draw_mermaid_png()
with open('graph.png', 'wb') as f:
    f.write(png_bytes)

if __name__ =="__main__":
    final_state = graph.invoke({'topic': 'Fitness rehabilitation after lower belly surgery'})
    print("Final subtopics: \n")
    for st in final_state["subtopics"]:
        print(f'{st.title}: {st.description}\n')




