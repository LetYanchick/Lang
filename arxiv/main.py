import operator
import os
import re
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal, TypedDict
import arxiv
load_dotenv()

# model = ChatOpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.getenv("HF_TOKEN"),
#     model="google/gemma-4-31B-it:cheapest", #:
#     temperature=0.1,
#     top_p=0.3
# )

model = OllamaLLM(model="qwen3:1.7b")

class Subtopic(BaseModel):
    title: str = Field(description="Technical title of reseach subtopic")
    description: str = Field(description="1-2 sentence explanation")

class SubtopicList(BaseModel):
    subtopics: List[Subtopic]

class ResearchPaper(BaseModel):
    title: str = Field(default="Untitled paper", description="Title of research paper")
    authors: str = Field(default="Unknown authors", description="Comma-separated author names")
    abstract: str = Field(default="", description="Abstract content")
    url: str = Field(default="#", description="ArXiv URL")
    published: str = Field(default="Unknown date", description="Publication date")

class ResearchResult(BaseModel):
    subtopic: Subtopic
    papers: List[ResearchPaper]
    research_gap: str

class AgentState(TypedDict):
    topic: str
    subtopics: List[Subtopic]
    approval: Literal['pending', 'approved', 'rejected']
    research_results: Annotated[List[ResearchResult], operator.add]

parser = PydanticOutputParser(pydantic_object=SubtopicList)

def fetch_arxiv_papers(query:str) -> List[ResearchPaper]:
    client = arxiv.Client()
    search = arxiv.Search(query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for res in client.results(search):
        authors = ", ".join(author.name for author in res.authors)
        papers.append(ResearchPaper(title=res.title, authors=authors, abstract=res.summary, url=res.entry_id, published=str(res.published.date())))
    return papers

def generate_subtopics(state: AgentState) -> AgentState:
    try:
        parser = PydanticOutputParser(pydantic_object=SubtopicList)
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
        return {"subtopics": out.subtopics, "approval": "pending"}
    except Exception as e:
        print(f"Error in 'generate_subtopics' func: {str(e)}")

def generate_descriptions(titles: List, main_topic) -> AgentState:
    personal_topics = []
    try:
        parser = PydanticOutputParser(pydantic_object=SubtopicList) #
        prompt = PromptTemplate(template=('''{format_instructions}\n
                                            You are a senior researcher.\n 
                                            Given 3 small topics: {titles} from main topic: {main_topic} provide a 1-2 sentence description for each of them.\n
                                            Output JSON object with given titles and generated descriptions: {{\"subtopics\": [{{\"title\":\"...\", \"description\":\"}}, ...]}}\n
                                        '''),
                                input_variables=["titles", "main_topic"],
                                partial_variables={"format_instructions": parser.get_format_instructions()} #
                                )
        chain = prompt | model | parser
        out: SubtopicList = chain.invoke({'titles': titles, 'main_topic': main_topic})
        return {"subtopics": out.subtopics, "approval": "approved"}
        
    except Exception as e:
        print(f"Error in 'generate_descriptions' func: {str(e)}")


def ask_approval(state:AgentState) -> AgentState:
    try:
        print("Possible subtopics:\n")
        for (i, st) in enumerate(state["subtopics"], 1):
            print(f'{i}. {st.title} - {st.description}\n')
        ans = input("Are you satisfied with these topics or would you like to set your's? (yes/no/edit)\n> ").strip().lower()
        if ans.startswith('y'):
            return {"approval": "approved"}
        elif (ans.startswith('n')):
            return {"approval": "rejected"}
        else:
            print("\nEnter your own topics, one per line, empty line to finish\n")
            personal_topics = []
            titles = [] 
            for i in range(3):
                while (True):
                    title = input(f"Title {i+1}: ".strip())
                    if title:
                        titles.append(title)
                        break
                    else:
                        title = input(f"You should input something!\n")    
            return generate_descriptions(titles, state["topic"])
    except Exception as e:
        print(f"Error in 'ask_approval' func: {str(e)}")
        return {"approval": "rejected"}
    
def analyze_papers(papers:List[ResearchPaper], subtopic:str) -> str:
    try:
        papers_info = "\n\n".join([
            f'Paper {i+1}: {p.title}\n'
            f'Abstract: {p.abstract[:500]}...\n'
            f'Published: {p.published}'
            for i, p in enumerate(papers)
            ])
        gap_prompt = PromptTemplate(input_variables=["subtopic", "papers_info"], 
                                    template=("Analyze these three research papers on {subtopic}: \n"
                                    "{papers_info}\n"
                                    "Write a complete text summary of the main ideas behind these researches considering" 
                                    "possible limitations and disadvantages of these studies.\n"
                                    "Include references to specific papers."
                                    "Optimaly the summary should consist of 10-20 sentences."
                                    ))
        gap = (gap_prompt | model).invoke({"subtopic": subtopic, "papers_info": papers_info})
        return gap
    except Exception as e:
       print(f"Error in 'analyze_papers' func: {str(e)}") 
       return "error"
    
def conduct_research(state: AgentState) -> AgentState:
    try:
        research_results = []
        for subtopic in state["subtopics"]:
            papers = fetch_arxiv_papers(subtopic.title)
            gap = analyze_papers(papers, subtopic.title)
            research_results.append(ResearchResult(subtopic=subtopic, papers=papers, research_gap=gap))
        return {"research_results": research_results} 
    except Exception as e:
        print(f"error in 'conduct_research' func: {str(e)}")

    
def compile_report(state: AgentState) -> AgentState:
    report = f"# Research report on {state['topic']}\n\n"
    for result in state["research_results"]:
        report += f"## {result.subtopic.title}\n"
        report += f"* {result.subtopic.description}*\n\n"
        report += "### Key Papers\n"
        for i, paper in enumerate(result.papers, 1):
            report += f"#### Paper{i}: {paper.title}\n"
            report += f"- URL: {paper.url}\n"
            report += f"- Authors: {paper.authors}\n"
            report += f"- Published: {paper.published}\n"
            report += f"- Abstract: {paper.abstract[:300]}...\n\n"
        
        report += "### Summary\n"
        report+= f"{result.research_gap}\n\n"
    with open("report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n Report saved as report.md")
    return {"report": report} #

            
builder = StateGraph(AgentState)
builder.add_node("generate", generate_subtopics)
builder.add_node("approve", ask_approval)
builder.add_node("research", conduct_research)
builder.add_node("report", compile_report)
builder.set_entry_point("generate")
builder.add_edge("generate", "approve")

def route(state:AgentState) -> str:
    if state["approval"] == "approved":
        return "SUCCESS"
    else:
        return "GenerateAgain"

builder.add_conditional_edges("approve", route, {"GenerateAgain": "generate", "SUCCESS": "research"})
builder.add_edge("research", "report")
builder.add_edge("report", END)

research_agent = builder.compile()
# research_agent_png = research_agent.get_graph(xray=True)
# png_bytes = research_agent_png.draw_mermaid_png()
# with open('research_graph.png', 'wb') as f:
#     f.write(png_bytes)

if __name__ =="__main__":
    topic = input("Enter your research topic:\n> ").strip() 
    init_state = AgentState(topic=topic, subtopics=[], approval='pending', research_results=[])
    final_state = research_agent.invoke(init_state)

#Fitness rehabilitation after lower belly surgery

#Posture and Alignment Correction
#How to not lose muscles after surgery
#Athlete's rehabilitation after surgery