import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    model="zai-org/GLM-5.2:cheapest", #google/gemma-4-31B-it:
    temperature=0.1,
    top_p=0.95
)

class Subtopic(BaseModel):
    title: str = Field(description="Technical title of reseach subtopic")
    description: str = Field(description="1-2 sentence explanation")

class SubtopicList(BaseModel):
    subtopics: List[Subtopic]

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
out: SubtopicList = chain.invoke({'topic': 'Fitness rehabilitation after lower belly surgery'})
for (i, st) in enumerate(out.subtopics):
    print(f'{i+1}: {st.title} - {st.description}')



