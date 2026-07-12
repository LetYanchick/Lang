import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    model="zai-org/GLM-5.2:cheapest", #google/gemma-4-31B-it:
    temperature=0.1,
    top_p=0.95
)

prompt_temp = PromptTemplate(template=(''' You are a senior researcher.\n 
                                        Given topic {topic} provide exactly 3 narrow, technical subtopics.\n
                                       '''),
                             input_variables=["topic"])



chain = prompt_temp | model
response = chain.invoke({'topic': 'fitness rehabilitation'})    
print(response)