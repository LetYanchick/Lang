import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base_prompt = 'You are a heplful assistant.'
    match user_role:
        case 'expert':
            return f'{base_prompt} Provide detailed technical responses.'
        case 'beginner':
            return f'{base_prompt} Keep your explanations simple.'
        case 'child':
            return f'{base_prompt} Explain everything as if you are talking to a five-year old.'
        case _:
            return base_prompt
        

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    model="google/gemma-4-31B-it:cheapest", 
    temperature=0.1,
    top_p=0.95
)
        
agent = create_agent(model, middleware=[user_role_prompt], context_schema=Context)

response = agent.invoke({
    'messages': [{'role': 'user', 'content': 'Explain OSI.'}]
}, context= Context(user_role='child'))

print(response['messages'][-1].content)
