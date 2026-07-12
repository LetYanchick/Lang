import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    model="google/gemma-4-31B-it:cheapest", 
    temperature=0.1,
    top_p=0.95
)

# conversation = [
#     SystemMessage("You are a helpful assistant for questions regarding programming"),
#     HumanMessage('What is Python?'),
#     AIMessage('Python is an interpreted programming language.'),
#     HumanMessage('When was is released?')
# ]

for chunk in model.stream("What is Python?"):
    print(chunk.text, end='', flush=True)

# print(response)
# print(response.content)




