import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

@tool('get_weather', description='Return weather information for a given city')
def get_weather(city:str):
    return f'The weather in {city} is perfect, just as usual.'

model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    model="google/gemma-4-31B-it:cheapest", 
    temperature=0.1,
    top_p=0.95
)

agent = create_agent(model, tools=[get_weather], system_prompt='You are a helpful weather assistant, who likes telling jokes')


result = agent.invoke({
    "messages": [
            {"role": "user", "content": "What is the weather like in Saint-Petersburg?"}
        ]
})

print(result)
print(result['messages'][-1].content)




