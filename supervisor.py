from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import MemorySaver
from researcher import *
from selector import *
from fantasy_FAQ import *
from data_collector import *

memory = MemorySaver()
LLM = ChatOpenAI(model = "gpt-4.1")

@tool
def add(a: int, b: int) -> int:
   """it adds two numbers"""
   return a + b

@tool
def multiply(a : int, b: int) -> int:
   """it hultiplies two numbers"""
   return a * b

math_agent = create_react_agent(
   model = LLM,
   tools = [add, multiply],
   prompt = "you are an math expert",
   name = "math_agent"
)

supervisor_agent = create_supervisor(
    model = LLM,
    agents = [research_agent, data_collector_agent, math_agent],
    output_mode = "full_history",
    prompt = (
  
    "You are a team supervisor managing three agents/teams."
     "call research_agent to get info regarding matches"
    "call data_collector_agent to get the data about players"
    "call math_agent for any calculation required by the user")
)

config = {"configurable": {"thread_id": "demo123"}}
agent = supervisor_agent.compile(checkpointer = memory)
while True:
  inp = input("Enter: ")
  if inp.lower() == "exit":
     break
  result = agent.invoke({"messages": [{"role": "user", "content": inp}]}, config = config)
  for mes in result["messages"]:
      mes.pretty_print()


#result = research_agent.invoke({"role": "user", "content": "tell me about the match detaials of the upcoming match of tamil league."})
#for mes in result["messages"]:
#    mes.pretty_print()