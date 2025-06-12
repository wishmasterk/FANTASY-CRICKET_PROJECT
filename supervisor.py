from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import MemorySaver
from researcher import *
from selector import *
from fantasy_FAQ import *
from data_collector import *
from faceoff import *
from form_accessor import *

load_dotenv()

memory = MemorySaver()
config = {"configurable": {"thread_id": "fantasy123"}} # for memory

LLM = ChatOpenAI(model = "gpt-4.1")

# search tools -> will be used to search the web at times
tavily_search = TavilySearchResults(max_results = 5) # return top 5 searches -> default, can be altered
duck_search = DuckDuckGoSearchRun() 

supervisor_agent = create_supervisor(
    model = LLM,
    agents = [research_agent, data_collector_agent, player_selector_agent, faceoff_agent, form_accessor_agent, FAQ_agent],
    tools = [tavily_search, duck_search],
    output_mode = "full_history",
    prompt = (
  
    "You are a team supervisor managing three agents/teams."
     "call research_agent to get info regarding matches"
    "call data_collector_agent to get the data about players"
    "call math_agent for any calculation required by the user")
)


agent = supervisor_agent.compile(checkpointer = memory)
while True:
  inp = input("Enter: ")
  if inp.lower() == "exit":
     break
  result = agent.invoke({"messages": [{"role": "user", "content": inp}]}, config = config)
  for message in result["messages"]:
      message.pretty_print()
