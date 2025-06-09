from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool
from dotenv import load_dotenv
from typing import List, Dict, Union
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
import re
import os

load_dotenv()

# Load the LLM
LLM = ChatOpenAI(model = "gpt-4.1")

# Search tools -> will be used to search the web at times
tavily_search = TavilySearchResults() 
duck_search = DuckDuckGoSearchRun() 

def create_vector_store():
    """
    Build or load a FAISS vectorstore for the Fantasy Cricket Guide PDF.

    - If the FAISS directory and index file exist, load the existing index
      (avoids redundant embedding).
    - Otherwise, load the PDF, split into chunks, embed, and save the index.

    Returns:
        FAISS: A FAISS vectorstore instance ready for similarity search.
    """
    FAISS_DB_PATH = "FAISS_VECTORSTORE"

    # If embeddings already exist, skip rebuilding
    if os.path.isdir(FAISS_DB_PATH) and \
       os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss")):
        emd_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.load_local(FAISS_DB_PATH, emd_model, allow_dangerous_deserialization = True)
        return db

    # otherwise build the vectorstore
    # step 1: laod the pdf
    loader = PyPDFLoader("FANTASY CRICKET GUIDE.pdf")
    docs = loader.load() # in the document format, where each document has pafe_content and metadate

    # step 2: create chunks 
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100,
        )
    chunks = text_splitter.split_documents(docs)

    # step 3: initilaize the emb model
    emd_model = OpenAIEmbeddings(model = "text-embedding-3-small") # 3072 dimensions, model by openai

    # step 4: make emd fo the chunks and store it in the vectorstore
    db = FAISS.from_documents(chunks, emd_model)
    db.save_local(FAISS_DB_PATH)
    return db


def clean_text(text: str) -> str:
    """
    Normalize whitespace in text by collapsing all runs of whitespace
    (spaces, tabs, newlines) into single spaces and trimming.

    Args:
        text (str): The raw text to clean.
    Returns:
        str: The cleaned, single-line text.
    """
    return re.sub(r'\s+', ' ', text).strip()


@tool 
def fantasy_guide_RAG(query: str) -> Dict[str, Union[str, List[str]]]:
    """ 
    Retrieval-Augmented-Generation (RAG) tool that:

    1. Loads or creates a FAISS vectorstore containing embedded chunks of the
       Fantasy Cricket Guide PDF.
    2. Embeds the user query and performs a similarity search (top 3 results).
    3. Cleans and returns the retrieved passages in a structured dict.

    Basically it retrieves the relevant passage to the user query.
    Args:
        query (str): The user's question or search string.

    Returns:
        dict: {
            "query": <original query>,
            "retrieved_text": [<passage1>, <passage2>, <passage3>]
        }
    """
    # 1) Load or build your FAISS store
    db = create_vector_store()  

    # 2) Perform similarity search
    docs = db.similarity_search(query, k = 3) # top 3 docs will be retrieved

    # 3) Extract the content from each Document
    retrieved_texts: List[str] = [clean_text(doc.page_content) for doc in docs]

    # 4) Return structured dict
    return {
        "query": query,
        "retrieved_text": retrieved_texts
    }

Fantasy_FAQ_Agent = create_react_agent(
    model = LLM,
    name = "fantasy_expert",
    tools = [tavily_search, duck_search, fantasy_guide_RAG],
    prompt = """
    - You are Fantasy Expert, an AI assistant specialized in fantasy cricket. 
    - Your goal is to provide accurate, beginner-friendly answers about fantasy cricket rules, scoring, team selection, and strategy. 
    - You will receive a query in string format as an input and you have to generate a response based on that query.
    - To do this, follow these steps:

    1. **Use RAG First**: Call the `fantasy_guide_RAG` tool with the user query. It returns the top 3 relevant passages from the official Fantasy Cricket Guide.
    2. **Evaluate RAG Output**:
    - If the retrieved passages fully answer the query, base your response solely on those passages and the original query.
    - If the passages are insufficient or incomplete, use web search tools (`tavily_search` and `duck_search`) to gather additional authoritative information.
    3. **Web Search Tools**:
    - Use the `tavily_search` tool for deep document search when you need more context.
    - Use the `duck_search` tool for general web search and quick facts.
    4. **Compose Your Answer**:
    - Use the information from RAG and, if used, web searches and the user query to generate a final response.
    - Avoid hallucinations: if uncertain, state that you don’t know or suggest the user check official sources.
    5. **Citing Sources**:
    - When using web search results, briefly mention the source (e.g., “According to [source]…”).

    Respond thoroughly, focusing on clarity and practical guidance for beginners."""
)
"""
inputs = {"messages": [{"role": "user", "content": "is there any cap on the maximum number of foreign players I can have in my team just reply in a single or two lines"}]}
result = Fantasy_FAQ_Agent.invoke(inputs)
#for r in result['messages']:
#    print(r)
print(result["messages"][-1].content)
"""