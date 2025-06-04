from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union, Any, Tuple
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
import requests
import re

load_dotenv()

# Load the LLM
LLM = ChatOpenAI(model = "gpt-4.1")

# Search tools -> will be used to search the web at times
tavily_search = TavilySearchResults() 
duck_search = DuckDuckGoSearchRun() 

@tool
def match_info() -> List[Dict[str, Any]]:
    """
    Fetch a list of upcoming IPL matches and their basic details.

    Steps:
    1. Query the Cricbuzz Upcoming Matches API for all upcoming matches.
    2. Filter to include only matches from the “Indian Premier League” series.
    3. For each IPL match, extract:
       - Match ID
       - Match Description (e.g., "Qualifier 1")
       - Teams (formatted as "Team A vs Team B")
       - Match Status (e.g., "Match yet to start")
       - Venue ID
       - Venue Name and City

    Returns:
    --------
    List[Dict[str, Any]] where each dict has:
      {
        "Match ID": str or int,
        "Match Desc": str,
        "Teams": str,
        "Status": str,
        "Venue ID": str,
        "Venue": str
      }
    Raises:
    -------
    Exception if the API request fails or returns a non‐200 status code.
    """
    url = "https://Cricbuzz-Official-Cricket-API.proxy-production.allthingsdev.co/matches/upcoming"
    headers = {
        'x-apihub-key': '9HN92wz6l7bberNNuKkhDCXeb4YH4lXo2fIKuVdgCpB82jpHlM',
        'x-apihub-host': 'Cricbuzz-Official-Cricket-API.allthingsdev.co',
        'x-apihub-endpoint': '1943a818-98e9-48ea-8d1c-1554e116ef44'
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}")
    
    data = response.json()
    ipl_match_list = []

    for type_match in data.get("typeMatches", []):
        for series_match in type_match.get("seriesMatches", []):
            series = series_match.get("seriesAdWrapper", {})
            if "Indian Premier League" in series.get("seriesName", ""):
                for match in series.get("matches", []):
                    match_info = match.get("matchInfo", {})
                    match_id = match_info.get("matchId")
                    match_desc = match_info.get("matchDesc")
                    match_status = match_info.get("status")
                    team1 = match_info.get("team1", {}).get("teamName", "Team 1")
                    team2 = match_info.get("team2", {}).get("teamName", "Team 2")
                    venue = match_info.get("venueInfo", {})
                    venue_id = venue.get("id", "Unknown ID")
                    ground = venue.get("ground", "Unknown Ground")
                    city = venue.get("city", "Unknown City")

                    ipl_match_list.append({
                        "Match ID": match_id,
                        "Match Desc": match_desc,
                        "Teams": f"{team1} vs {team2}",
                        "Status": match_status,
                        "Venue ID": venue_id,
                        "Venue": f"{ground}, {city}"
                    })
    
    return ipl_match_list


@tool
def additional_info(match_id: str) -> str:
    """
    Fetch detailed commentary information for a given match ID to derive pitch report,
    probable playing XI, injuries, and ground records (if available in commentary).
    
    Steps:
    1. Call the Cricbuzz Commentary API for the specified match ID.
    2. Concatenate all 'commText' entries from 'commentaryList'.
    3. Clean the concatenated text by:
       - Removing markers like “B0$”, “B1$”, “B14$” (any B<number>$).
       - Replacing escaped newlines (“\\n”) with spaces.
       - Collapsing multiple whitespace characters into a single space.
    4. Return the cleaned commentary string.

    Input:
    ------
    match_id : str
      The unique ID of the match (from match_info tool).

    Returns:
    --------
    str : Cleaned, concatenated commentary text containing pitch details, probable XI mentions, injuries, etc.

    Raises:
    -------
    Exception if the commentary API returns a non‐200 status code.
    """
    url = f"https://Cricbuzz-Official-Cricket-API.proxy-production.allthingsdev.co/match/{match_id}/commentary"
    headers = {
        'x-apihub-key': '9HN92wz6l7bberNNuKkhDCXeb4YH4lXo2fIKuVdgCpB82jpHlM',
        'x-apihub-host': 'Cricbuzz-Official-Cricket-API.allthingsdev.co',
        'x-apihub-endpoint': '8cb69a0f-bcaa-45b5-a016-229a2e7594f6'
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to get commentary: {response.status_code}")

    data = response.json()
    full_text = ""

    for item in data.get("commentaryList", []):
        full_text += item.get("commText", "") + " "

    # Remove any markers like B<number>$
    cleaned = re.sub(r'\s*B\d+\$', '', full_text)
    # Remove escaped newlines and extra spaces
    cleaned = cleaned.replace("\\n", " ")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


# Researcher Agent: gathers match details (teams, venue, pitch, injuries, probable XI)
research_agent = create_react_agent(
    model=LLM,
    name="researcher",
    tools=[match_info, additional_info, tavily_search, duck_search],
    prompt="""
You are a cricket expert assistant. When asked about an upcoming IPL match:

1. Use the match_info tool to list all upcoming IPL matches, then identify the specific match (by ID) requested by the user.
2. Use the additional_info tool with that match ID to retrieve commentary text containing pitch report, probable playing XI, injury updates, and venue records.
3. If any detail is missing or unclear, use tavily_search or duck_search to locate supplementary information.
4. Present all gathered details in a clear, structured format:
   - Match ID, Teams, Venue, Match Status
   - Pitch Report
   - Probable Playing XIs for both teams
   - Current Injuries or Player Availability Notes
   - Any relevant venue history or records

Always verify correctness; if you cannot find a particular field (e.g. “pitch report”), state “Information not found” rather than guessing.
"""
)
