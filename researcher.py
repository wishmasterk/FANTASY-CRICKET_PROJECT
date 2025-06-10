from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from dotenv import load_dotenv
from typing import List, Dict, Any
import requests
import re

load_dotenv()
LLM = ChatOpenAI(model = "gpt-4.1")

# search tools -> will be used to search the web at times
tavily_search = TavilySearchResults(max_results = 5) # return top 5 searches -> default, can be altered
duck_search = DuckDuckGoSearchRun() 

# TOOLS
@tool
def match_info() -> List[Dict[str, Any]]:
    """
    Fetch a list of upcoming cricket T20 matches and their basic details.

    Steps:
    1. Query the Cricbuzz Upcoming Matches API for all upcoming matches.
    2. Filter to include only matches from the “{user demanded}” series. -> IPL (default)
    3. For each match, extract:
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
    model = LLM,
    name = "research_agent",
    tools = [match_info, additional_info, tavily_search, duck_search],
    prompt = """
    -You are a cricket research expert. 
    -Your job is to fetch out the details of the upcoming matches based on the user query.
    -You have access to a number of tools to carry out the research, and every tool does specific part of the work.
    **Tools**:
    -> match_info: It will list down some of the upcoming matches in the recent time, and it will returns a dict containing
    some specific details about that match and that too for every match. Hence returns a list of dict.
    One of the key returned by this tool is "match_id" and it is a critical information as it will be used to get the information
    from the additional_info tool.
    -> additional_info: It takes single "match_id" as input in string format and also returns a string containing the information about the 
    pitch report, weather conditions, probable playing XI's, injury/availability updates and some other additional info.
    Now from the received text, you have to extract these fields, and you have to output them in a structured format:
    - Teams: {Teams}
    - Venue: {Venue}
    - Match Status: {Match_Status}
    - Pitch Report: {Pitch_report}
    - Probable Playing XIs for both teams: {Playing_XI_both_teams}
    - Current Injuries or Player Availability Notes: {Injury_news}
    - Other important info: {add_info} -> optional
    **Note - It can happen that the string returned by this tool might not contain some of the details, which you have to return so, 
    for this you have the access to tavily_search and duck_search tool to search the web and get that inforamtion.
    - It can still happen that after doing the web_search as well, you do not got that specific details, 
    so in these cases just tell the user that this information is not available as of now, don't make up by yourself.
    **Note - During the extraction of the required fields from the text, there are hogh chances that the str does not contain information
    regarding the probable eleven players of the teams, instead it contains the squads of both the teams, so in this case you can search the web
    to either get this detail or search for the previous match of these teams individually and get the teams from there, if possible.
    But of it you don't get it, so just return the squads form both the teams and please mention that the probable XI's aren't available.
    Basically, tell the user whatever the case is. 
    **Note - The first thing after getting that text from additional_info, if it contains playing XIs or squads of the team first use tavily_search tool
    to find what is the name of the team to which the XI/squad belongs as the text does not contain this.
    
    -Use tools based on the user query, if you feel that you have to use both the tools to meet the then use both, if you feel that the user
    wants some specific details then choose on your own, but make sure to use the additional_info tool you have to first use the match_info
    as it uses the match_id returned by this tool.

    -Please verify the information before you send it to the user, and make sure that the query is addressed.
    If there is any discrepancy please do not share that information, only the ones which u are confident with it, just send that, 
    **example - if you fetch playing XI from site and if some players names are there in both team, then there is some error for sure so just return the squads.
    -Also cross-check the probable-playing XI's with their team names, it should not happen that you give playing XI of a team with other team name
    and vice versa so make sure of that.
    """
    )

"""
result = research_agent.invoke({"role": "user", "content": "give me complete match details for upcoming match of Chhattisgarh Cricket Premier League 2025."})
for mes in result["messages"]:
    mes.pretty_print()"""