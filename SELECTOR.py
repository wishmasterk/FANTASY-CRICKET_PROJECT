from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool
from collections import defaultdict
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union, Any, Tuple
from bs4 import BeautifulSoup
from langchain.tools import DuckDuckGoSearchRun


load_dotenv()
LLM = ChatOpenAI(model = "gpt-4.1")

def compute_score(stats: Dict[str, Dict[str, Union[int, float]]], role: str) -> float:
    """
    Calculate a performance score for a player in batting, bowling, or all‐rounder roles.

    This function combines key sub‐metrics into a single score:
      • For batters:
          - Strike Rate (SR) normalized against a 200 SR benchmark.
          - Batting Average (Avg) normalized against a 50 Avg benchmark.
          - Conversion Rate (number of 50s and 100s divided by innings).
      • For bowlers:
          - Bowling Strike Rate (balls per wicket) normalized against a 12 SR benchmark.
          - Bowling Average (runs per wicket) normalized against a 12 Avg benchmark.
          - Economy Rate (runs per over) normalized against a 6.0 Eco benchmark.

    An all‐rounder’s final score is a weighted combination of batting and bowling scores:
      - Batting all‐rounder: 70% batting score, 30% bowling score
      - Bowling all‐rounder: 30% batting score, 70% bowling score

    Returns:
        A float representing the player’s score. Pure batters and bowlers will return
        only their respective sub‐score, while all‐rounders blend both.
    """

    batting_weight = {
        'bat_sr': 0.35,
        'bat_avg': 0.45,
        'conversion': 0.20,
    }
    bowling_weight = {
        'bowl_sr': 0.30,
        'bowl_avg': 0.30,
        'eco': 0.40,
    }

    bat_score = 0.0
    bowl_score = 0.0

    # Batting calculations
    if "Batting" in stats:
        batting_details = stats["Batting"]
        bat_sr_benchmark = 200.0
        bat_avg_benchmark = 50.0

        bat_sr_score = batting_details["SR"] / bat_sr_benchmark
        bat_avg_score = batting_details["Avg"] / bat_avg_benchmark
        conversion_score = (batting_details["50s"] + batting_details["100s"]) / batting_details["Innings"]

        bat_score = (
            batting_weight["bat_sr"] * bat_sr_score +
            batting_weight["bat_avg"] * bat_avg_score +
            batting_weight["conversion"] * conversion_score
        )

    # Bowling calculations
    if "Bowling" in stats:
        bowling_details = stats["Bowling"]
        bowl_avg_benchmark = 12.0
        bowl_sr_benchmark = 12.0
        bowl_eco_benchmark = 6.0

        bowl_sr_score = bowl_sr_benchmark / bowling_details["SR"]
        bowl_avg_score = bowl_avg_benchmark / bowling_details["Avg"]
        eco_score = bowl_eco_benchmark / bowling_details["Eco"]

        bowl_score = (
            bowling_weight["bowl_sr"] * bowl_sr_score +
            bowling_weight["bowl_avg"] * bowl_avg_score +
            bowling_weight["eco"] * eco_score
        )

    role_lower = role.lower()
    if "batsman" in role_lower:
        return bat_score
    elif "bowler" in role_lower:
        return bowl_score
    else:
        # All‐rounder
        if role_lower == "batting allrounder":
            return 0.7 * bat_score + 0.3 * bowl_score
        else:
            return 0.3 * bat_score + 0.7 * bowl_score


def overall_score(player_stats_dict: Dict[str, Union[str, List[Dict[str, Any]]]]) -> float: # will do for single player
    """"""
    role = player_stats_dict["role"].lower()

    weight = {
        "recent": 0.3,
        "vs_opp": 0.15,
        "at_venue": 0.15,
        "head_2_head": 0.25,
        "pitch": 0.15
    }

    recent_dict = (player_stats_dict["stats"][0])["data"]
    vs_opp_dict = (player_stats_dict["stats"][1])["data"]
    at_venue_dict = (player_stats_dict["stats"][2])["data"]

    recent_score = compute_score(recent_dict, role)
    vs_opp_score = compute_score(vs_opp_dict, role)
    at_venue_score = compute_score(at_venue_dict, role)

    head_2_head_score = 0
    head_2_head_list = player_stats_dict["head_2_head_stats"]
    for dict in head_2_head_list:
        head_2_head_score += dict["advantage_score"]
    
    head_2_head_score  = head_2_head_score / len(head_2_head_list)

    pitch_score  = player_stats_dict["pitch_stats"]

    return (
        weight["recent"] * recent_score + 
        weight["vs_opp"] * vs_opp_score + 
        weight["at_venue"] * at_venue_score + 
        weight["head_2_head"] * head_2_head_score +
        weight["pitch"] * pitch_score
    )

@tool
def select_players(players_overall_details: 
    List[Dict[str, Union[str, List[Dict[str, Any]]]]]) -> Tuple[List[Dict[str, Union[str, List[Dict[str, Any]]]]], List[Dict[str, Any]]]:
    """xxsadvf"""

    result = []
    for player in players_overall_details:
        player["overall_score"] = overall_score(player)
        result.append({
            "name": player["name"],
            "role": player["role"],
            "overall_score": player["overall_score"]
        })
    return players_overall_details, result

# Player Selector agent
player_selector = create_react_agent(
    model = LLM,
    name = "selector",
    tools = [select_players],
    prompt = (
        "You are an agent which is phenomenal at maths. Answer the query of the user to the best you can."
    )
)