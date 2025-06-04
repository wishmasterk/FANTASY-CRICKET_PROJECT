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
    """This function just compute the overall score"""
    role = player_stats_dict["role"].lower()

    weight = {
        "recent": 0.5,
        "vs_opp": 0.25,
        "at_venue": 0.25
    }

    recent_dict = (player_stats_dict["stats"][0])["data"]
    vs_opp_dict = (player_stats_dict["stats"][1])["data"]
    at_venue_dict = (player_stats_dict["stats"][2])["data"]

    recent_score = compute_score(recent_dict, role)
    vs_opp_score = compute_score(vs_opp_dict, role)
    at_venue_score = compute_score(at_venue_dict, role)

    return (
        weight["recent"] * recent_score + 
        weight["vs_opp"] * vs_opp_score + 
        weight["at_venue"] * at_venue_score
    )

@tool
def select_players(players_overall_details: 
    List[Dict[str, Union[str, List[Dict[str, Any]]]]]) -> Tuple[List[Dict[str, Union[str, List[Dict[str, Any]]]]], List[Dict[str, Any]]]:
    """
    Compute and append an overall_score for each player, then return two lists:
    
    1. The original list of player dictionaries, each augmented with an "overall_score" key.
       - Input format for each player dict:
         {
           "name": str,
           "role": str,  # "batsman", "bowler", "batting allrounder", or "bowling allrounder"
           "is_wicketkeeper": "True"/"False",
           "is_overseas": "True"/"False",
           "batting_style": str,
           "bowling_style": str,
           "stats": [
             {"title": "last_8_innings_stats",        "data": { … }},
             {"title": "career_stats_vs_<Opposition>", "data": { … }},
             {"title": "career_stats_at_<Venue>",      "data": { … }}
           ]
         }

    2. A secondary list of simplified dictionaries for selection overview:
       [
         {
           "name": str,
           "role": str,
           "overall_score": float
         },
         ...
       ]

    Steps:
    - For each player in players_overall_details:
      1. Extract their "role" and their three "data" dictionaries under "stats".
      2. Call `overall_score(player)` (which computes scores based on recent form, vs opposition, at venue).
      3. Add the returned float under player["overall_score"].
      4. Append { "name", "role", "overall_score" } to the secondary list.

    Returns:
    ----------
    Tuple[
      List[dict],  # The input list, but now each dict has "overall_score"
      List[dict]   # Simplified list of { "name", "role", "overall_score" }
    ]
    """

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
        """
    You are “Selector”, a specialized agent whose job is to build an ideal fantasy XI by ranking and selecting players based on their computed overall_score. You have one tool available:

    • select_players(players_overall_details)
    - Input: a list of player dictionaries, each containing:
        • name (string)
        • role (string: "batsman", "bowler", "batting allrounder", or "bowling allrounder")
        • is_wicketkeeper (True/False)
        • is_overseas (True/False)
        • batting_style, bowling_style
        • stats: a list of three dicts:
            1. last_8_innings_stats → “data” of recent form
            2. career_stats_vs_<Opposition> → “data” vs opposition
            3. career_stats_at_<Venue> → “data” at that venue
    - Output: 
        1. The same list augmented with “overall_score” for each player.
        2. A simplified list of dicts: { "name", "role", "overall_score" }.

    Your task:
    1. Call the tool `select_players` with the full list of player stats. 
    2. Receive back each player’s `overall_score`. 
    3. Then, consider standard team‐building constraints (you may search the web if needed):
    - Exactly 11 players.
    - A minimum of 5 batsmen (including at least one wicketkeeper).
    - A minimum of 4 bowlers.
    - At least 1 and at most 2 all‐rounders.
    - At most 4 overseas players.
    - Remaining spots filled by best‐scoring players by role.

    4. Rank players by `overall_score` within each role category (batsman, bowler, all‐rounder).
    5. Select players in a balanced way to satisfy the constraints:
    • Pick top wicketkeeper‐batsman by score (counts toward “batsman” and “wk”).
    • Then fill remaining batting slots from highest‐scoring non‐keeper batsmen.
    • Pick all‐rounders (1–2) by descending score.
    • Fill four bowler slots from top bowlers.
    • Ensure overseas count ≤ 4; if excess, drop lowest‐scoring overseas until constraint met.
    6. For each selected player, provide a rationale:
    • Their overall_score.
    • Role and category ranking (e.g. “3rd highest among batsmen”).
    • Any special note (e.g. “Only overseas seamer with top‐4 score”).

    7. As output, return a JSON object with:
    {
        "selected_players": [
        {
            "name": "...", 
            "role": "...", 
            "overall_score": ..., 
            "rationale": "..."
        },
        …
        ],
        "reasoning": "Concise summary of how team was constructed"
    }

    If asked to compare two players, simply compute both of their overall_score values and explain which has the higher score and why, referencing recent form, venue, and opposition breakdown.

    Be concise, clear, and always justify selections by citing scores and role‐specific constraints. Use websearch only if you need to confirm official team composition rules (for example, verifying “max 4 overseas” in this league).  

            """
        )
    )

print(select_players.invoke({"players_overall_details": [
    {
      "name": "Virat Kohli",
      "role": "batsman",
      "is_wk": "False",
      "is_overseas": "False",
      "batting_style": "Right Handed Bat",
      "bowling_style": "Right-arm medium",
      "stats": [
        {
          "title": "last_8_innings_stats",
          "data": {
            "Batting": {
              "Matches": 8,
              "Innings": 8,
              "Runs": 366,
              "Balls": 246,
              "Outs": 7,
              "4s": 43,
              "6s": 9,
              "50s": 5,
              "100s": 0,
              "SR": 148.78,
              "Avg": 52.29
            }
          }
        },
        {
          "title": "career_stats_vs_Punjab_Kings",
          "data": {
            "Batting": {
              "Matches": 35,
              "Innings": 35,
              "Runs": 1116,
              "Balls": 839,
              "Outs": 31,
              "4s": 117,
              "6s": 33,
              "50s": 6,
              "100s": 1,
              "SR": 133.01,
              "Avg": 36.0
            }
          }
        },
        {
          "title": "career_stats_at_M_Chinnaswamy_Stadium",
          "data": {
            "Batting": {
              "Matches": 109,
              "Innings": 106,
              "Runs": 3618,
              "Balls": 2514,
              "Outs": 92,
              "4s": 329,
              "6s": 154,
              "50s": 27,
              "100s": 4,
              "SR": 143.91,
              "Avg": 39.32
            }
          }
        }
      ]
    },
    {
      "name": "Shreyas Iyer",
      "role": "batsman",
      "is_wk": "False",
      "is_overseas": "False",
      "batting_style": "Right Handed Bat",
      "bowling_style": "Right-arm legbreak",
      "stats": [
        {
          "title": "last_8_innings_stats",
          "data": {
            "Batting": {
              "Matches": 8,
              "Innings": 8,
              "Runs": 340,
              "Balls": 201,
              "Outs": 5,
              "4s": 26,
              "6s": 19,
              "50s": 3,
              "100s": 0,
              "SR": 169.15,
              "Avg": 68.0
            }
          }
        },
        {
          "title": "career_stats_vs_Royal_Challengers_Bengaluru",
          "data": {
            "Batting": {
              "Matches": 17,
              "Innings": 17,
              "Runs": 408,
              "Balls": 339,
              "Outs": 16,
              "4s": 34,
              "6s": 13,
              "50s": 4,
              "100s": 0,
              "SR": 120.35,
              "Avg": 25.5
            }
          }
        },
        {
          "title": "career_stats_at_M_Chinnaswamy_Stadium",
          "data": {
            "Batting": {
              "Matches": 11,
              "Innings": 11,
              "Runs": 305,
              "Balls": 222,
              "Outs": 9,
              "4s": 26,
              "6s": 14,
              "50s": 3,
              "100s": 0,
              "SR": 137.38,
              "Avg": 33.88
            }
          }
        }
      ]
    },
    {
      "name": "Hardik Pandya",
      "role": "batting allrounder",
      "is_wk": "False",
      "is_overseas": "False",
      "batting_style": "Right Handed Bat",
      "bowling_style": "Right-arm fast-medium",
      "stats": [
        {
          "title": "last_8_innings_stats",
          "data": {
            "Batting": {
              "Matches": 8,
              "Innings": 7,
              "Runs": 120,
              "Balls": 76,
              "Outs": 5,
              "4s": 9,
              "6s": 6,
              "50s": 0,
              "100s": 0,
              "SR": 157.89,
              "Avg": 24.0
            },
            "Bowling": {
              "Matches": 8,
              "Innings": 7,
              "Overs": 13.0,
              "Maidens": 0,
              "Runs": 146,
              "Wkts": 3,
              "Eco": 11.23,
              "Avg": 48.67,
              "SR": 26.0
            }
          }
        },
        {
          "title": "career_stats_vs_Royal_Challengers_Bengaluru",
          "data": {
            "Batting": {
              "Matches": 18,
              "Innings": 17,
              "Runs": 361,
              "Balls": 220,
              "Outs": 8,
              "4s": 22,
              "6s": 26,
              "50s": 2,
              "100s": 0,
              "SR": 164.09,
              "Avg": 45.12
            },
            "Bowling": {
              "Matches": 18,
              "innings": 12,
              "Overs": 29.0,
              "Maidens": 0,
              "Runs": 303,
              "Wkts": 7,
              "Eco": 10.44,
              "Avg": 43.28,
              "SR": 24.86
            }
          }
        },
        {
          "title": "career_stats_at_M_Chinnaswamy_Stadium",
          "data": {
            "Batting": {
              "Matches": 12,
              "Innings": 9,
              "Runs": 162,
              "Balls": 112,
              "Outs": 6,
              "4s": 12,
              "6s": 9,
              "50s": 1,
              "100s": 0,
              "SR": 144.64,
              "Avg": 27.0
            },
            "Bowling": {
              "Matches": 12,
              "innings": 10,
              "Overs": 27.0,
              "Maidens": 0,
              "Runs": 240,
              "Wkts": 11,
              "Eco": 8.88,
              "Avg": 21.81,
              "SR": 14.73
            }
          }
        }
      ]
    }
  ]
}
))