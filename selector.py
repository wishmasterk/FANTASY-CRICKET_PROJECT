from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from dotenv import load_dotenv
from typing import List, Dict, Union, Any, Tuple


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


def overall_score(player_stats_dict: Dict[str, Any]) -> float: # will do for single player
    """This function just compute the overall score -> weighted sum of 5 scores"""
    role = player_stats_dict["role"].lower()

    weight = {
        "recent": 0.3,
        "vs_opp": 0.15,
        "at_venue": 0.15,
        "head_2_head": 0.25,
        "pitch_fit": 0.15
    }

    recent_dict = (player_stats_dict["recent_stats"][0])["data"]
    vs_opp_dict = (player_stats_dict["recent_stats"][1])["data"]
    at_venue_dict = (player_stats_dict["recent_stats"][2])["data"]

    recent_score = compute_score(recent_dict, role)
    vs_opp_score = compute_score(vs_opp_dict, role)
    at_venue_score = compute_score(at_venue_dict, role)

    return (
        weight["recent"] * recent_score + 
        weight["vs_opp"] * vs_opp_score + 
        weight["at_venue"] * at_venue_score +
        weight["head_2_head"] * player_stats_dict["head_2_head_score"] +
        weight["pitch_fit"] * player_stats_dict["pitch_score"]
    )

@tool
def select_players(players_overall_details: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute and append an overall_score for each player, then return two lists:
    
    1. The original list of player dictionaries, each augmented with an "overall_score" key.
    2. A secondary list of simplified dictionaries for selection overview:
       [
         {
           "name": str,
           "role": str,
           "overall_score": float
         },
         ...
       ]
    
    Argument: List of dict where, each player dict:
         {
           "name": str,
           "role": str,
           "is_wicketkeeper": str,
           "is_overseas":  str,
           "batting_style": str,
           "bowling_style": str,
           "recent_stats": [
             {"title": "last_8_innings_stats",        "data": { … }},
             {"title": "career_stats_vs_<Opposition>", "data": { … }},
             {"title": "career_stats_at_<Venue>",      "data": { … }}
           ]
           "head_2_head_stats": [
              {....},
              {....},
              ...
           ],
           "bowling_type_stats": {
           "pace": {....},
           "spin": {....}
           },
           "head_2_head_score": <float>,
           "pitch_score": <float>
         }    

    Steps:
    - For each player in players_overall_details:
      1. Extract their "role" and their three "data" dictionaries under "recent_stats".
      2. Call `overall_score(player)` (which computes scores based on recent form, vs opposition, at venue, head_2_head_score, pitch_score).
      3. Add the returned float under player["overall_score"].
      4. Append { "name", "role", "overall_score" } to the secondary list.

    Returns:
    ----------
    Tuple[
      List[dict],  # The input list, but now each dict has "overall_score" as an additional key.
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
player_selector_agent = create_react_agent(
    model = LLM,
    name = "player_selector_agent",
    tools = [select_players],
    prompt = 
    """
    - You are the Player Selector Agent. Your job is to take a list of player statistics, compute an overall performance score for each player, and return both the full list with score and a concise selection list.

    **Input**  
    - A list of player dicts, each with:
    "name": str,
    "role": str,
    "is_wicketkeeper": str,
    "is_overseas":  str,
    "batting_style": str,
    "bowling_style": str,
    "recent_stats": [
      {"title": "last_8_innings_stats",        "data": { … }},
      {"title": "career_stats_vs_<Opposition>", "data": { … }},
      {"title": "career_stats_at_<Venue>",      "data": { … }}
      ]
    "head_2_head_stats": [
      {....},
      {....},
      ...
      ],
    "bowling_type_stats": {
      "pace": {....},
      "spin": {....}
      },
    "head_2_head_score": <float>,
    "pitch_score": <float>

    **Tool**
    - You have access to a tool - select_players
      - Appends an 'overall_score' to each player dict.
      - Args: The above list of dicts which you receive from the user, passed it as it is to this tool.
      - Returns a tuple:
        1. Full list of updated player dicts.(with "overall_score" additioanl key)
        2. Simplified list: [{ "name", "role", "overall_score" }, …]. -> it will have dicts corresponding to every player in the main list.

    **Return**
    - Just return both the lists which you will get from the select_players tool, as it is to the user.Strictly adhered to this.
    - Verify both the list before sending them to the user.
  """
)

result = player_selector_agent.invoke({"messages": [{"role": "user", "content":
      """
  [
    {
        "name": "Virat Kohli",
        "role": "batsman",
        "is_wk": "False",
        "is_overseas": "False",
        "batting_style": "Right Handed Bat",
        "bowling_style": "Right-arm medium",
        "recent_stats": [
            {"title": "last_8_innings_stats", "data": {"Batting": {"Matches": 8, "Innings": 8, "Runs": 408, "Balls": 278, "Outs": 7, "4s": 46, "6s": 9, "50s": 5, "100s": 0, "SR": 146.76, "Avg": 58.29}}},
            {"title": "career_stats_vs_Punjab_Kings","data": {"Batting": {"Matches": 36, "Innings": 36, "Runs": 1159, "Balls": 874, "Outs": 32, "4s": 120, "6s": 33, "50s": 6, "100s": 1, "SR": 132.6, "Avg": 36.21}}},
            {"title": "career_stats_at_M_Chinnaswamy_Stadium", "data": {"Batting": {"Matches": 109, "Innings": 106, "Runs": 3618, "Balls": 2514, "Outs": 92, "4s": 329, "6s": 154, "50s": 27, "100s": 4, "SR": 143.91, "Avg": 39.32}}}
        ],
        "head_2_head_stats": [
            {
                "opponent": "Jasprit Bumrah",
                "opp_role": "bowler",
                "stats": {"Innings": "17", "Runs": "150", "Balls": "101", "Outs": "5", "Dots": "37", "4s": "15", "6s": "6", "SR": "148.5", "Avg": "30.0"},
                "advantage_score": 0.08818651718112991
            }
        ],
        "bowler_type_stats": {
            "pace": {"Runs": 8425, "Balls": 5988, "Outs": 217, "4s": 872, "6s": 260, "50s": 14, "100s": 0, "SR": 140.7, "Avg": 38.82},   
            "spin": {"Runs": 4840, "Balls": 3866, "Outs": 84, "4s": 314, "6s": 169, "50s": 0, "100s": 0, "SR": 125.19, "Avg": 57.62}     
        },
        "head_2_head_score": 0.08818651718112991,
        "pitch_score": 0.7419933372093022
    },
    {
        "name": "Hardik Pandya",
        "role": "batting allrounder",
        "is_wk": "False",
        "is_overseas": "False",
        "batting_style": "Right Handed Bat",
        "bowling_style": "Right-arm fast-medium",
        "recent_stats": [
            {"title": "last_8_innings_stats", "data": {"Batting": {"Matches": 8, "Innings": 7, "Runs": 120, "Balls": 76, "Outs": 5, "4s": 9, "6s": 6, "50s": 0, "100s": 0, "SR": 157.89, "Avg": 24.0}, "Bowling": {"Matches": 8, "Innings": 7, "Overs": 13.0, "Maidens": 0, "Runs": 146, "Wkts": 3, "Eco": 11.23, "Avg": 48.67, "SR": 26.0}}},
            {"title": "career_stats_vs_Royal_Challengers_Bengaluru", "data": {"Batting": {"Matches": 18, "Innings": 17, "Runs": 361, "Balls": 220, "Outs": 8, "4s": 22, "6s": 26, "50s": 2, "100s": 0, "SR": 164.09, "Avg": 45.12}, "Bowling": {"Matches": 18, "innings": 12, "Overs": 29.0, "Maidens": 0, "Runs": 303, "Wkts": 7, "Eco": 10.44, "Avg": 43.28, "SR": 24.86}}},
            {"title": "career_stats_at_M_Chinnaswamy_Stadium", "data": {"Batting": {"Matches": 12, "Innings": 9, "Runs": 162, "Balls": 112, "Outs": 6, "4s": 12, "6s": 9, "50s": 1, "100s": 0, "SR": 144.64, "Avg": 27.0}, "Bowling": {"Matches": 12, "innings": 10, "Overs": 27.0, "Maidens": 0, "Runs": 240, "Wkts": 11, "Eco": 8.88, "Avg": 21.81, "SR": 14.73}}}
        ],
        "head_2_head_stats": [
            {
                "opponent": "Jasprit Bumrah", "opp_role": "bowler",
                "stats": {"Innings": "1", "Runs": "6", "Balls": "3", "Outs": "0", "Dots": "0", "4s": "1", "6s": "0", "SR": "200.0", "Avg": "0.0"},
                "advantage_score": 0.0
            },
            {
                "opponent": "Shreyas Iyer", "opp_role": "batsman",
                "stats": {"Innings": "4", "Runs": "24", "Balls": "21", "Outs": "1", "Dots": "8", "4s": "2", "6s": "1", "SR": "114.3", "Avg": "24.0"},
                "advantage_score": -0.01584523809523808
            }
        ],
        "bowler_type_stats": {
            "pace": {"Runs": 3263, "Balls": 2130, "Outs": 127, "4s": 273, "6s": 170, "50s": 2, "100s": 0, "SR": 153.19, "Avg": 25.69},   
            "spin": {"Runs": 1309, "Balls": 1037, "Outs": 32, "4s": 71, "6s": 74, "50s": 0, "100s": 0, "SR": 126.23, "Avg": 40.91}       
        },
        "head_2_head_score": -0.00792261904761904,
        "pitch_score": 0.6673397421383648
    },
    {
        "name": "Jasprit Bumrah",
        "role": "bowler",
        "is_wk": "False",
        "is_overseas": "False",
        "batting_style": "Right Handed Bat",
        "bowling_style": "Right-arm fast",
        "recent_stats": [
            {"title": "last_8_innings_stats", "data": {"Bowling": {"Matches": 8, "Innings": 8, "Overs": 31.2, "Maidens": 0, "Runs": 197, "Wkts": 14, "Eco": 6.31, "Avg": 14.07, "SR": 13.43}}},
            {"title": "career_stats_vs_Royal_Challengers_Bengaluru", "data": {"Bowling": {"Matches": 20, "innings": 20, "Overs": 78.0, "Maidens": 2, "Runs": 581, "Wkts": 29, "Eco": 7.44, "Avg": 20.03, "SR": 16.14}}},
            {"title": "career_stats_at_M_Chinnaswamy_Stadium", "data": {"Bowling": {"Matches": 10, "innings": 10, "Overs": 78.0, "Maidens": 2, "Runs": 581, "Wkts": 29, "Eco": 7.44, "Avg": 20.03, "SR": 16.14}}}
        ],
        "head_2_head_stats": [
            {
                "opponent": "Virat Kohli", "opp_role": "batsman",
                "stats": {"Innings": "17", "Runs": "150", "Balls": "101", "Outs": "5", "Dots": "37", "4s": "15", "6s": "6", "SR": "148.5", "Avg": "30.0"},
                "advantage_score": -0.08818651718112991
            },
            {
                "opponent": "Hardik Pandya", "opp_role": "batsman",
                "stats": {"Innings": "1", "Runs": "6", "Balls": "3", "Outs": "0", "Dots": "0", "4s": "1", "6s": "0", "SR": "200.0", "Avg": "0.0"},
                "advantage_score": -0.0
            }
        ],
        "bowler_type_stats": {
            "pace": {},
            "spin": {}
        },
        "head_2_head_score": -0.044093258590564954,
        "pitch_score": 0.3
    },
    {
        "name": "Shreyas Iyer",
        "role": "batsman",
        "is_wk": "False",
        "is_overseas": "False",
        "batting_style": "Right Handed Bat",
        "bowling_style": "Right-arm legbreak",
        "recent_stats": [
            {"title": "last_8_innings_stats", "data": {"Batting": {"Matches": 8, "Innings": 8, "Runs": 316, "Balls": 187, "Outs": 6, "4s": 25, "6s": 18, "50s": 3, "100s": 0, "SR": 168.98, "Avg": 52.67}}},
            {"title": "career_stats_vs_Royal_Challengers_Bengaluru", "data": {"Batting": {"Matches": 18, "Innings": 18, "Runs": 409, "Balls": 341, "Outs": 17, "4s": 34, "6s": 13, "50s": 4, "100s": 0, "SR": 119.94, "Avg": 24.05}}},
            {"title": "career_stats_at_M_Chinnaswamy_Stadium", "data": {"Batting": {"Matches": 11, "Innings": 11, "Runs": 305, "Balls": 222, "Outs": 9, "4s": 26, "6s": 14, "50s": 3, "100s": 0, "SR": 137.38, "Avg": 33.88}}}
        ],
        "head_2_head_stats": [
            {
                "opponent": "Hardik Pandya", "opp_role": "bowler",
                "stats": {"Innings": "4", "Runs": "24", "Balls": "21", "Outs": "1", "Dots": "8", "4s": "2", "6s": "1", "SR": "114.3", "Avg": "24.0"},
                "advantage_score": 0.01584523809523808
            }
        ],
        "bowler_type_stats": {
            "pace": {"Runs": 3179, "Balls": 2366, "Outs": 99, "4s": 328, "6s": 104, "50s": 1, "100s": 0, "SR": 134.36, "Avg": 32.11},    
            "spin": {"Runs": 1843, "Balls": 1386, "Outs": 48, "4s": 93, "6s": 100, "50s": 0, "100s": 0, "SR": 132.97, "Avg": 38.4}       
        },
        "head_2_head_score": 0.01584523809523808,
        "pitch_score": 0.6469136020408163
    }
  ] 
      """
      }]})
for message in result["messages"]:
    message.pretty_print()