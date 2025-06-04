from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool
from collections import defaultdict
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union, Any
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
import difflib
import time
import requests
import urllib.parse
import re
import os

load_dotenv()
LLM = ChatOpenAI(model = "gpt-4.1")

def fetch_response(url: str, headers: Dict) -> dict:
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
       return {}
    return response

def get_espn_player_id(player_name: str) -> Optional[int]:
    """
    Given a player's name (partial or full), performs a Bing search
    specifically targeting ESPNcricinfo’s “cricketers” pages. Parses the
    search results to find the first matching URL of the form
    "espncricinfo.com/cricketers/<name>-<id>" and returns that numeric ID.

    Returns:
        int: The extracted ESPNcricinfo player ID if found.
        None: If the HTTP response is not 200 or if no matching player URL
              appears in the search results.
    """
    query = f"{player_name} site:espncricinfo.com/cricketers"
    url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    response = fetch_response(url)
    
    # Looking for ESPNcricinfo player URL
    match = re.search(r"espncricinfo\.com/cricketers/[^/]+-(\d+)", response.text)
    if match:
        player_id = match.group(1)
        return player_id
    return None 


def resolve_to_id(name: str, mapping: Dict[str, int]) -> Optional[int]:
    """
    Fuzzy-match the provided name against mapping keys.
    Basically it is for mapping the team name and the venue name 
    with their espn id.
    Returns the mapped ID if a close match is found, else None.
    """
    if not name:
        return None
    # Exact case-insensitive match
    for key, val in mapping.items():
        if key.lower() == name.lower():
            return val
    # Substring match
    for key, val in mapping.items():
        if name.lower() in key.lower():
            return val
    # Fuzzy close match
    keys = list(mapping.keys())
    matches = difflib.get_close_matches(name, keys, n=1, cutoff=0.6)
    if matches:
        return mapping[matches[0]]
    return None


def get_recent_stats(player_id: int, role_type: str) -> Dict[str, Any]:
    """
    Fetch and aggregate stats over the last 8 t20 innings for a player.
    If the player hasn't played 8 t20 innings then it will fetch as many inn as the player has played.

    Supports three role_type values:
      - "batting": returns batting aggregates over last 8 innings:
          total_runs, total_balls, total_4s, total_6s, 50s, 100s, strike_rate, average
      - "bowling": returns bowling aggregates over last 8 innings:
          total_overs, total_maidens, total_runs_conceded, total_wickets, economy_rate, average, strike_rate
      - "allround": returns both batting and bowling aggregates.

    Args:
        player_id (int): ESPNcricinfo player ID.
        role_type (str): One of "batting", "wk-batsman", "bowling", "allround".

    Returns:
        Dict[Dict[str, Any]]: Aggregated stats depending on role_type.
        Returns a dict which contains one or two dict depending upon role_type
    Raises:
        ValueError: If role_type is not one of the supported options.
        RuntimeError: If the “Match by match list” table cannot be found on the page.
    """
    
    n = 8  # Number of recent innings to consider
    role_type = role_type.lower()

    def scrape_table(role_type: str):
        """
        Fetches the Statsguru “Match by match list” table for the given datatype.
        Args:
            role_type (str): Either "batting" or "bowling".
        Returns:
            Tuple[List[str], List[bs4.element.Tag]]: A tuple of (headers list, last n <tr> rows).
        Raises:
            RuntimeError: If the “Match by match list” table is not found.
        """
        url = f"https://stats.espncricinfo.com/ci/engine/player/{player_id}.html"
        params = {"class": 6, "template": "results", "type": role_type, "view": "match"}
        headers={"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers = headers); 
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tbl in soup.find_all("table", class_="engineTable"):
            cap = tbl.find("caption")
            if cap and "Match by match list" in cap.text:
                headers = [th.get_text(strip=True) for th in tbl.find("tr").find_all("th")]
                rows = tbl.find_all("tr")[1:][-n:]
                return headers, rows
        raise RuntimeError("Match-by-match table not found")

    def aggregate_batting(headers, rows):
        """
        Aggregates batting stats from the given table rows..
        Args:
            headers (List[str]): List of column names (from <th>).
            rows (List[bs4.element.Tag]): List of <tr> tags for the last n innings.

        Returns:
            Dict[Dict[str, Any]]: Batting aggregates:
            {
            "batting":
                { 
                  "Matches": int,
                  "Innings": int,
                  "Runs": int,
                  "Balls": int,
                  "Outs": int,
                  "4s": int,
                  "6s": int,
                  "50s": int,
                  "100s": int,
                  "SR": float,
                  "Avg": float
                }
            }
        """
        i_run = headers.index("Bat1")
        i_bf  = headers.index("BF")
        i_4s  = headers.index("4s")
        i_6s  = headers.index("6s")

        total_runs = total_balls = total_4s = total_6s = 0
        inns = not_outs = hundreds = fifties = 0

        for tr in rows:
            tds = tr.find_all("td")
            txt = tds[i_run].get_text(strip=True)

            # Skip if no batting data ("-")
            if txt == "-" or txt == "":
                continue
            not_out = txt.endswith("*")
            runs_str = txt.rstrip("*")
            if not runs_str.isdigit():
                continue
            runs = int(runs_str)
            # count fifties/hundreds
            if runs >= 100:
                hundreds += 1
            elif runs >= 50 and runs < 100:
                fifties += 1

            balls = int(tds[i_bf].get_text(strip=True) or 0)
            fours = int(tds[i_4s].get_text(strip=True) or 0)
            sixes = int(tds[i_6s].get_text(strip=True) or 0)

            inns += 1
            not_outs += 1 if not_out else 0
            total_runs  += runs
            total_balls += balls
            total_4s    += fours
            total_6s    += sixes

        dismissals = max(inns - not_outs, 0)
        avg = (total_runs / dismissals) if dismissals else float(total_runs)
        sr  = (total_runs / total_balls * 100) if total_balls else 0.0

        return {
            "Matches": 8,
            "Innings": inns,
            "Runs": total_runs,
            "Balls": total_balls,
            "Outs": dismissals,
            "4s": total_4s,
            "6s": total_6s,
            "50s": fifties,
            "100s": hundreds,
            "SR": round(sr, 2),
            "Avg": round(avg, 2),    
        }

    def aggregate_bowling(headers, rows):
        """
        Aggregates bowling stats from the given table rows.
        Args:
            headers (List[str]): List of column names (from <th>).
            rows (List[bs4.element.Tag]): List of <tr> tags for the last n innings.

        Returns:
           Dict[Dict[str, Any]]:: Bowling aggregates:
           {
           "Bowling":
                {
                  "Matches": int
                  "Innings": int,
                  "Overs": float,
                  "Maidens": int,
                  "Runs": int,
                  "Wkts": int,
                  "Eco": float or None,
                  "Avg": float or None,
                  "SR": float or None
                }
            }
        """
        i_ov = headers.index("Overs")
        i_md = headers.index("Mdns")
        i_cd = headers.index("Runs")
        i_wk = headers.index("Wkts")

        total_overs = total_maidens = total_conceded = total_wkts = 0.0
        inns = 0
        for tr in rows:
            tds = tr.find_all("td")
            overs_txt = tds[i_ov].get_text(strip=True)
            
            # Skip if no bowling data ("-")
            if overs_txt == "-" or overs_txt == "":
                continue

            inns += 1
            def to_f(i): 
                try: return float(tds[i].get_text(strip=True))
                except: return 0.0
            total_overs    += to_f(i_ov)
            total_maidens  += to_f(i_md)
            total_conceded += to_f(i_cd)
            total_wkts     += to_f(i_wk)

        balls =  (int(total_overs)) * 6 + round((total_overs - int(total_overs)) * 10)
        avg  = (total_conceded / total_wkts) if total_wkts else None
        econ = (total_conceded / total_overs) if total_overs else None
        sr   = (balls / total_wkts) if total_wkts else None

        return {
            "Matches": 8,
            "Innings": inns,
            "Overs": total_overs,
            "Maidens": int(total_maidens),
            "Runs": int(total_conceded),
            "Wkts": int(total_wkts),
            "Eco": round(econ, 2) if econ is not None else None,
            "Avg": round(avg, 2) if avg is not None else None,
            "SR": round(sr, 2) if sr is not None else None,
        }

    if role_type == "batting":
        hdrs, rows = scrape_table("batting")
        return {
            "Batting": aggregate_batting(hdrs, rows)
            }
    elif role_type in "bowling":
        hdrs, rows = scrape_table("bowling")
        return {
            "Bowling": aggregate_bowling(hdrs, rows)
            }
    elif role_type == "allround":
        bh, br = scrape_table("batting"); bowling_h, bowling_r = scrape_table("bowling")
        return {
            "Batting":  aggregate_batting(bh, br),
            "Bowling": aggregate_bowling(bowling_h, bowling_r)
        }
    else:
        raise ValueError("role_type must be 'batting', 'bowler' or 'allround'")
    
    
def get_opp_venue_stats(player_id: int, role_type: str, opposition_id: Optional[int] = None, venue_id: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Fetches filtered career stats for a given player,
    optionally restricted to a specific opposition or venue.

    Args:
        player_id (int): ESPNcricinfo player ID.
        role_type (str): One of 'batting', 'bowling', or 'allround'.
        opposition_id (Optional[int]): If provided, filter stats vs the given opposition.
        venue_id (Optional[int]): If provided, filter stats at the given ground.
                                  (Cannot use both together; opposition takes precedence if set.)

    Returns:
        Dict[str, Dict[str, Any]]:
          - If role_type == 'batsman', returns:
                {
                "Batting": {
                    "Matches": int,
                    "Innings": int,
                    "Runs": int,
                    "Balls": int,
                    "Outs": int,
                    "4s": int,
                    "6s": int,
                    "50s": int,
                    "100s": int,
                    "SR": float,
                    "Avg": float
                    }
                }
          - If role_type == 'bowling', returns:
                {
                "Bowling": {
                    "Matches": int,
                    "innings": int,
                    "Overs": float,
                    "Maidens": int,
                    "Runs": int,
                    "Wkts": int,
                    "Eco": float,
                    "Avg": float,
                    "SR": float
                    }
                }
          - If role_type == 'allround', returns:
                {
                  "Batting": {…batting fields…},
                  "Bowling": {…bowling fields…}
                }

    Raises:
        ValueError: If role_type is invalid or filtered row not found.
        RuntimeError: If the “Career averages” table cannot be located.
    """
    role_type = role_type.lower()
    if role_type not in ("batting", "bowling", "allround"):
        raise ValueError("role_type must be 'batting', 'bowling' or 'allround'")

    def fetch_filtered_row(role_type: str) -> list:
        """Fetches the 'filtered' stats row for a player and role."""
        url = f"https://stats.espncricinfo.com/ci/engine/player/{player_id}.html"
        params = {"class": 6, "template": "results", "type": role_type}
        # Apply opposition or venue filter
        if opposition_id is not None:
            params["opposition"] = opposition_id
        elif venue_id is not None:
            params["ground"] = venue_id

        headers={"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params = params, headers = headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Locate the <table> with caption containing "Career averages"
        target_table = None
        for tbl in soup.find_all("table", class_="engineTable"):
            cap = tbl.find("caption")
            if cap and "Career averages" in cap.text:
                target_table = tbl
                break

        if not target_table:
            raise RuntimeError("Career averages table not found")

        # Within that table, find the <tr> whose first cell is "filtered"
        for row in target_table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            if cells and cells[0].lower() == "filtered":
                return cells

        raise ValueError(f"Filtered row not found for {role_type} (opposition={opposition_id}, venue={venue_id})")

    def parse_batting(cells: list) -> Dict[str, Dict[str, Any]]:
        """Parses batting stats from the filtered row."""
        return {
            "Batting": {
                "Matches": int(cells[2]),
                "Innings": int(cells[3]),
                "Runs": int(cells[5]),
                "Balls": int(cells[8]),
                "Outs": int(cells[3]) - int(cells[4]),  # Inns - NO
                "4s": int(cells[13]),
                "6s": int(cells[14]),
                "50s": int(cells[11]),
                "100s": int(cells[10]),
                "SR": float(cells[9]),
                "Avg": float(cells[7])
                }
            }

    def parse_bowling(cells: list) -> Dict[str, Any]:
        """Parses bowling stats from the filtered row."""
        overs_txt = cells[4]
        overs = float(overs_txt) if overs_txt != "-" else 0.0
        whole_overs = int(overs)
        fraction = overs - whole_overs
        balls = whole_overs * 6 + round(fraction * 10)
        wickets = int(cells[7])

        avg_val = float(cells[9]) if cells[9] != "-" else None
        econ_val = float(cells[10]) if cells[10] != "-" else None
        sr_val = round(balls / wickets, 2) if wickets else None

        return {
            "Bowling": {
                "Matches": int(cells[2]),
                "innings": int(cells[3]),
                "Overs": overs,
                "Maidens": int(cells[5]),
                "Runs": int(cells[6]),
                "Wkts": wickets,
                "Eco": econ_val,
                "Avg": avg_val,
                "SR": sr_val
                }
            }

    # Branch based on role_type
    if role_type == "batting":
        batting_cells = fetch_filtered_row("batting")
        return parse_batting(batting_cells)

    elif role_type == "bowling":
        bowling_cells = fetch_filtered_row("bowling")
        return parse_bowling(bowling_cells)

    else:  # role_type == "allround"
        batting_cells = fetch_filtered_row("batting")
        bowling_cells = fetch_filtered_row("bowling")
        result_batting = parse_batting(batting_cells)
        result_bowling = parse_bowling(bowling_cells)
        return {
            "Batting": result_batting["Batting"],
            "Bowling": result_bowling["Bowling"]
        }


def combine_recent_stats(player_name: str, player_role: str, overall_stats: dict, opp_stats: dict, venue_stats: dict, 
                         opposition_label: str, venue_label: str, is_wk: bool, is_overseas: bool, batting_style: str, bowling_style: str) -> Dict[str, Any]:
    """For combining the the player details in a single dict."""
    n = 8 # number of recent matches
    result = {
        "name": player_name,
        "role": player_role,
        "is_wk": is_wk,
        "is_overseas": is_overseas,
        "batting_style": batting_style,
        "bowling_style": bowling_style,
        "stats": []
    }

    # 1) Overall
    result["stats"].append({
        "title": f"last_{n}_innings_stats",
        "data": overall_stats
    })

    # 2) Versus opposition
    if opp_stats is not None and opposition_label:
        safe_label = opposition_label.replace(' ', '_')
        result["stats"].append({
            "title": f"career_stats_vs_{safe_label}",
            "data": opp_stats
        })

    # 3) At venue
    if venue_stats is not None and venue_label:
        # only take first part before comma if present
        base = venue_label.split(',')[0].replace(' ', '_')
        result["stats"].append({
            "title": f"career_stats_at_{base}",
            "data": venue_stats
        })

    return result

# TOOLS -> DATA COLLECTOR AGENT
@tool
def player_details(player_names: List[str]) -> List[dict]:
    """This tool returns the player ID and name for a list of player names.
        It takes a list containing names of players as input and output a list 
        of dict with each dict containing the player id and name of that player."""
    
    id_headers = {
        'x-apihub-key': '9HN92wz6l7bberNNuKkhDCXeb4YH4lXo2fIKuVdgCpB82jpHlM',
        'x-apihub-host': 'Cricbuzz-Official-Cricket-API.allthingsdev.co',
        'x-apihub-endpoint': 'b0242771-45ea-4c07-be42-a6da38cdec41'
    }

    results = []

    for name in player_names:
        id_url = f"https://Cricbuzz-Official-Cricket-API.proxy-production.allthingsdev.co/browse/player?search={name.replace(' ', '+')}"
        data = fetch_response(id_url, id_headers)
        players = data.get("player", [])
        if not players:
            results.append({"player name": name, "error": "No player found"})
            continue

        player_id = players[0].get("id")
        player_name = players[0].get("name")
        is_overseas = True
        if players[0].get("teamName") == "India":
            is_overseas = False

        if not player_id:
            results.append({"player name": name, "error": "Player ID not found"})
            continue

        role_url = f"https://Cricbuzz-Official-Cricket-API.proxy-production.allthingsdev.co/browse/player/{player_id}"
        role_headers = {
            'x-apihub-key': '9HN92wz6l7bberNNuKkhDCXeb4YH4lXo2fIKuVdgCpB82jpHlM',
            'x-apihub-host': 'Cricbuzz-Official-Cricket-API.allthingsdev.co',
            'x-apihub-endpoint': 'a055bf38-0796-4fab-8fe3-6f042f04cdba'
        }
        info = fetch_response(role_url, role_headers)
        player_role = info.get("role", "Unknown").lower()
        
        is_wicketkeeper = "wk" in player_role

        batting_style = None
        bowling_style = None
        if "batsman" in player_role:
            batting_style = info.get("bat", "Unknown")
        elif "bowler" in player_role:
            bowling_style = info.get("bowl", "Unknown")
        else:
            batting_style = info.get("bat", "Unknown")
            bowling_style = info.get("bowl", "Unknown")

        results.append({
            "name": player_name,
            "role": player_role,
            "is_wicketkeeper": is_wicketkeeper,
            "is_overseas": is_overseas,
            "batting_style": batting_style,
            "bowling_style": bowling_style
        })

    return results


@tool
def player_stats(player_details: List[Dict[str, str]], venue_name: str) -> Dict[str, Any]:
    """
    Computes a player's form profile by aggregating T20 stats in 3 scopes:
    (1) overall last 8 innings, (2) vs a specific opposition, and (3) at a specific venue.

    Args:
        player_details (List[Dict[str, str]]): List of player metadata dicts.
            Each dict contain:
              - name (str): Full player name (e.g., "virat kohli")
              - role (str): Player's primary role (e.g., "batsman", "bowler", etc.)
              - is_wicketkeeper (str): "True"/"False"
              - is_overseas (str): "True"/"False"
              - batting_style (str): e.g., "Right-hand-Batsman"
              - bowling_style (str): e.g., "Right-arm-Medium"
              - opposition (str): Opponent team name (e.g., "kolkata")
        venue_name (str): Venue name (e.g., "eden")

    Returns:
        Dict[str, Any]: One summary dict per player in the format:
            {
              "player name": <str>,
              "role": <str>,
              "is_wicketkeeper": <bool>,
              "is_overseas": <bool>,
              "batting_style": <str>,
              "bowling_style": <str>,
              "stats": [
                {
                  "title": "last_8_innings_stats",
                  "data": { ... }
                },
                {
                  "title": "career_stats_vs_<Opposition>",
                  "data": { ... }
                },
                {
                  "title": "career_stats_at_<Venue>",
                  "data": { ... }
                }
              ]
            }
    """

    # Static Mapping
    opposition_ids = {
        "Royal Challengers Bengaluru": 4340,
        "Kolkata Knight Riders": 4341,
        "Punjab Kings": 4342,
        "Chennai Super Kings": 4343,
        "Delhi Capitals": 4344,
        "Rajasthan Royals": 4345,
        "Mumbai Indians": 4346,
        "Sunrisers Hyderabad": 5143,
        "Lucknow Super Giants": 6903,
        "Gujrat Titans": 6904
    }

    # Static Mapping
    venue_ids = {
        "Arun Jaitley Stadium, Delhi": 333,
        "Ekana Cricket Stadium, Lucknow": 3355,
        "Eden Gardens, Kolkata":292,
        "Chinnaswamy Stadium, Bengaluru": 683,
        "MA Chidambaram Stadium, Chepauk, Chennai": 291,
        "Narendra Modi Stadium, Ahmedabad": 840,
        "Sawai Mansingh Stadium, Jaipur": 664,
        "Wankhede Stadium, Mumbai": 713,
        "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur, CHandigarh": 3585,
        "Rajiv Gandhi International stadium, Hyderabad": 1981,
        "Barsapara Cricket Stadium, Guwahati": 2865,
        "Himachal Pradesh Cricket Association Stadium, Dharamshala": 1920,
        "Brabourne Stadium, Mumbai": 393,
        "Barabati Stadium, Cuttack, ": 442,
        "Dr DY Patil Sports Academy, Navi Mumbai": 2361,
        "Dr Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": 1896,
        "Holkar Cricket Stadium, Indore": 1055
    }
    
    results = []

    for detail in player_details:
        name = detail.get("name", "").strip()
        opposition = detail.get("opposition", "").strip()
        role = detail.get("role", "").strip().lower()
        is_wk = detail.get("is_wicketkeeper", "")
        is_overseas = detail.get("is_overseas", "")
        batting_style = detail.get("batting_style", "")
        bowling_style = detail.get("bowling_style", "")

        # Retry logic for resolving player ID
        player_id = None
        err = "Unknown error"
        for attempt in range(10):
            try:
                player_id = get_espn_player_id(name)
                if player_id is not None:
                    break  # Success
            except Exception as e:
                err = str(e)
            time.sleep(0.25)

        if player_id is None:
            results.append({"name": name, "error": f"Could not resolve ID after 10 attempts. Last error: {err}"})
            continue

        # Resolve opposition and venue IDs
        opposition_id = resolve_to_id(opposition, opposition_ids)
        venue_id = resolve_to_id(venue_name, venue_ids)

        if "batsman" in role: # wk-batsman, batsman
            role_type = "batting" # It is a parameter in the url
        elif "bowler" in role: # bowler
            role_type = "bowling"
        else:
            role_type = "allround" # battingallrounder, bowlingallrounder

        overall_stats_dict = get_recent_stats(player_id = player_id, role_type = role_type)
        opp_stats_dict = get_opp_venue_stats(player_id = player_id, opposition_id = opposition_id, venue_id = None, role_type = role_type)
        venue_stats_dict = get_opp_venue_stats(player_id = player_id, opposition_id = None, venue_id = venue_id, role_type = role_type)

    return combine_recent_stats(name, role, overall_stats_dict, opp_stats_dict, venue_stats_dict, opposition, venue_name, is_wk, is_overseas, batting_style, bowling_style)


# Data Collector Agent: 
data_collector_agent = create_react_agent(
    model = LLM,
    name = "data_miner",
    tools = [player_stats],
    prompt = (
        "You are an agent which is phenomenal at maths. Answer the query of the user to the best you can."
    )
)

print(player_details.invoke({"player_names": ["virat kohli"]}))