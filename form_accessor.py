# cricmetric website scraped using scrape.do API
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
from bs4 import BeautifulSoup
import requests
import urllib.parse

load_dotenv()

LLM = ChatOpenAI(model = "gpt-4.1")

tavily_search = TavilySearchResults(max_results = 5) 
duck_search = DuckDuckGoSearchRun() 

def name_variants(full_name: str) -> list:
    """
    Generate common abbreviated forms for a multi-word player name to improve lookup success.

    Parameters:
        full_name (str): 
            The player’s full name (e.g., "Ayush Mhatre" or "Virat Kohli").

    Returns:
        List[str]: 
            A list containing:
              - The original full name.
              - If two words: the initial of the first name + the last name (e.g., "A Mhatre").
              - If three words: combinations like "AB C" and "A B C" (to match site variants).

    Example:
        >>> name_variants("Ayush Mhatre")
        ["Ayush Mhatre", "A Mhatre"]
    """
    parts = full_name.strip().split()
    variants = [full_name.strip()]

    if len(parts) == 2:
        first, last = parts
        variants.append(f"{first[0]} {last}")
    elif len(parts) == 3:
        a, b, c = parts
        variants.append(f"{a[0]}{b[0]} {c}")  # e.g. "AB C"
        variants.append(f"{a[0]} {b} {c}")    # e.g. "A B C" → but already full

    return variants


def fetch_table(batsman: str, bowler: str) -> str:
    """
    Fetch and filter head-to-head matchup tables for a batsman vs. bowler from CricMetric via scrape.do.

    Process:
      1. Build a URL for the full batsman/bowler names and call the scrape.do API (with rendering).
      2. Parse the returned HTML with BeautifulSoup.
      3. Locate all `<div class="panel panel-default">` sections.
      4. Keep only those whose heading text contains “T20I” or “TWENTY20”.
      5. Extract the inner `<table class="table">` blocks and concatenate them.
      6. If no tables found, retry using abbreviated name variants from `name_variants`. -> becoz on website some players stats are not available as complete names
      but as abbreviated names. Ex - A mhatre in place of Ayush mhatre
      7. Return the concatenated HTML of matched tables, or an empty string if none.

    Parameters:
        batsman (str): 
            Full name of the batsman (e.g., "Virat Kohli").
        bowler (str):
            Full name of the bowler (e.g., "Mitchell Starc").

    Returns:
        str: 
            Raw HTML string containing only the filtered `<table class="table">` blocks
            for T20I/TWENTY20 matchups, or "" if no matching tables were found.
    """
    base_matchup = "https://www.cricmetric.com/matchup.py"

    def attempt_fetch(name_a: str, name_b: str) -> str:
        """Attempt to fetch, then extract and return only T20I/TWENTY20 tables."""
        a_q = name_a.replace(" ", "+")
        b_q = name_b.replace(" ", "+")
        matchup_url = f"{base_matchup}?batsman={a_q}&bowler={b_q}&groupby=match"
        quoted = urllib.parse.quote(matchup_url, safe="")
        token = "c7cda0a41de3446abf92b8b0154c65e7922123609fe"
        scrape_do = f"http://api.scrape.do/?token={token}&url={quoted}&render=true"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(scrape_do, headers = headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        filtered_html = []

        # 1) Find every panel that wraps an entire section (ODI, T20I, etc.)
        for panel in soup.find_all("div", class_="panel panel-default"):
            # 2) Read its heading text
            heading_div = panel.find("div", class_="panel-heading")
            label = heading_div.get_text(strip=True).upper() if heading_div else ""

            # 3) If that heading says "T20I" or "TWENTY20", grab its <table class="table">
            if "T20I" in label or "TWENTY20" in label:
                # Inside this panel, find the first <table class="table">
                tbl = panel.find("table", class_="table")
                if tbl:
                    filtered_html.append(str(tbl))

        # 4) Return all matched tables concatenated (or "" if none)
        return "".join(filtered_html)

    # First try full names
    table_html = attempt_fetch(batsman, bowler)
    if table_html:
        return table_html

    # For some cicketers there are shrt forms on the website, like (A Mhatre) in place of Ayush Mhatre so creating diff variants and checking
    bats_variants = name_variants(batsman)
    bowl_variants = name_variants(bowler)

    for bv in bats_variants:
        for ov in bowl_variants:
            if bv == batsman and ov == bowler:
                continue
            table_html = attempt_fetch(bv, ov)
            if table_html:
                return table_html

    # If none worked, return empty
    return ""


def parse_table(table_html: str) -> Dict[str, str]:
    """
    Parse aggregated batting statistics from one or more HTML tables.

    Steps:
      1. Parse the provided HTML into BeautifulSoup.
      2. Identify all `<table class="table">` blocks.
      3. From the first table, read the header row (`<th>`) to get column names.
      4. For each table:
         a. Count `<tbody>` rows to accumulate total innings.
         b. Extract the `<tfoot><tr>` “Total” row.
         c. Sum each numeric column (Runs, Balls, Outs, Dots, 4s, 6s).
      5. Compute combined Strike Rate `SR = (total_runs/total_balls)*100` and 
         Average `Avg = total_runs/total_outs`.
      6. Return a dict with stringified values for:
         `"Innings"`, `"Runs"`, `"Balls"`, `"Outs"`, `"Dots"`, `"4s"`, `"6s"`, `"SR"`, and `"Avg"`.

    Parameters:
        table_html (str):
            HTML containing one or more `<table class="table">` elements,
            already filtered for T20I/TWENTY20 by `fetch_table`.

    Returns:
        Dict[str, str]:
            {
              "Innings": "<total innings count>",
              "Runs":    "<sum of runs>",
              "Balls":   "<sum of balls faced>",
              "Outs":    "<sum of dismissals>",
              "Dots":    "<sum of dot balls>",
              "4s":      "<sum of fours>",
              "6s":      "<sum of sixes>",
              "SR":      "<combined strike rate (one decimal)>",
              "Avg":     "<combined batting average (one decimal)>"
            }

    Raises:
        RuntimeError: if no tables are found or the header/total rows are malformed.
    """
    stats: Dict[str, str] = {}

    soup = BeautifulSoup(table_html, "html.parser")
    tables = soup.find_all("table", class_="table")
    if not tables:
        raise RuntimeError("No <table class='table'> blocks found to extract totals.")

    # Extract headers from the first table
    first_header_row = tables[0].find("tr")
    if not first_header_row:
        raise RuntimeError("No <tr> found in first table to extract headers.")
    headers = [th.get_text(strip=True) for th in first_header_row.find_all("th")]

    # Initialize running totals for each numeric column
    running_totals: Dict[str, float] = {col: 0.0 for col in headers[1:]}
    total_innings = 0

    for tbl in tables:
        # Count how many <tr> exist inside <tbody> for this table
        tbody = tbl.find("tbody")
        if not tbody:
            continue
        body_rows = tbody.find_all("tr")
        total_innings += len(body_rows)

        # Extract the <tfoot><tr> from this table
        tfoot = tbl.find("tfoot")
        if not tfoot:
            continue
        total_row = tfoot.find("tr")
        if not total_row:
            continue
        cells = total_row.find_all("td")
        if len(cells) != len(headers):
            raise RuntimeError(
                f"Header count ({len(headers)}) != Total row cell count ({len(cells)})."
            )

        # Sum up this table’s totals into running_totals
        for col_name, cell in zip(headers[1:], cells[1:]):
            text = cell.get_text(strip=True).replace(",", "")
            try:
                val = float(text)
            except ValueError:
                val = 0.0
            running_totals[col_name] += val

    # Now compute combined SR and Avg from aggregated Runs, Balls, Outs
    total_runs = running_totals.get("Runs", 0.0)
    total_balls = running_totals.get("Balls", 0.0)
    total_outs = running_totals.get("Outs", 0.0)

    combined_sr = 0.0
    if total_balls > 0:
        combined_sr = (total_runs / total_balls) * 100.0

    combined_avg = 0.0
    if total_outs > 0:
        combined_avg = total_runs / total_outs

    # Populate stats dict
    stats["Innings"] = str(total_innings - 1)
    stats["Runs"] = str(int(running_totals.get("Runs", 0.0)))
    stats["Balls"] = str(int(running_totals.get("Balls", 0.0)))
    stats["Outs"] = str(int(running_totals.get("Outs", 0.0)))
    stats["Dots"] = str(int(running_totals.get("Dots", 0.0)))
    stats["4s"] = str(int(running_totals.get("4s", 0.0)))
    stats["6s"] = str(int(running_totals.get("6s", 0.0)))
    stats["SR"] = f"{combined_sr:.1f}"
    stats["Avg"] = f"{combined_avg:.1f}"

    return stats


def players_faceoff(batsman: str, bowler: str) -> Dict[str, str]:
    """
    Compare a batsman and bowler head-to-head in T20I cricket and return aggregated career stats.

    Uses two internal helpers:
      1. `fetch_table` — retrieves the raw HTML head-to-head tables via scrape.do.
      2. `parse_table` — parses that HTML and computes totals and averages.

    Workflow:
      - Call `fetch_table(batsman, bowler)`.
      - If no data returned, immediately return `{}`.
      - Otherwise parse the HTML with `parse_table` to get career summary.

    Input:
        batsman (str): 
            Full name of the batsman (e.g., "AB de Villiers").
        bowler  (str): 
            Full name of the bowler (e.g., "Lasith Malinga").

    Output:
        Dict[str, str]:
            A dictionary with keys matching:
            `"Innings"`, `"Runs"`, `"Balls"`, `"Outs"`, `"Dots"`, `"4s"`, `"6s"`, `"SR"`, `"Avg"`.
            If no head-to-head records exist, returns an empty dict.

    Example:
        >>> players_faceoff("Virat Kohli", "Mitchell Starc")
        {
          "Innings": "24",
          "Runs":    "680",
          "Balls":   "610",
          "Outs":    "18",
          "Dots":    "120",
          "4s":      "70",
          "6s":      "25",
          "SR":      "111.5",
          "Avg":     "37.8"
        }
    """
    table_html = fetch_table(batsman, bowler)
    if not table_html:
        return {}
    return parse_table(table_html)

def get_player_pace_spin_stats(player_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetches a batsman’s T20 career stats broken down by opponent bowling type (pace vs. spin),
    using Scrape.do to retrieve the page. 
    - player_name: e.g. "sachin tendulkar"
    - role: "batsman" or "batting allrounder", "bowling_allrounder"
    
    Returns a dict with two keys: "pace" and "spin", each mapping to an aggregated stats dict:
        {
            "pace": {
                "Innings": int,
                "Runs": int,       
                "Balls": int,      
                "Outs": int,       
                "4s": int,
                "6s": int,
                "50s": int,
                "100s": int,
                "HS": int,
                "SR": float,
                "Avg": float,
            },
            "spin": { ... same fields ... }
        }
    If the Scrape.do fetch fails or no table is found, returns {"pace": {}, "spin": {}}.
    """
    # Construct the direct CricMetric URL
    raw_url = (
        "https://www.cricmetric.com/playerstats.py?"
        f"player={player_name.replace(' ', '+')}&role=batsman"
        "&format=All_T20&groupby=opp_player_type"
    )
    # Wrap with Scrape.do
    quoted = urllib.parse.quote(raw_url, safe="")
    SCRAPER_TOKEN = "c7cda0a41de3446abf92b8b0154c65e7922123609fe"
    scrape_url = f"http://api.scrape.do/?token={SCRAPER_TOKEN}&url={quoted}&render=true"

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(scrape_url, headers = headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Locate the single <table class="table scoretable">
    table = soup.find("table", class_="table scoretable")
    if not table:
        return {"pace": {}, "spin": {}}

    # Extract headers to map column indices
    header_cells = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    col_index = {name: idx for idx, name in enumerate(header_cells)}

    def parse_int(text: str) -> int:
        try:
            return int(text.replace(",", "").strip())
        except:
            return 0

    def parse_float(text: str) -> float:
        try:
            return float(text.replace(",", "").strip())
        except:
            return 0.0

    # Initialize accumulators
    def new_accumulator() -> Dict[str, Any]:
        return {
            "Runs": 0,
            "Balls": 0,
            "Outs": 0,
            "4s": 0,
            "6s": 0,
            "50s": 0,
            "100s": 0,
        }

    pace_acc = new_accumulator()
    spin_acc = new_accumulator()

    # Process each row in <tbody>
    for row in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        vs_type = cells[col_index["Versus Player Type"]].lower()

        is_pace = "fast" in vs_type or "medium" in vs_type
        is_spin = any(keyword in vs_type for keyword in ["chinaman", "orthodox", "legbreak", "offbreak"])
        if not (is_pace or is_spin):
            continue

        runs = parse_int(cells[col_index["Runs"]])
        balls = parse_int(cells[col_index["Balls"]])
        outs = parse_int(cells[col_index["Outs"]])
        fifties = parse_int(cells[col_index["50"]]) if "50" in col_index else 0
        hundreds = parse_int(cells[col_index["100"]]) if "100" in col_index else 0
        fours = parse_int(cells[col_index["4s"]])
        sixes = parse_int(cells[col_index["6s"]])

        acc = pace_acc if is_pace else spin_acc
        acc["Runs"] += runs
        acc["Balls"] += balls
        acc["Outs"] += outs
        acc["4s"] += fours
        acc["6s"] += sixes
        acc["50s"] += fifties
        acc["100s"] += hundreds

    # Compute final metrics
    def finalize(acc: Dict[str, Any]) -> Dict[str, Any]:
        total_runs = acc["Runs"]
        total_balls = acc["Balls"]
        total_outs = acc["Outs"]
        sr = (total_runs / total_balls) * 100.0 if total_balls > 0 else 0.0
        avg = (total_runs / total_outs) if total_outs > 0 else 0.0

        return {
            "Runs": total_runs,
            "Balls": total_balls,
            "Outs": total_outs,
            "4s": acc["4s"],
            "6s": acc["6s"],
            "50s": acc["50s"],
            "100s": acc["100s"],
            "SR": round(sr, 2),
            "Avg": round(avg, 2),
        }

    return {
        "pace": finalize(pace_acc),
        "spin": finalize(spin_acc)
    }


def compute_faceoff_score(faceoff: dict) -> float:
    """
    Compute a numeric "advantage" score from head-to-head stats.

    Parameters:
        faceoff (Dict[str, str]): Output of `players_faceoff` fucntion.
        It is a dict containing the head-2-head stats.

    Returns:
        float: Positive favors batsman, negative favors bowler.
    """

    # Convert all to numeric safely
    def to_float(val):
        try:
            return float(val)
        except:
            return None

    min_inns = 5 # min inns
    min_balls = 20 # min balls
    inns = to_float(faceoff.get('Innings', 0))
    runs = to_float(faceoff.get('Runs', 0))
    balls = to_float(faceoff.get('Balls', 0))
    outs = to_float(faceoff.get('Outs', 0))
    dots = to_float(faceoff.get('Dots', 0))
    fours = to_float(faceoff.get('4s', 0))
    sixes = to_float(faceoff.get('6s', 0))
    sr = to_float(faceoff.get('SR', 0))
    avg = to_float(faceoff.get('Avg')) if faceoff.get('Avg') not in [None, '-', ''] else None

    if min_inns > inns and min_balls > balls:
        return 0.0

    # Define weights
    batter_weights = {
        'sr': 0.5,
        'avg': 0.4,
        'boundaries': 0.1
    }

    bowler_weights = {
        'dots': 0.2,
        'outs': 0.8,
    }

    # Compute boundary impact
    boundaries = fours + sixes

    # Normalize SR
    sr_benchmark = 200
    sr_score = sr / sr_benchmark # taking strike_rate >= 200 excellent

    # Normalize Avg (avg >= 50 shows pure dominance)
    avg_benchmark = 50
    avg_score = avg / avg_benchmark

    # Normalize Dots
    dot_score = (dots / balls)

    # Normalize Outs (more outs = bowler dominance)
    # becoz I think that if the bowler dismissed the batsman even the half of the inns, still it outperforms the batter no matter what the strike rate and avg is
    out_score = 2 * (outs / inns) 

    # Normalize Boundaries (higher is better for batsman)
    boundary_score = ((boundaries / balls))

    # Weighted Score
    batter_score = (
        sr_score * batter_weights['sr'] +
        avg_score * batter_weights['avg'] +
        boundary_score * batter_weights['boundaries']
    )

    bowler_score = (
        dot_score * bowler_weights['dots'] +
        out_score * bowler_weights['outs']
    )

    raw_score = batter_score - bowler_score # mostly b/w -1 to 1, but can go either ways

    return raw_score


def compute_pitch_score(player: Dict[str, Any], pitch_cond: str):
    """
    Compute a pitch-condition-adjusted score for a player.
    It is computed differently for batsman, allrounders and for bowlers.

    Parameters:
        player (Dict[str, Any]): Player stats dict with "bowler_type_stats" included.(entire dict)
        pitch_cond (str): One of "seamer_friendly", "spin_friendly", "flat", "balanced" or None.

    Returns:
        float: Score between 0.0 and 1.0 indicating fit to pitch.
    """
    def compute(stats: Dict[str, Any])  -> Tuple[float, float]:
        """It computes the pitch score for batsman/allrounders on the basis of there performance against bowling_type.
        Args:
        Takes a dict as input containing the stats of that player against pace and against spin.
        
        Returns:
        A tuple containing the pace_score and spin_score of that player."""

        pace = stats["pace"]
        spin = stats["spin"]
        pace_wkts_ratio = pace["Outs"] / (pace["Outs"] + spin["Outs"])
        spin_wkts_ratio = 1 - pace_wkts_ratio

        avg_benchmark = 50.0 # basically we are seeing how good are the records of the batter against pace and spin
        sr_benchmark = 200.0

        weight = {
        'sr': 0.30,
        'avg': 0.30,
        'wkt_ratio': 0.40,
        }

        pace_sr_score =  pace["SR"] / sr_benchmark # more leads to more 
        pace_avg_score = pace["Avg"] / avg_benchmark # more means it is better against the pacer
        pace_wkt_ratio_score = pace_wkts_ratio # less it be better is the bowler against pace

        pace_score = (
            weight["sr"] * pace_sr_score +
            weight["avg"] * pace_avg_score +
            weight["wkt_ratio"] * (1 - pace_wkt_ratio_score)
        )

        spin_sr_score = spin["SR"] / sr_benchmark # more -> more
        spin_avg_score = spin["Avg"] / avg_benchmark # more -> more
        spin_wkt_ratio_score = spin_wkts_ratio     # less -> more

        spin_score = (
            weight["sr"] * spin_sr_score +
            weight["avg"] * spin_avg_score +
            weight["wkt_ratio"] * (1 - spin_wkt_ratio_score)
        )

        return pace_score, spin_score

    if not pitch_cond: # case in which pitch_cond is not fetched
        return 0.0
    
    role = player["role"].lower()
    bowling_style = player["bowling_style"].lower()
    if "bowler" in role:
        if pitch_cond == "seamer_friendly":
            if "fast" in bowling_style or "medium" in bowling_style:
                return 0.7 # since cond is favourable for pace, hence more score to the pacers
            else:
                return 0.3 # low score to the spinners

        elif pitch_cond == "spin_friendly":
            if "fast" in bowling_style or "medium" in bowling_style:
                return 0.3 # since cond is favourable for spin, hence low score to the pacers
            else:
                return 0.7 # high score to the spinners

        else: # flat, balanced
            return 0.5 # same score for both

    else: # for batsman, allrounders
        stats = player["bowler_type_stats"]

        if pitch_cond == "seamer_friendly":
            pace_score, spin_score = compute(stats)
            return (0.7 * pace_score + 0.3 * spin_score) # more weightage as pitch is pace friendly, so players which has good record against the pacers will get higher points

        elif pitch_cond == "spin_friendly":
            pace_score, spin_score = compute(stats)
            return (0.3 * pace_score + 0.7 * spin_score) # more weightage for spin

        else: # flat, balanced
            pace_score, spin_score = compute(stats)
            return (0.5 * pace_score + 0.5 * spin_score) # equal weightage
    
    
@ tool
def head_2_head(team_A : List[Dict[str, Any]], team_B : List[Dict[str, Any]], pitch_cond: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """ It appends the head_2_head stats between players of two teams, append the bowling type stats for players, append the head_2_head score
    and pitch_score for all players to the original list of dicts for both the teams.
    Parameters:
    team_A (List[Dict[str, Any]]):
        List of player dicts for Team A. Each dict has a structure:
        {
            "name": <str>,
            "role": <str>,              
            "is_wk": <str>          
            "is_overseas": <str>    
            "batting_style": <str>    
            "bowling_style": <str>     
            "recent_stats": [
                {
                    "title": "last_8_innings_stats",
                    "data": {
                        "Batting": {
                        .....
                        },
                        "Bowling": { # in case when player is a allrounder
                        .....
                        }
                    }
                },
                {
                    "title": "career_stats_vs_<opposition>",
                    "data": {
                        "Batting":{
                        .....
                        },
                        "Bowling": {
                        .....
                        }
                    }
                },
                {
                    "title": "career_stats_at_<venue>",
                    "data": {
                        "Batting":{
                        .....
                        },
                        "Bowling": {
                        .....
                        }
                    }
                }
            ]
        }

    team_B (List[Dict[str, Any]]):
        Same structure as team_A for Team B.
    pitch_cond (str):
        One of "seamer_friendly", "spin_friendly", "flat", "balanced" or "None".
    
    Returns: 
    Updated list of dict containing the stats of the players for both team -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]],
    separate list for separate team:
    [
        {
            "name": <str>,
            "role": <str>,              
            "is_wk": <str>          
            "is_overseas": <str>    
            "batting_style": <str>    
            "bowling_style": <str>     
            "recent_stats": [
                {
                    "title": "last_8_innings_stats",
                    "data": {
                        "Batting": {
                        .....
                        },
                        "Bowling": { # in case when player is a allrounder
                        .....
                        }
                    }
                },
                {
                    "title": "career_stats_vs_<opposition>",
                    "data": {
                        "Batting":{
                        .....
                        },
                        "Bowling": {
                        .....
                        }
                    }
                },
                {
                    "title": "career_stats_at_<venue>",
                    "data": {
                        "Batting":{
                        .....
                        },
                        "Bowling": {
                        .....
                        }
                    }
                }
            ], 
            # — Head-to-head breakdowns
            "head_2_head_stats": [  # one per opponent encountered
                {
                    "opponent":       "<str: opponent name>",
                    "opp_role":       "<str: 'bowler' or 'batsman'>",
                    "stats": {        # raw output from players_faceoff
                        "Title": "<str: e.g. 'Virat Kohli V/S Jasprit Bumrah'>",
                        "Stats": {
                            "Innings": "<str>",
                            "Runs":    "<str>",
                            "Balls":   "<str>",
                            "Outs":    "<str>",
                            "Dots":    "<str>",
                            "4s":      "<str>",
                            "6s":      "<str>",
                            "SR":      "<str>",
                            "Avg":     "<str>",
                        },
                    },
                    "advantage_score": <float>,
                },
                # … additional opponents dict …
            ],

            # — Aggregated career split by bowling type
            "bowler_type_stats": {
                "pace": {
                    "Runs":  <int>,
                    "Balls": <int>,
                    "Outs":  <int>,
                    "4s":    <int>,
                    "6s":    <int>,
                    "50s":   <int>,
                    "100s":  <int>,
                    "SR":    <float>,
                    "Avg":   <float>,
                },
                "spin": {
                    # same fields as above
                }
            },

            # — Summary scores
            "head_2_head_score": <float>,
            "pitch_score":       <float>, 
        },
        {
        ..... # separate dict for each player have the similar kind of structure
        },
    ] 
        """

    def categorize_players(players: List[Dict[str, Any]]):
        """
        Splits a list of player dicts containing recent stats into batting‐side and bowling‐side lists.
        Any role containing 'batsman' or 'allrounder' goes into batting_side.
        Any role containing 'bowler' or 'allrounder' goes into bowling_side.
        """
        batting_side = []
        bowling_side = []
        for p in players:
            role = p["role"].lower()
            name = p["name"]
            if "batsman" in role or "allrounder" in role:
                batting_side.append(name)
            if "bowler" in role or "allrounder" in role:
                bowling_side.append(name)
        return batting_side, bowling_side

    # Categorize each team's players into batsman or bowler to get head-2-head stats
    tA_bats, tA_bowl = categorize_players(team_A)
    tB_bats, tB_bowl = categorize_players(team_B)

    # Initializing an empty head_2_head_stats list for every player and also head_2_head_score = None
    for player in team_A:
        player["head_2_head_stats"] = []
        player["bowler_type_stats"] = {}
        player["head_2_head_score"] = None
        player["pitch_score"] = None
    for player in team_B:
        player["head_2_head_stats"] = []
        player["bowler_type_stats"] = {}
        player["head_2_head_score"] = None
        player["pitch_score"] = None

    # Build a dict for each player with name as key and their dict as value
    lookup_A = {p["name"]: p for p in team_A} # p -> dict 
    lookup_B = {p["name"]: p for p in team_B}

    # For each batsman/allrounder in Team A vs each bowler/allrounder in Team B
    for bats in tA_bats:
        for bowl in tB_bowl:
            stats = players_faceoff(bats, bowl) # returns a dict of stats

            faceoff_score = compute_faceoff_score(stats) # a float value
            # if faceoff is +ve -> batsman domination, means same score with -ve sign will be added to bowler
            # if faceoff is -ve ->  bowler domination, means same score with +ve sign will be added to bowler

            # Append to batsman's head_2_head_stats
            lookup_A[bats]["head_2_head_stats"].append({
                "opponent": bowl,
                "opp_role": "bowler", # imp in case of allrounders otherwise we won't know whether it is his batting score or bowling score
                "stats": stats,
                "advantage_score": faceoff_score
            })
            # Also append to bowler's head_2_head_stats
            lookup_B[bowl]["head_2_head_stats"].append({
                "opponent": bats,
                "opp_role": "batsman",
                "stats": stats,
                "advantage_score": (-1)*faceoff_score
            })

    # For each batsman/allrounder in Team B vs each bowler/allrounder in Team A
    for bats in tB_bats:
        for bowl in tA_bowl:
            stats = players_faceoff(bats, bowl)

            faceoff_score = compute_faceoff_score(stats) # a float value
            # if faceoff is +ve -> batsman domination, means same score with -ve sign will be added to bowler
            # if faceoff is -ve ->  bowler domination, means same score with +ve sign will be added to bowler

            lookup_B[bats]["head_2_head_stats"].append({
                "opponent": bowl,
                "opp_role": "bowler",
                "stats": stats,
                "advantage_score": faceoff_score
            })
            lookup_A[bowl]["head_2_head_stats"].append({
                "opponent": bats,
                "opp_role": "batsman",
                "stats": stats,
                "advantage_score": (-1)*faceoff_score
            })

    # now calc for head_2_head_score for a player
    def compute_player_h2h(player: Dict[str, Any]):
        role_lower = player["role"].lower()
        entries = player["head_2_head_stats"] # list of ll head_to_heads, for allrounder it contains both against batsman and against bowler, will use opp_role to get the respective head_2_head_stats

        # Determine which advantage_scores to average
        if "battingallrounder" in role_lower:
            scores_bat = [e["advantage_score"] for e in entries if e["opp_role"] == "bowler"]
            scores_bowl = [e["advantage_score"] for e in entries if e["opp_role"] == "batsman"]
            if not scores_bat and scores_bowl:
                return 0.0
            elif not scores_bat:
                return scores_bowl
            elif not scores_bowl:
                return scores_bat
            else:
                return (0.7 * scores_bat + 0.3 * scores_bowl) # weighted sum
            
        if "bowlingallrounder" in role_lower:
            scores_bat = [e["advantage_score"] for e in entries if e["opp_role"] == "bowler"]
            scores_bowl = [e["advantage_score"] for e in entries if e["opp_role"] == "batsman"]
            if not scores_bat and scores_bowl: # maybe a new player
                return 0.0
            elif not scores_bat:
                return scores_bowl
            elif not scores_bowl:
                return scores_bat
            else:
                return (0.3 * scores_bat + 0.7 * scores_bowl) # weighted sum  
        else:
            # Pure batsman or pure bowler: use all entries
            scores = [e["advantage_score"] for e in entries]

        # Avoid division by zero
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    for player in team_A:
        if "batsman" in player["role"].lower() or "allrounder" in player["role"].lower():
            player["bowler_type_stats"] = get_player_pace_spin_stats(player["name"])
        else:
            player["bowler_type_stats"] = { # As for bowlers we don't have any data against left and right, whicvh does not make much sense, hence we haven't included that
                "pace": {},
                "spin": {}
            }
        player["head_2_head_score"] = compute_player_h2h(player)
        player["pitch_score"]  = compute_pitch_score(player, pitch_cond)
    for player in team_B:
        if "batsman" in player["role"].lower() or "allrounder" in player["role"].lower():
            player["bowler_type_stats"] = get_player_pace_spin_stats(player["name"])
        else:
            player["bowler_type_stats"] = {
                "pace": {},
                "spin": {}
            }
        player["head_2_head_score"] = compute_player_h2h(player)
        player["pitch_score"]  = compute_pitch_score(player, pitch_cond)

    return team_A, team_B

form_accessor_agent = create_react_agent(
    model = LLM,
    tools = [head_2_head, tavily_search, duck_search],
    name = "form_accessor_agent",
    prompt = """
    You are the Form Accessor Agent, responsible for comprehensive head-to-head and form analysis of two teams.

    **Inputs**  
    - A list of two rosters (team_A and team_B), each a list of player dicts with keys:  
    `"name"`, `"role"`, `"is_wk"`, `"is_overseas"`, `"batting_style"`, `"bowling_style"`, and `"recent_stats"`.  
    - A raw pitch description string (e.g. “spin can play a role in today's match”).

    **Tools**  
    1. `tavily_search(query: str) → List[SearchResult]`  
    2. `duck_search(query: str) → List[SearchResult]`  
    3. `head_2_head(team_A: List[Dict], team_B: List[Dict], pitch_cond: str) → Tuple[List[Dict], List[Dict]]`  
    - Computes for every player:  
        • `head_2_head_stats` vs opponents (batsman vs bowler)  
        • `bowler_type_stats` (career splits vs pace/spin)  
        • `head_2_head_score` (avg advantage float)  
        • `pitch_score (fit‐to‐pitch metric)

    **How to Respond**  
    1. **Normalize** the raw pitch description into one of:  
    "seamer_friendly", "spin_friendly", "flat", or "balanced".
    - If no information is available about the pitch then make it None. 
    2. **Call** the `head_2_head` tool with your two team lists and the canonical `pitch_cond` string:  
    ```json
    {
        "tool": "head_2_head",
        "args": {
        "team_A": <team_A list>,
        "team_B": <team_B list>,
        "pitch_cond": "<seamer_friendly|spin_friendly|flat|balanced|None>"
        }
    } 
    **Note - You have to output the teams in the same format as returned by the head_2_head_tool, do not make any changes to it,
    strictly adhered to it.
    """
)

"""
result = form_accessor_agent.invoke({"messages": [{"role": "user", "content": 
    
    Team_A = 
    [
        {
        'name': 'Virat Kohli',
        'role': 'batsman',
        'is_wk': 'False',
        'is_overseas': 'False',
        'batting_style': 'Right Handed Bat',
        'bowling_style': 'Right-arm medium',
        'recent_stats': [{'title': 'last_8_innings_stats', 'data': {'Batting': {'Matches': 8, 'Innings': 8, 'Runs': 408, 'Balls': 278, 'Outs': 7, '4s': 46, '6s': 9, '50s': 5, '100s': 0, 'SR': 146.76, 'Avg': 58.29}}}, {'title': 'career_stats_vs_Punjab_Kings', 'data': {'Batting': {'Matches': 36, 'Innings': 36, 'Runs': 1159, 'Balls': 874, 'Outs': 32, '4s': 120, '6s': 33, '50s': 6, '100s': 1, 'SR': 132.6, 'Avg': 36.21}}}, {'title': 'career_stats_at_M_Chinnaswamy_Stadium', 'data': {'Batting': {'Matches': 109, 'Innings': 106, 'Runs': 3618, 'Balls': 2514, 'Outs': 92, '4s': 329, '6s': 154, '50s': 27, '100s': 4, 'SR': 143.91, 'Avg': 39.32}}},
        {
        'name': 'Hardik Pandya',
        'role': 'batting allrounder',
        'is_wk': 'False', 
        'is_overseas': 'False', 
        'batting_style': 'Right Handed Bat', 
        'bowling_style': 'Right-arm fast-medium', 
        'recent_stats': [{'title': 'last_8_innings_stats', 'data': {'Batting': {'Matches': 8, 'Innings': 7, 'Runs': 120, 'Balls': 76, 'Outs': 5, '4s': 9, '6s': 6, '50s': 0, '100s': 0, 'SR': 157.89, 'Avg': 24.0}, 'Bowling': {'Matches': 8, 'Innings': 7, 'Overs': 13.0, 'Maidens': 0, 'Runs': 146, 'Wkts': 3, 'Eco': 11.23, 'Avg': 48.67, 'SR': 26.0}}}, {'title': 'career_stats_vs_Royal_Challengers_Bengaluru', 'data': {'Batting': {'Matches': 18, 'Innings': 17, 'Runs': 361, 'Balls': 220, 'Outs': 8, '4s': 22, '6s': 26, '50s': 2, '100s': 0, 'SR': 164.09, 'Avg': 45.12}, 'Bowling': {'Matches': 18, 'innings': 12, 'Overs': 29.0, 'Maidens': 0, 'Runs': 303, 'Wkts': 7, 'Eco': 10.44, 'Avg': 43.28, 'SR': 24.86}}}, {'title': 'career_stats_at_M_Chinnaswamy_Stadium', 'data': {'Batting': {'Matches': 12, 'Innings': 9, 'Runs': 162, 'Balls': 112, 'Outs': 6, '4s': 12, '6s': 9, '50s': 1, '100s': 0, 'SR': 144.64, 'Avg': 27.0}, 'Bowling': {'Matches': 12, 'innings': 10, 'Overs': 27.0, 'Maidens': 0, 'Runs': 240, 'Wkts': 11, 'Eco': 8.88, 'Avg': 21.81, 'SR': 14.73}}}    
    ],
    Team_B = 
    [
    {
    'name': 'Jasprit Bumrah', 
    'role': 'bowler', 
    'is_wk': 'False', 
    'is_overseas': 'False', 
    'batting_style': 'Right Handed Bat', 
    'bowling_style': 'Right-arm fast', 
    'recent_stats': [{'title': 'last_8_innings_stats', 'data': {'Bowling': {'Matches': 8, 'Innings': 8, 'Overs': 31.2, 'Maidens': 0, 'Runs': 197, 'Wkts': 14, 'Eco': 6.31, 'Avg': 14.07, 'SR': 13.43}}}, {'title': 'career_stats_vs_Royal_Challengers_Bengaluru', 'data': {'Bowling': {'Matches': 20, 'innings': 20, 'Overs': 78.0, 'Maidens': 2, 'Runs': 581, 'Wkts': 29, 'Eco': 7.44, 'Avg': 20.03, 'SR': 16.14}}}, {'title': 'career_stats_at_M_Chinnaswamy_Stadium', 'data': {'Bowling': {'Matches': 10, 'innings': 10, 'Overs': 78.0, 'Maidens': 2, 'Runs': 581, 'Wkts': 29, 'Eco': 7.44, 'Avg': 20.03, 'SR': 16.14}}},
    {
    'name': 'Shreyas Iyer', 
    'role': 'batsman', 
    'is_wk': 'False', 
    'is_overseas': 'False', 
    'batting_style': 'Right Handed Bat', 
    'bowling_style': 'Right-arm legbreak', 
    'recent_stats': [{'title': 'last_8_innings_stats', 'data': {'Batting': {'Matches': 8, 'Innings': 8, 'Runs': 316, 'Balls': 187, 'Outs': 6, '4s': 25, '6s': 18, '50s': 3, '100s': 0, 'SR': 168.98, 'Avg': 52.67}}}, {'title': 'career_stats_vs_Royal_Challengers_Bengaluru', 'data': {'Batting': {'Matches': 18, 'Innings': 18, 'Runs': 409, 'Balls': 341, 'Outs': 17, '4s': 34, '6s': 13, '50s': 4, '100s': 0, 'SR': 119.94, 'Avg': 24.05}}}, {'title': 'career_stats_at_M_Chinnaswamy_Stadium', 'data': {'Batting': {'Matches': 11, 'Innings': 11, 'Runs': 305, 'Balls': 222, 'Outs': 9, '4s': 26, '6s': 14, '50s': 3, '100s': 0, 'SR': 137.38, 'Avg': 33.88}}}
    ],
    "The pitch will be quite slow, hence it can benefit the spinners.

    
    }]})
for message in result["messages"]:
    message.pretty_print()
    """