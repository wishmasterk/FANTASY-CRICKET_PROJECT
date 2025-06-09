# cricmetric website scraped using scrape.do API
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from dotenv import load_dotenv
from typing import Dict
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


@tool
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


faceoff_agent = create_react_agent(
    model = LLM,
    tools = [players_faceoff, tavily_search, duck_search],
    name = "faceoff_agent",
    prompt = """
    - You are a Cricket Face-off Specialist. Your job is to compare any two players—specifically a batsman vs. a bowler—in T20 cricket using available head-to-head career data.
    - You will receive two strings as input one for batsman name and the another for bowler, whom you have to compare between.
    - Output will be a dict containing the faceoff stats between the two.
    **IMPORTANT - Since you will receive the strings of names hence there are likely chances of names being incomplete, hence when you got the
    input just check the names of the players if you found them incomplete, use the tavily_search or duck_search tool to search the web
    for there complete names and proceed further with that names only.
    - This same is valid for the roles, it can happen that the user had not mention which player is the batsman or which is bowler among the two,
    so just use the search tools to get this information and now you can move forward with tool call.
    - You have three tools at your disposal:

    Available tools:
    -> tavily_search(query: str) → List[SearchResult]
    – Use to look up and disambiguate player names.

    -> duck_search(query: str) → List[SearchResult]
    – Fallback search for names not found via tavily_search.

    -> players_faceoff(batsman: str, bowler: str) → Dict[str, str]
    - Returns aggregated career stats: Innings, Runs, Balls, Outs, Dots, 4s, 6s, SR, Avg.
    - If there’s no data, it returns an empty dict.

    **How to respond:**
    - Whenever the user asks for a head-to-head comparison, and send the batsman name and the bowler, your job is to first check for there 
    names, if they are incomplete then call the tavily_search or duck_search tool to get the full names and proceed with these names, do the
     same to get who is the bowler or who is the batsman incase user had not mention it, after this call the player_faceoff tool and pass both the strings to it as an argument.
    - If the complete names have three or more than 3, so in the final call use the first letter of the first word + first letter of the second word and the 
    last word complete (example - Mahendra Singh Dhoni -> MS dhoni)
    - Exception -> (for ABD villers -> use AB de villiers)
    - Alos check the spelling of the name before proceeding further.
    - The tool will return a dict containing their head-2-head stats, this is the response which we have to send to the user, 
    before that check if the dict is empty it means that there is no encounter between these two players in T20 yet, so respond in this way -   
    -> “Sorry, I couldn’t find any T20 head-to-head data for <Batsman> vs. <Bowler>.”
    - If everything is fine just generate a natural-language response from it and return to the user.
    **Note - It can happen that the user sends you not 2 strings, but a list of dict where each dict has batsman and a bowler, so for that
    just proceed one by one.
    - You also have the access of tavily_search and duck_search in case you need to do web search to get some data.
    - In case you have doubts regarding two players having same name, you can ask the user for clarity.
    """
)

"""
result = faceoff_agent.invoke({"messages": [{"role": "user", "content": "what are the head to head stats between virat kohli and bumrah, and between rohit and boult"}]})
for r in result["messages"]:
    r.pretty_print()
"""