from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool
from collections import defaultdict
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union, Any, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup
from langchain.tools import DuckDuckGoSearchRun
import difflib
import time
import requests
import urllib.parse
import re
import os


def _name_variants(full_name: str) -> list:
    """
    Generate possible abbreviated variants for a multi-word name.
    - If two words (First Last): ["First Last", "F Last"]
    - If three words (A B C): ["A B C", "AB C", "A BC"]
    Additional variants can be added as needed.
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


def fetch_cricmetric_table_via_scrapedo(
    batsman: str, bowler: str
) -> str:
    """
    1. Tries full `batsman` vs `bowler` name pair on CricMetric via Scrape.do.
    2. If no <table> found, tries abbreviated variants (e.g., "F Last", "AB C").
    3. Filters for only the “T20I” and “TWENTY20” tables on the page (using the
       enclosing panel-heading text).
    4. Returns raw HTML consisting of those filtered <table class="table">…</table>
       blocks concatenated, or "" if none matched.
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

    # No table for full names: generate and try variants
    bats_variants = _name_variants(batsman)
    bowl_variants = _name_variants(bowler)

    for bv in bats_variants:
        for ov in bowl_variants:
            if bv == batsman and ov == bowler:
                continue
            table_html = attempt_fetch(bv, ov)
            if table_html:
                return table_html

    # If none worked, return empty
    return ""


def parse_cricmetric_total_row(table_html: str, batsman: str, bowler: str) -> Dict[str, str]:
    """
    Given HTML containing one or more <table class="table"> blocks (already
    filtered to only T20I/TWENTY20 by fetch_cricmetric_table_via_scrapedo), this:
      1) Parses each table’s header row to get column names.
      2) Sums up <tbody> row counts across tables to get total “Innings.”
      3) Extracts each <tfoot><tr> “Total” row from every table, converts numeric cells,
         and aggregates them column‐wise.
      4) Computes combined Strike Rate and Average from aggregated Runs, Balls, Outs.
      5) Returns a dict:
         {
           "Title": "Batsman V/S Bowler",
           "Stats": {
             "Innings": "<total_innings>",
             "Runs": "<sum_of_runs>",
             "Balls": "<sum_of_balls>",
             "Outs": "<sum_of_outs>",
             "Dots": "<sum_of_dots>",
             "4s": "<sum_of_4s>",
             "6s": "<sum_of_6s>",
             "SR": "<computed_SR>",
             "Avg": "<computed_Avg>"
           }
         }
    """
    result: Dict[str, Any] = {}
    result["Title"] = f"{batsman} V/S {bowler}"
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

    result["Stats"] = stats
    return result


def players_faceoff(
    batsman: str, bowler: str
) -> Dict[str, str]:
    """
    High-level helper that:
      1) Calls `fetch_cricmetric_table_via_scrapedo(...)` to retrieve the <table> HTML.
      2) If no table HTML is returned, returns an empty dict.
      3) Otherwise calls `parse_cricmetric_total_row(...)` to extract the “Total” row + match count.
      4) Returns the combined dict.
    """
    table_html = fetch_cricmetric_table_via_scrapedo(batsman, bowler)
    if not table_html:
        return {}
    return parse_cricmetric_total_row(table_html, batsman, bowler)
