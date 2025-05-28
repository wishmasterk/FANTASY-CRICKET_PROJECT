import requests
import re

def get_espn_player_id(player_name):
    query = f"{player_name} site:espncricinfo.com/cricketers"
    url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    # Look for ESPNcricinfo player URL
    match = re.search(r"espncricinfo\.com/cricketers/[^/]+-(\d+)", response.text)
    if match:
        player_id = match.group(1)
        return player_id
    return None

# Example usage:
name = "Sai Sudharsan"
player_id = get_espn_player_id(name)
if player_id:
    print(f"Player ID for {name}: {player_id}")
else:
    print("Player not found.")
