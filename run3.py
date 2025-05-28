import requests

player_name = "Virat Kohli"
query = f"{player_name} site:espncricinfo.com/player"
url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
response = requests.get(url, headers=headers)
print(response.status_code)
print(response.text[:1000])  # print first 1000 characters