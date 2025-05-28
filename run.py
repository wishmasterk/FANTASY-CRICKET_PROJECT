import requests

url = "https://stats.espncricinfo.com/ci/engine/player/253802.html?class=6;opposition=4346;template=results;type=allround;view=match"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
print(response.status_code)
print(response.text[:1000])  # print first 1000 characters
