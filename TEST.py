import requests, os

SERPAPI_KEY = os.getenv("SERPAPI_KEY") or "paste-your-key-here"
url = "https://serpapi.com/search.json"
params = {"engine": "google_news", "q": "Narendra Modi", "api_key": "799569563913383aff03e5244e447a0d37ec15b80ff617051fa759281c79e4a0"}
r = requests.get(url, params=params)
print(r.status_code)
print(r.json().get("news_results", [])[:2])
