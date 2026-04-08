from tavily import TavilyClient
import os 
from dotenv import load_dotenv

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=tavily_api_key)

def web_search(query: str):
    response = client.search(
        query=query,
        max_results=5
    )

    results = []

    for r in response["results"]:
        results.append({
            "text": r["title"] + "\n" + r["content"],
            "source": r["url"],
            "score": 0.6
        })

    return results