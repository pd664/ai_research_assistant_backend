from app.llm.generation import generate

def decide_tool(query: str) -> str:
    prompt = f"""You are an AI agent deciding which tool to use.

Tools:
- vector_search: search internal uploaded documents
- web_search: search the internet for current/external info

Question: {query}

Reply with exactly one word — either: vector_search OR web_search"""

    decision = generate(prompt).lower()

    if "web_search" in decision:
        return "web_search"
    if "vector_search" in decision:
        return "vector_search"
    return "vector_search"  # safe default