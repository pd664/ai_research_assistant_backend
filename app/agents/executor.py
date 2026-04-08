from app.tools.web_search import web_search
from app.pipeline.ask import ask

async def execute_action(action, query, embedder, vector_store):
    if action == "web_search":
        return web_search(query)

    elif action == "vector_search":
        return await ask(query, embedder, vector_store)

    return []