from app.llm.generation import generate

def decide_next_action(query, memory_context):
    prompt = f"""
You are an intelligent AI agent with access to a personal document store (vector_search) and the web (web_search).

TOOL PRIORITY RULES — follow strictly:
1. ALWAYS try vector_search first if the query is about a person, resume, document, or uploaded content.
2. Only use web_search if vector_search returned no useful results OR the query needs current/external information.
3. Use finish when you have enough information to answer.

User Query:
{query}

Previous Steps:
{memory_context}

Available actions:
- vector_search  ← use this first for document/resume questions
- web_search     ← use only if vector_search failed or query needs external info
- finish         ← use when you have enough context to answer

Respond in this exact format:
Thought: <your reasoning>
Action: <one action from the list above>
"""

    response = generate(prompt)

    thought = ""
    action = "finish"

    for line in response.split("\n"):
        if line.lower().startswith("thought"):
            thought = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("action"):
            action = line.split(":", 1)[-1].strip().lower()

    return thought, action
