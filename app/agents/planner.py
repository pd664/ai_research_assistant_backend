from app.llm.generation import generate

def decide_next_action(query, memory_context):
    prompt = f"""
        You are an intelligent AI agent.

        User Query:
        {query}

        Previous Steps:
        {memory_context}

        Decide next step.

        Available actions:
        - web_search
        - vector_search
        - finish

        Respond in format:

        Thought: <your reasoning>
        Action: <one action>
        """

    response = generate(prompt)

    thought = ""
    action = "finish"

    for line in response.split("\n"):
        if "thought" in line.lower():
            thought = line.split(":", 1)[-1].strip()
        elif "action" in line.lower():
            action = line.split(":", 1)[-1].strip().lower()

    return thought, action