from app.llm.generation import generate

def score_answer(query, answer):
    prompt = f"""
Rate this answer from 1-10.

Question: {query}
Answer: {answer}

Only return a number.
"""
    return generate(prompt)