def should_use_Web(query: str):
    keywords = ["latest", "news", "today", "current", "2026"]
    return any(k in query.lower() for k in keywords)

def is_good_reterival(results):
    if not results:
        return False
    return results[0].get("score", 0) > 0.75

def validate_answer(answer, chunks):

    context_text = " ".join([c["text"] for c in chunks]).lower()

    sentences = answer.lower().split(".")

    for sentence in sentences:
        words = sentence.split()

        # check if key words exist in context
        matches = sum(1 for w in words if w in context_text)

        if len(words) > 5 and matches / len(words) < 0.6:
            return False

    return True