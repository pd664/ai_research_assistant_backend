from app.llm.generation import generate

def build_entity_hint(chunks: list[str]) -> str:
    """
    Scan retrieved chunks for names and build a short hint for the LLM.
    Works for any document — stories, resumes, technical docs.
    """
    import re
    # Find capitalized names (simple heuristic)
    names = set()
    for chunk in chunks:
        found = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', chunk)
        names.update(found)

    # Filter out common non-name capitalized words
    stopwords = {"The", "A", "An", "In", "On", "At", "By", "For", "And", "But",
                 "He", "She", "It", "We", "They", "His", "Her", "Their", "This",
                 "That", "With", "From", "Into", "When", "Then", "There"}
    names = [n for n in names if n not in stopwords and len(n) > 2]

    if not names:
        return ""

    return f"Characters or entities mentioned in the document: {', '.join(sorted(names)[:12])}."


async def ask(question, embedder, vector_store):
    try:
        if not question:
            return []

        q_embeddings = embedder.embed_query(question)  # fix the [question] bug too
        results = vector_store.search(q_embeddings, top_k=8)

        if not results:
            return []

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return [
            {
                "text": r["text"],
                "source": r.get("source", "internal"),
                "score": r["score"]
            }
            for r in results
        ]
    except Exception as e:
        return {"error": str(e)}
