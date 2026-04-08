def rerank_chunks(chunks, top_k=8):
    return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

def get_retrieval_confidence(results):
    if not results:
        return 0.0
    top_score = results[0]["score"]

    return top_score

def should_use_vector(results, threshold=0.5):

    if not results:
        return False
    top_score = get_retrieval_confidence(results)

    print(f"Top score: {top_score:.2f}")

    return top_score >= threshold