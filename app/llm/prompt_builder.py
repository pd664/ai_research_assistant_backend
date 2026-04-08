def build_prompt(query, chunks):
    context = "\n\n".join([
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    ])

    return f"""You are a highly intelligent AI assistant specialized in answering questions using retrieved context.

Your goal is to provide the most accurate and complete answer possible using ONLY the context provided below.

---------------------
INSTRUCTIONS:
---------------------
1. Use ONLY the provided context. Do NOT use outside knowledge.
2. The answer may NOT be explicitly stated — you MUST infer and synthesize from multiple sources.
3. Combine information across chunks to form a coherent answer.
4. Always support your answer with citations like [1], [2].
5. If partial information exists, you MUST still answer using best possible inference.
6. If ANY relevant information exists, you MUST provide an answer.
7. Do NOT say "Not found" if indirect or partial evidence is present.
8. The answer may be implicit — extracting meaning is part of your task.
9. Be concise but complete.

---------------------
CRITICAL RULE:
---------------------
You are NOT allowed to refuse if relevant information exists.
You MUST produce an answer by reasoning over the context.

---------------------
REASONING STRATEGY:
---------------------
- Identify relevant information
- Connect ideas across sources
- Infer missing links
- Produce final answer

---------------------
CONTEXT:
---------------------
{context}

---------------------
QUESTION:
---------------------
{query}

---------------------
ANSWER:
---------------------
"""