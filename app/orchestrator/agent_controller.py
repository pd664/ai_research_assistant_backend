from app.agents.memory import AgentMemory
from app.agents.planner import decide_next_action
from app.agents.executor import execute_action

from app.llm.generation import generate
from app.utils.helpers import rerank_chunks, should_use_vector
from app.llm.prompt_builder import build_prompt
from app.evaluation.hallucination import check_grounding

MAX_STEPS = 4


async def agent_controller(query, embedder, vector_store):
    memory = AgentMemory()

    # ── Step 1: Fast RAG path ──────────────────────────────────────────────────
    vector_results = await execute_action("vector_search", query, embedder, vector_store)

    if should_use_vector(vector_results):
        chunks = rerank_chunks(vector_results, top_k=8)  # 8 not 5 — avoid cutting off relevant chunks
        prompt = build_prompt(query, chunks)
        answer = generate(prompt)

        is_not_found = "not found in context" in answer.strip().lower()
        is_grounded = check_grounding(answer, chunks)

        print(f"Fast RAG answer: {answer[:80]}... | not_found={is_not_found} | grounded={is_grounded}")

        if is_grounded and not is_not_found:
            return {
                "query": query,        # ← actual query, not "Answer"
                "answer": answer,
                "mode": "fast_rag",
                "sources": chunks
            }
        # if not_found or not grounded → fall through to agent loop below

    # ── Step 2: Agent Loop ─────────────────────────────────────────────────────
    all_chunks = []   # ← start empty, don't re-add vector_results (already tried above)
    mode = "agent"

    for step in range(MAX_STEPS):
        memory_context = memory.get_context()
        thought, action = decide_next_action(query, memory_context)

        print(f"Step {step+1} | Thought: {thought[:80]} | Action: {action}")

        if action == "finish":
            break

        results = await execute_action(action, query, embedder, vector_store)

        if not results:
            print(f"  No results from {action}, continuing...")
            memory.add_steps(thought, action, [])
            continue

        formatted = [
            {
                "text": r.get("text", ""),
                "source": r.get("source", action),
                "score": r.get("score", 0.6)
            }
            for r in results
        ]

        all_chunks.extend(formatted)
        memory.add_steps(thought, action, formatted)

    # ── Step 3: Final Answer ───────────────────────────────────────────────────

    # if agent gathered nothing, fall back to original vector results
    if not all_chunks and vector_results:
        all_chunks = vector_results
        mode = "fast_rag_fallback"

    if not all_chunks:
        return {
            "query": query,
            "answer": "I could not find any relevant information to answer this question.",
            "mode": "no_results",
            "sources": [],
            "steps": memory.steps
        }

    chunks = rerank_chunks(all_chunks, top_k=8)
    prompt = build_prompt(query, chunks)
    answer = generate(prompt)

    is_not_found = "not found in context" in answer.strip().lower()
    is_grounded = check_grounding(answer, chunks)

    # only replace answer if BOTH checks fail — don't be too aggressive
    if not is_grounded and is_not_found:
        answer = "I could not find a reliable answer based on available sources."

    return {
        "query": query,        # ← actual query
        "answer": answer,
        "mode": mode,
        "sources": chunks,
        "steps": memory.steps
    }
