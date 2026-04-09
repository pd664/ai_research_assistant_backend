def build_context(chunks):
    return "\n\n".join([
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    ])