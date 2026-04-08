from langchain_text_splitters import RecursiveCharacterTextSplitter

def detect_doc_type(text: str) -> str:
    """Heuristically detect document type."""
    text_lower = text.lower()

    # Resume signals
    resume_keywords = ["experience", "education", "skills", "objective", "resume", "curriculum vitae", "references"]
    resume_hits = sum(1 for k in resume_keywords if k in text_lower)
    if resume_hits >= 3:
        return "resume"

    # Code/technical doc signals
    code_keywords = ["def ", "class ", "import ", "function", "return", "const ", "var ", "http"]
    code_hits = sum(1 for k in code_keywords if k in text)
    if code_hits >= 3:
        return "technical"

    # Story/narrative signals — long paragraphs, dialogue
    lines = text.split("\n")
    avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)
    has_dialogue = text.count('"') > 5 or text.count("'") > 5
    if avg_line_len > 100 or has_dialogue:
        return "narrative"

    return "general"


def get_splitter(doc_type: str) -> RecursiveCharacterTextSplitter:
    """Return best splitter config for detected doc type."""

    if doc_type == "narrative":
        # Stories need larger chunks to preserve context across dialogue
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            chunk_size=1000,
            chunk_overlap=150
        )

    elif doc_type == "resume":
        # Resumes are structured — split on sections, keep sections intact
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=500,
            chunk_overlap=100
        )

    elif doc_type == "technical":
        # Technical docs need context but split cleanly on paragraphs
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=600,
            chunk_overlap=120
        )

    else:  # general
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=700,
            chunk_overlap=100
        )


def chunkker(text: str) -> list[str]:
    if not text or not text.strip():
        raise ValueError("Please provide non-empty text for chunking.")

    if not isinstance(text, str):
        raise ValueError("Please provide text as a string.")

    try:
        doc_type = detect_doc_type(text)

        splitter = get_splitter(doc_type)
        chunks = splitter.split_text(text)

        # Post-process: drop chunks that are too short to be meaningful
        chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

        return chunks

    except Exception as e:
        raise RuntimeError(f"Chunking error: {e}")
