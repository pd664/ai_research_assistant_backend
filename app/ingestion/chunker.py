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


def _recursive_split(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursively splits text using a priority list of separators,
    then merges small pieces into chunks respecting chunk_size and chunk_overlap.
    Replicates the core behaviour of RecursiveCharacterTextSplitter.
    """

    def _split_by_sep(text: str, sep: str) -> list[str]:
        """Split and re-attach the separator to the end of each piece (except the last)."""
        if sep == "":
            return list(text)
        parts = text.split(sep)
        result = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + sep)
            elif part:
                result.append(part)
        return result

    def _merge(splits: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
        """Merge small splits into chunks of up to chunk_size, with overlap."""
        chunks = []
        current: list[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if current_len + split_len > chunk_size and current:
                chunks.append("".join(current))
                # Carry over the overlap tail
                overlap: list[str] = []
                overlap_len = 0
                for piece in reversed(current):
                    if overlap_len + len(piece) <= chunk_overlap:
                        overlap.insert(0, piece)
                        overlap_len += len(piece)
                    else:
                        break
                current = overlap
                current_len = overlap_len
            current.append(split)
            current_len += split_len

        if current:
            chunks.append("".join(current))

        return chunks

    # Find the first separator that actually appears in the text
    sep_to_use = ""
    remaining_seps = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            sep_to_use = sep
            remaining_seps = separators[i + 1:]
            break

    splits = _split_by_sep(text, sep_to_use)

    # Recurse on pieces that are still too large
    good: list[str] = []
    for split in splits:
        if len(split) > chunk_size and remaining_seps:
            good.extend(_recursive_split(split, remaining_seps, chunk_size, chunk_overlap))
        else:
            good.append(split)

    return _merge(good, chunk_size, chunk_overlap)


def get_splitter_config(doc_type: str) -> dict:
    """Return best splitter config for detected doc type."""

    if doc_type == "narrative":
        return dict(
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            chunk_size=1000,
            chunk_overlap=150,
        )
    elif doc_type == "resume":
        return dict(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=500,
            chunk_overlap=100,
        )
    elif doc_type == "technical":
        return dict(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=600,
            chunk_overlap=120,
        )
    else:  # general
        return dict(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=700,
            chunk_overlap=100,
        )


def chunkker(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("Please provide text as a string.")

    if not text or not text.strip():
        raise ValueError("Please provide non-empty text for chunking.")

    try:
        doc_type = detect_doc_type(text)
        config = get_splitter_config(doc_type)
        chunks = _recursive_split(text, **config)

        # Post-process: drop chunks that are too short to be meaningful
        chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

        return chunks

    except Exception as e:
        raise RuntimeError(f"Chunking error: {e}")