# def check_grounding(answer, chunks):
#     context = " ".join([c["text"] for c in chunks]).lower()
#     answer = answer.lower()

#     # check if key parts of answer exist in context
#     words = answer.split()

#     match_count = sum(1 for w in words if w in context)

#     score = match_count / (len(words) + 1e-5)

#     return score > 0.5

def check_grounding(answer: str, chunks: list) -> bool:
    if not chunks or not answer:
        return True

    STOPWORDS = {
        "about", "after", "also", "although", "answer", "based", "because",
        "been", "before", "being", "between", "both", "cannot", "context",
        "could", "during", "either", "every", "found", "from", "given",
        "have", "hence", "however", "information", "into", "itself",
        "local", "might", "more", "most", "much", "never", "noted",
        "often", "only", "other", "over", "provides", "quite", "rather",
        "really", "should", "since", "some", "states", "still", "such",
        "than", "that", "their", "them", "then", "there", "therefore",
        "these", "they", "this", "those", "through", "under", "until",
        "upon", "using", "very", "well", "were", "what", "when", "where",
        "which", "while", "with", "within", "would", "your"
    }

    context = " ".join([c["text"] for c in chunks]).lower()
    answer_lower = answer.lower()

    # only check meaningful tokens: long words, numbers, emails — minus stopwords
    tokens = [
        w.strip(".,!?;:\"'()[]") for w in answer_lower.split()
        if (len(w) > 4 or "@" in w or w.isdigit())
        and w.strip(".,!?;:\"'()[]") not in STOPWORDS
    ]

    if not tokens:
        return True  # nothing meaningful to verify

    matches = sum(1 for t in tokens if t in context)
    ratio = matches / len(tokens)

    print(f"Grounding check: {matches}/{len(tokens)} tokens matched ({ratio:.2f})")

    # 0.4 threshold — allows for paraphrasing and synthesis
    return ratio > 0.4

# def check_grounding(answer, chunks):
#     context = " ".join([c["text"] for c in chunks]).lower()
#     answer = answer.lower()

#     # Extract important tokens (long words, emails, numbers)
#     tokens = [
#         w for w in answer.split()
#         if len(w) > 4 or "@" in w or w.isdigit()
#     ]

#     if not tokens:
#         return True  # nothing meaningful to check

#     matches = sum(1 for t in tokens if t in context)

#     return matches / len(tokens) > 0.6