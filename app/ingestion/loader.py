import fitz

async def load_doc(file_path: str):
    pages = []

    if file_path.endswith(".pdf"):
        pages = load_pdf(file_path)

    if file_path.endswith(".txt"):
        pages = load_txt(file_path)

    return pages

def load_pdf(file_path):
    if not file_path:
        raise ValueError("Please select a file first")

    try:
        doc = fitz.open(file_path)
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text()

            pages.append({
                "page_content": text,
                "metadata": {
                    "page": page_num,
                    "source": file_path
                }
            })

        return pages

    except Exception as e:
        raise RuntimeError(f"There is some error: {e}")

def load_txt(file_path):
    if not file_path:
        raise ValueError("Please select a file first")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return [{
            "page_content": content,
            "metadata": {"source": file_path}
        }]


    except Exception as e:
        raise RuntimeError(f"There is some error: {e}")