from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

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
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

    except Exception as e:
        raise RuntimeError(f"There is some error: {e}")
    

def load_txt(file_path):
    if not file_path:
        raise ValueError("Please select a file first")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return [Document(page_content=content, metadata={"source": file_path})]

    except Exception as e:
        raise RuntimeError(f"There is some error: {e}")