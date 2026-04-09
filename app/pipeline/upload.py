import os
from app.ingestion.loader import load_doc
from app.ingestion.chunker import chunkker
import re
import pathlib
from dotenv import load_dotenv

load_dotenv()

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove weird chars
    text = text.replace("\u007f", " ")
    return text

async def upload_doc(file, embedder, vector_store, session_id):
    if file is None:
        raise TypeError("Please select a file")
    
    if not file.filename.endswith((".txt", ".pdf")):
        raise TypeError("Please select a valid file.")

    os.makedirs(os.getenv("BASE_DATA_DIR"), exist_ok=True)

    session_dir = os.path.join(os.getenv("BASE_DATA_DIR"), session_id)

    safe_name = pathlib.Path(file.filename).stem
    file_path = os.path.join(session_dir, f"{safe_name}{pathlib.Path(file.filename).suffix}")
    try:
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        pages = await load_doc(file_path)
        all_chunks = []
        metadata = []
        for page in pages:
            text = clean_text(page["page_content"])
            chunks = chunkker(text)
            if chunks:
                for chunk in chunks:
                    all_chunks.append(chunk)
                    metadata.append({
                        "text": chunk,
                        **page["metadata"]
                    })
        existing_texts = {m["text"] for m in vector_store.metadata.values()}
        unique_chunks = []
        unique_metadata = []
        for chunk, meta in zip(all_chunks, metadata):
            if meta["text"] not in existing_texts:
                unique_chunks.append(chunk)
                unique_metadata.append(meta)
        if not unique_chunks:
            return {"message": "File already indexed, no new content found."}
        
        embeddings = embedder.embed_documents(unique_chunks)
        vector_store.add(embeddings, unique_metadata)
        vector_store.save()
        return {"message": "File indexed successfully"}

    except Exception as e:
        return f"There is some error {e}"