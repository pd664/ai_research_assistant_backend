import os
from app.ingestion.loader import load_doc
from app.ingestion.chunker import chunkker
import re
import pathlib
import logging

logger = logging.getLogger(__name__)

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.replace("\u007f", " ")
    return text


async def upload_doc(file, embedder, vector_store, session_id):
    if file is None:
        raise TypeError("Please select a file")

    if not file.filename.endswith((".txt", ".pdf")):
        raise TypeError("Please select a valid file.")

    base_dir = os.getenv("BASE_DATA_DIR", "data")
    os.makedirs(base_dir, exist_ok=True)

    session_dir = os.path.join(base_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)

    safe_name = pathlib.Path(file.filename).stem
    file_path = os.path.join(
        session_dir,
        f"{safe_name}{pathlib.Path(file.filename).suffix}"
    )

    try:
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        print("File saved successfully")

        # Load document
        pages = await load_doc(file_path)
        print(f"Loaded {len(pages)} pages")

        all_chunks = []
        metadata = []

        for page in pages:
            text = clean_text(page["page_content"])
            chunks = chunkker(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "text": chunk,
                    **page["metadata"]
                })

        print(f"Total chunks: {len(all_chunks)}")

        # Remove duplicates
        existing_texts = {m["text"] for m in vector_store.metadata.values()}

        unique_chunks = []
        unique_metadata = []

        for chunk, meta in zip(all_chunks, metadata):
            if meta["text"] not in existing_texts:
                unique_chunks.append(chunk)
                unique_metadata.append(meta)

        if not unique_chunks:
            return {"message": "File already indexed, no new content found."}

        print(f"Unique chunks: {len(unique_chunks)}")

        # 🚀 Embeddings (batched + safe)
        embeddings = embedder.embed_documents(unique_chunks)

        # Store
        vector_store.add(embeddings, unique_metadata)
        vector_store.save()

        print("Indexing completed successfully")

        return {"message": "File indexed successfully"}

    except Exception as e:
        logger.exception("Upload failed")
        return {"error": str(e)}