import cohere
import numpy as np
import os
import time
import random
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not set")
        self.client = cohere.Client(api_key)

    def embed_documents(self, chunks: list[str], batch_size: int = 90) -> np.ndarray:
        """
        Cohere free tier allows 100 texts per request, keeping at 90 to be safe.
        """
        all_embeddings = []
        total_batches = -(-len(chunks) // batch_size)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1} / {total_batches}")

            embeddings = self._embed_with_retry(batch, input_type="search_document")
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        embeddings = self._embed_with_retry([query], input_type="search_query")
        return np.array(embeddings, dtype="float32")

    def _embed_with_retry(self, texts: list[str], input_type: str, retries: int = 5):
        for attempt in range(retries):
            try:
                response = self.client.embed(
                    texts=texts,
                    model="embed-english-v3.0",
                    input_type=input_type,
                )
                return response.embeddings

            except cohere.TooManyRequestsError:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited. Retrying in {wait:.1f}s...")
                time.sleep(wait)

            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                raise

        raise RuntimeError("Max retries exceeded for embedding request")
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import numpy as np


# class Embedder:
#     def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
#         self.model = HuggingFaceEmbeddings(
#             model_name=model_name
#         )
#     def embed_documents(self, chunks):
#         vectors = self.model.embed_documents(chunks)
#         return np.array(vectors, dtype="float32")

#     def embed_query(self, query):
#         instruction = "Represent this sentence for searching relevant passages: "
#         query = instruction + query
#         vector = self.model.embed_query(query)
#         return np.array([vector], dtype="float32")