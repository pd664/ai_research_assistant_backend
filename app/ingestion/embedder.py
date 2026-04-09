import numpy as np
import httpx
import os

class Embedder:
    def __init__(self):
        self.api_key = os.getenv("EMBEDDING_API_KEY")

    def embed_documents(self, chunks):
        return np.array([self._embed(text) for text in chunks], dtype="float32")

    def embed_query(self, query):
        return np.array([self._embed(query)], dtype="float32")

    def _embed(self, text):
        response = httpx.post(
            "https://api.openai.com/v1/embeddings",  # or groq if supported
            headers={
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "text-embedding-3-small",
                "input": text
            }
        )
        return response.json()["data"][0]["embedding"]
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