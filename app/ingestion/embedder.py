from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


class Embedder:

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name
        )
    def embed_documents(self, chunks):
        vectors = self.model.embed_documents(chunks)
        return np.array(vectors, dtype="float32")

    def embed_query(self, query):
        instruction = "Represent this sentence for searching relevant passages: "
        query = instruction + query
        vector = self.model.embed_query(query)
        return np.array([vector], dtype="float32")