import faiss
import os
import json
import numpy as np


class VectorStore:
    def __init__(self, dim: int, index_path: str, metadata_path: str):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.npy_path = index_path.replace(".faiss", ".npy")

        self.index = faiss.IndexFlatIP(dim)
        self.metadata = {}
        self.next_id = 0

        self._load()

    # ---------------- LOAD ----------------
    def _load(self):
        try:
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)

                if self.metadata:
                    self.next_id = max(int(k) for k in self.metadata.keys()) + 1

            # Load embeddings
            if os.path.exists(self.npy_path):
                embeddings = np.load(self.npy_path, allow_pickle=False)

                if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
                    raise ValueError("Corrupted embeddings file")

                if embeddings.shape[0] > 0:
                    self.index.add(embeddings)

        except Exception as e:
            raise RuntimeError(f"Failed to load vector store: {str(e)}")

    # ---------------- ADD ----------------
    def add(self, embeddings, metadata_list):
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dim:
            raise ValueError("Embedding dimension mismatch")

        if len(metadata_list) != embeddings.shape[0]:
            raise ValueError("Metadata length mismatch")

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(embeddings)

        # Store metadata
        for i, meta in enumerate(metadata_list):
            self.metadata[str(self.next_id)] = meta
            self.next_id += 1

    # ---------------- SEARCH ----------------
    def search(self, q_embeddings, top_k=5):
        if self.index.ntotal == 0:
            return []

        q_embeddings = np.asarray(q_embeddings, dtype="float32")

        if q_embeddings.ndim == 1:
            q_embeddings = q_embeddings.reshape(1, -1)

        # Normalize
        norms = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        q_embeddings = q_embeddings / norms

        q_embeddings = np.ascontiguousarray(q_embeddings, dtype=np.float32)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_embeddings, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            meta = self.metadata.get(str(idx))
            if not meta:
                continue

            item = meta.copy()
            item["score"] = float(score)
            results.append(item)

        return results

    # ---------------- SAVE ----------------
    def save(self):
        try:
            print(21)
            os.makedirs(os.path.dirname(self.npy_path), exist_ok=True)
            print(22)
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            print(23)

            if self.index.ntotal == 0:
                # Save empty state safely
                print(24)
                np.save(self.npy_path, np.empty((0, self.dim), dtype="float32"))
                print(25)
            else:
                print(26)
                embeddings = faiss.rev_swig_ptr(
                    self.index.get_xb(),
                    self.index.ntotal * self.dim
                )
                print(27)
                embeddings = np.array(embeddings, dtype="float32").reshape(
                    self.index.ntotal, self.dim
                )
                print(28)
                np.save(self.npy_path, embeddings)
                print(29)

            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
                print(30)
            print(31)

        except Exception as e:
            raise RuntimeError(f"Failed to save vector store: {str(e)}")
        
# import faiss
# import os
# import json
# import numpy as np

# class VectorStore:
#     def __init__(self, dim, index_path, metadata_path):
#         self.dim = dim
#         self.metadata_path = metadata_path
#         # Separate paths for npy and faiss
#         self.npy_path = index_path.replace(".faiss", ".npy")

#         self.index = faiss.IndexFlatIP(dim)

#         if os.path.exists(metadata_path):
#             with open(metadata_path, "r") as f:
#                 self.metadata = json.load(f)
#         else:
#             self.metadata = {}

#         if self.metadata:
#             self.next_id = max(int(k) for k in self.metadata.keys()) + 1
#         else:
#             self.next_id = 0

#         # Load embeddings ONLY from npy, never from faiss binary
#         if os.path.exists(self.npy_path):
#             embeddings = np.load(self.npy_path, allow_pickle=False)
#             if embeddings.shape[0] > 0:
#                 self.index.add(embeddings)

#     def add(self, embeddings, metadata_list):
#         embeddings = np.asarray(embeddings, dtype="float32")
#         embeddings = np.ascontiguousarray(embeddings)

#         if embeddings.ndim == 1:
#             embeddings = embeddings.reshape(1, -1)

#         if embeddings.shape[1] != self.dim:
#             raise ValueError("Dimensions mismatched")

#         if len(metadata_list) != embeddings.shape[0]:
#             raise ValueError("Metadata length mismatched")

#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         norms = np.where(norms == 0, 1, norms)
#         embeddings = np.ascontiguousarray(embeddings / norms, dtype=np.float32)

#         self.index.add(embeddings)

#         for idx, meta in enumerate(metadata_list):
#             self.metadata[str(self.next_id + idx)] = meta

#         self.next_id += len(metadata_list)

#     def search(self, q_embeddings, top_k=5):
#         if self.index.ntotal == 0:
#             return []

#         q_embeddings = np.asarray(q_embeddings, dtype="float32")
#         q_embeddings = np.ascontiguousarray(q_embeddings)
#         if q_embeddings.ndim == 1:
#             q_embeddings = q_embeddings.reshape(1, -1)

#         norms = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
        
#         norms = np.where(norms == 0, 1, norms)
#         q_embeddings = np.ascontiguousarray(q_embeddings / norms, dtype=np.float32)

#         top_k = min(top_k, self.index.ntotal)
#         scores, indices = self.index.search(q_embeddings, top_k)

#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             if idx < 0:
#                 continue
#             meta = self.metadata.get(str(int(idx)))
#             if meta is None:
#                 continue
#             item = meta.copy()
#             item["score"] = float(score)
#             results.append(item)
#         return results

#     def save(self):
#         os.makedirs(os.path.dirname(self.npy_path), exist_ok=True)
#         os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

#         # Save all embeddings as npy
#         embeddings = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * self.dim)
#         embeddings = np.array(embeddings, dtype="float32").reshape(self.index.ntotal, self.dim)
#         np.save(self.npy_path, embeddings)

#         with open(self.metadata_path, "w") as f:
#             json.dump(self.metadata, f, indent=2)
