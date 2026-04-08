import os
from dotenv import load_dotenv

load_dotenv()


def get_session_paths(session_id: str):
    session_dir = os.path.join(os.getenv("BASE_DATA_DIR"), session_id)
    os.makedirs(session_dir, exist_ok=True)

    index_path = os.path.join(session_dir, "index.npy")
    metadata_path = os.path.join(session_dir, "metadata.json")

    return index_path, metadata_path