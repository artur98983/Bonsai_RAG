import os


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

BONSAI_URL = os.getenv(
    "BONSAI_URL",
    "http://localhost:8080"
)

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    "/data/rag.db"
)

PROJECTS_PATH = os.getenv(
    "PROJECTS_PATH",
    "/projects"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

COLLECTION_NAME = "code"
