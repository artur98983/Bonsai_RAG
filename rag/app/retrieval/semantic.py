from qdrant_client import QdrantClient

from ..config import QDRANT_URL, COLLECTION_NAME
from ..embeddings import embedding_model


qdrant = QdrantClient(url=QDRANT_URL)


def semantic_search(
    project_id: int,
    query: str,
    limit: int = 8,
) -> list[dict]:
    vector = embedding_model.encode(
        [query]
    )[0].tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter={
            "must": [
                {
                    "key": "project_id",
                    "match": {
                        "value": project_id
                    },
                }
            ]
        },
        limit=limit,
        with_payload=True,
    )

    output = []

    for point in results.points:
        payload = point.payload or {}

        output.append(
            {
                "id": point.id,
                "score": point.score,
                **payload,
                "retrieval": "semantic",
            }
        )

    return output
