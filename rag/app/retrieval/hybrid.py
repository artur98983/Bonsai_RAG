from .exact import find_explicit_symbols
from .semantic import semantic_search


def hybrid_search(
    project_id: int,
    query: str,
    limit: int = 8,
) -> list[dict]:
    """
    Combine exact symbol retrieval and semantic retrieval.

    Exact symbol matches have priority.
    """

    explicit_symbols = find_explicit_symbols(
        project_id,
        query,
    )

    explicit_results = []

    for symbol in explicit_symbols:
        explicit_results.append(
            {
                "id": symbol["id"],
                "score": 1.0,
                "project_id": project_id,
                "file": symbol["file"],
                "symbol": symbol["qualified_name"],
                "symbol_type": symbol["type"],
                "language": symbol["language"],
                "start_line": symbol["start_line"],
                "end_line": symbol["end_line"],
                "text": symbol["code"] or "",
                "retrieval": "exact_symbol",
            }
        )

    semantic_results = semantic_search(
        project_id,
        query,
        limit,
    )

    output = []
    seen_ids = set()

    for item in explicit_results:
        symbol_id = item["id"]

        if symbol_id in seen_ids:
            continue

        seen_ids.add(symbol_id)
        output.append(item)

    for item in semantic_results:
        symbol_id = item.get("id")

        if symbol_id in seen_ids:
            continue

        seen_ids.add(symbol_id)
        output.append(item)

    return output[:limit]
