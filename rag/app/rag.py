import httpx

from qdrant_client import QdrantClient

from .config import (
    QDRANT_URL,
    BONSAI_URL,
    COLLECTION_NAME,
)

from .database import (
    get_symbol,
    get_symbol_by_id,
    get_callees,
    get_callers,
    get_children,
)

from .embeddings import embedding_model


# ---------------------------------------------------------
# Qdrant
# ---------------------------------------------------------

qdrant = QdrantClient(url=QDRANT_URL)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def symbol_to_dict(row):
    if row is None:
        return None

    return {
        "id": row["id"],
        "file": row["file"],
        "name": row["name"],
        "qualified_name": row["qualified_name"],
        "type": row["type"],
        "language": row["language"],
        "parent_symbol_id": row["parent_symbol_id"],
        "parent_qualified_name": (
            row["parent_qualified_name"]
            if "parent_qualified_name" in row.keys()
            else None
        ),
        "start_line": row["start_line"],
        "end_line": row["end_line"],
        "code": row["code"],
    }

def find_explicit_symbols(project_id, query):
    """
    Find symbols explicitly mentioned in the user's query.

    We extract identifier-like tokens from the query and
    resolve them against SQLite.
    """

    import re

    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_.]*",
        query,
    )

    symbols = []

    seen = set()

    for token in tokens:
        symbol = get_symbol(
            project_id,
            token,
        )

        if symbol is None:
            continue

        symbol_id = symbol["id"]

        if symbol_id in seen:
            continue

        seen.add(symbol_id)
        symbols.append(symbol)

    return symbols

def caller_to_dict(row):
    """
    Convert a code reference into the symbol that calls
    the requested symbol.
    """

    return {
        "id": row["id"],
        "name": row["source_symbol"],
        "qualified_name": row["source_qualified_name"],
        "file": row["source_file"],
        "start_line": row["source_start_line"],
        "end_line": row["source_end_line"],
        "reference_type": row["reference_type"],
    }


def callee_to_dict(row):
    """
    Convert a code reference into the symbol called by
    the requested symbol.

    For external functions that are not present in the
    symbols table, target_symbol is NULL and target_name
    is used instead.
    """

    target_symbol = row["target_symbol"]

    return {
        "id": row["id"],
        "name": (
            target_symbol
            if target_symbol is not None
            else row["target_name"]
        ),
        "qualified_name": row["target_qualified_name"],
        "file": row["target_file"],
        "start_line": row["target_start_line"],
        "end_line": row["target_end_line"],
        "reference_type": row["reference_type"],
    }


# ---------------------------------------------------------
# Semantic search
# ---------------------------------------------------------

def search(project_id, query, limit=8):
    """
    Hybrid code search.

    Retrieval consists of:

    1. Explicit symbol detection using SQLite.
    2. Semantic search using Qdrant.

    Explicitly mentioned symbols receive priority because
    code identifiers are much more precise than semantic
    similarity alone.
    """

    # -----------------------------------------------------
    # Explicit symbols
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Semantic search
    # -----------------------------------------------------

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

    semantic_results = []

    for point in results.points:
        payload = point.payload or {}

        semantic_results.append(
            {
                "id": point.id,
                "score": point.score,
                **payload,
                "retrieval": "semantic",
            }
        )

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    output = []
    seen_ids = set()

    # Exact symbols always come first.
    for item in explicit_results:

        symbol_id = item["id"]

        if symbol_id in seen_ids:
            continue

        seen_ids.add(symbol_id)
        output.append(item)

    # Then semantic results.
    for item in semantic_results:

        symbol_id = item.get("id")

        if symbol_id in seen_ids:
            continue

        seen_ids.add(symbol_id)
        output.append(item)

    return output[:limit]

# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------

def get_graph_context(project_id, symbol_name):
    """
    Build a structural graph around a symbol.

    Includes:

    - symbol itself
    - parent
    - children
    - callers
    - callees
    """

    symbol = get_symbol(
        project_id,
        symbol_name
    )

    if symbol is None:
        return None

    result = {
        "symbol": symbol_to_dict(symbol),
        "parent": None,
        "children": [],
        "callers": [],
        "callees": [],
    }

    # -----------------------------------------------------
    # Parent
    # -----------------------------------------------------

    parent_symbol_id = symbol["parent_symbol_id"]

    if parent_symbol_id is not None:
        parent = get_symbol_by_id(
            project_id,
            parent_symbol_id
        )

        if parent is not None:
            result["parent"] = symbol_to_dict(parent)

    # -----------------------------------------------------
    # Children
    # -----------------------------------------------------

    children = get_children(
        project_id,
        symbol["id"]
    )

    result["children"] = [
        symbol_to_dict(row)
        for row in children
    ]

    # -----------------------------------------------------
    # Callers
    # -----------------------------------------------------

    callers = get_callers(
        project_id,
        symbol["qualified_name"]
    )

    result["callers"] = [
        caller_to_dict(row)
        for row in callers
    ]

    # -----------------------------------------------------
    # Callees
    # -----------------------------------------------------

    callees = get_callees(
        project_id,
        symbol["qualified_name"]
    )

    result["callees"] = [
        callee_to_dict(row)
        for row in callees
    ]

    return result


# ---------------------------------------------------------
# Graph expansion for RAG
# ---------------------------------------------------------

def expand_graph(project_id, search_results):
    """
    Expand semantic search results with structural
    information from SQLite.

    Qdrant payload currently contains:
        id
        project_id
        file
        symbol
        symbol_type
        language
        start_line
        end_line
        text

    The Qdrant point ID corresponds to the SQLite
    symbol ID, so we use item["id"] first.

    Fallback:
        item["symbol"] -> SQLite symbol lookup
    """

    expanded = []

    seen_symbols = set()

    for item in search_results:
        symbol = None

        # -------------------------------------------------
        # Primary lookup: Qdrant point ID == SQLite symbol ID
        # -------------------------------------------------

        symbol_id = item.get("id")

        if symbol_id is not None:
            try:
                symbol = get_symbol_by_id(
                    project_id,
                    int(symbol_id)
                )
            except (TypeError, ValueError):
                symbol = None

        # -------------------------------------------------
        # Fallback: lookup by symbol name
        # -------------------------------------------------

        if symbol is None:
            symbol_name = item.get("symbol")

            if symbol_name:
                symbol = get_symbol(
                    project_id,
                    symbol_name
                )

        # -------------------------------------------------
        # Could not resolve symbol
        # -------------------------------------------------

        if symbol is None:
            expanded.append(
                {
                    "search": item,
                    "graph": None,
                }
            )
            continue

        # -------------------------------------------------
        # Avoid duplicate graph entries
        # -------------------------------------------------

        symbol_key = symbol["id"]

        if symbol_key in seen_symbols:
            continue

        seen_symbols.add(symbol_key)

        # -------------------------------------------------
        # Build structural graph
        # -------------------------------------------------

        graph = get_graph_context(
            project_id,
            symbol["qualified_name"]
        )

        expanded.append(
            {
                "search": item,
                "graph": graph,
            }
        )

    return expanded

# ---------------------------------------------------------
# Text representation of graph context
# ---------------------------------------------------------

def build_graph_context_text(graph_context):
    """
    Convert structural graph information into compact text
    suitable for the LLM prompt.
    """

    if not graph_context:
        return ""

    chunks = []

    for item in graph_context:
        graph = item.get("graph")

        if not graph:
            continue

        symbol = graph.get("symbol")

        if not symbol:
            continue

        chunks.append(
            "SYMBOL\n"
            f"name: {symbol.get('qualified_name')}\n"
            f"type: {symbol.get('type')}\n"
            f"file: {symbol.get('file')}\n"
            f"lines: {symbol.get('start_line')}-"
            f"{symbol.get('end_line')}\n"
            f"code:\n{symbol.get('code') or ''}"
        )

        # -------------------------------------------------
        # Parent
        # -------------------------------------------------

        parent = graph.get("parent")

        if parent:
            chunks.append(
                "PARENT\n"
                f"{parent.get('qualified_name')}\n"
                f"file: {parent.get('file')}\n"
                f"lines: {parent.get('start_line')}-"
                f"{parent.get('end_line')}"
            )

        # -------------------------------------------------
        # Children
        # -------------------------------------------------

        children = graph.get("children") or []

        if children:
            lines = ["CHILDREN"]

            for child in children:
                lines.append(
                    f"- {child.get('qualified_name')} "
                    f"({child.get('type')}) "
                    f"{child.get('file')}:"
                    f"{child.get('start_line')}-"
                    f"{child.get('end_line')}"
                )

            chunks.append("\n".join(lines))

        # -------------------------------------------------
        # Callers
        # -------------------------------------------------

        callers = graph.get("callers") or []

        if callers:
            lines = ["CALLERS"]

            for caller in callers:
                lines.append(
                    f"- {caller.get('qualified_name')} "
                    f"({caller.get('file')}:"
                    f"{caller.get('start_line')})"
                )

            chunks.append("\n".join(lines))

        # -------------------------------------------------
        # Callees
        # -------------------------------------------------

        callees = graph.get("callees") or []

        if callees:
            lines = ["CALLEES"]

            for callee in callees:
                name = (
                    callee.get("qualified_name")
                    or callee.get("name")
                )

                location = ""

                if callee.get("file"):
                    location = (
                        f" ({callee.get('file')}:"
                        f"{callee.get('start_line')})"
                    )

                lines.append(
                    f"- {name}{location}"
                )

            chunks.append("\n".join(lines))

    return "\n\n".join(chunks)


# ---------------------------------------------------------
# Bonsai
# ---------------------------------------------------------

async def ask_bonsai(message, graph_context):
    """
    Send a RAG request to Bonsai.

    The graph context is included as additional information
    for the model.
    """

    context_text = build_graph_context_text(
        graph_context
    )

    system_prompt = (
        "You are an expert software engineering and "
        "reverse-engineering assistant.\n\n"
        "Use the supplied project context when answering. "
        "Distinguish facts found in the code from your "
        "inferences. Do not invent functions, files or "
        "relationships that are not present in the context.\n"
    )

    if context_text:
        system_prompt += (
            "\nPROJECT STRUCTURAL CONTEXT:\n"
            "--------------------------------\n"
            f"{context_text}\n"
            "--------------------------------\n"
        )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(
        timeout=300.0
    ) as client:

        response = await client.post(
            f"{BONSAI_URL}/v1/chat/completions",
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

    choice = data["choices"][0]

    message_data = choice.get(
        "message",
        {}
    )

    content = message_data.get(
        "content",
        ""
    )

    reasoning = message_data.get(
        "reasoning_content"
    )

    return {
        "content": content,
        "reasoning": reasoning,
        "raw": data,
    }
