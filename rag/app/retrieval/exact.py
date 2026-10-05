import re

from ..database import get_symbol


def extract_identifier_tokens(query: str) -> list[str]:
    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_.]*",
        query,
    )


def find_explicit_symbols(
    project_id: int,
    query: str,
) -> list:
    """
    Find symbols explicitly mentioned in the user's query.

    Exact symbol resolution is intentionally separate from
    semantic retrieval.
    """

    tokens = extract_identifier_tokens(query)

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
