from ..database import (
    get_symbol,
    get_symbol_by_id,
    get_callees,
    get_callers,
    get_children,
)


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


def caller_to_dict(row):
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


def get_graph_context(
    project_id: int,
    symbol_name: str,
) -> dict | None:

    symbol = get_symbol(
        project_id,
        symbol_name,
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

    parent_symbol_id = symbol["parent_symbol_id"]

    if parent_symbol_id is not None:
        parent = get_symbol_by_id(
            project_id,
            parent_symbol_id,
        )

        if parent is not None:
            result["parent"] = symbol_to_dict(parent)

    children = get_children(
        project_id,
        symbol["id"],
    )

    result["children"] = [
        symbol_to_dict(row)
        for row in children
    ]

    callers = get_callers(
        project_id,
        symbol["qualified_name"],
    )

    result["callers"] = [
        caller_to_dict(row)
        for row in callers
    ]

    callees = get_callees(
        project_id,
        symbol["qualified_name"],
    )

    result["callees"] = [
        callee_to_dict(row)
        for row in callees
    ]

    return result
