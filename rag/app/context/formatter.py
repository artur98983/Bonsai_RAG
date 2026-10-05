def build_graph_context_text(graph_context: list[dict]) -> str:
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

        parent = graph.get("parent")

        if parent:
            chunks.append(
                "PARENT\n"
                f"{parent.get('qualified_name')}\n"
                f"file: {parent.get('file')}\n"
                f"lines: {parent.get('start_line')}-"
                f"{parent.get('end_line')}"
            )

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
