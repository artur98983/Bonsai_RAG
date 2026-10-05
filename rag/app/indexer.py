import ast
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from .config import QDRANT_URL, COLLECTION_NAME
from .embeddings import embedding_model
from .database import (
    clear_project_code,
    insert_symbol,
    insert_reference,
)


qdrant = QdrantClient(url=QDRANT_URL)


SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
}


def get_language(path):
    return SUPPORTED_EXTENSIONS.get(Path(path).suffix.lower())


def get_symbol_name(node, text):
    name_node = node.child_by_field_name("name")

    if name_node is not None:
        return text[name_node.start_byte:name_node.end_byte]

    for child in node.children:
        if child.type in {
            "identifier",
            "type_identifier",
            "name",
        }:
            return text[child.start_byte:child.end_byte]

    return None


def get_node_code(text, node):
    return text[node.start_byte:node.end_byte]


def extract_code_symbols(text, language):
    """
    Extract functions/classes/methods and preserve their nesting.

    Result:
    [
        {
            "name": "Weapon",
            "qualified_name": "Weapon",
            "type": "class_definition",
            "parent_index": None,
            ...
        },
        {
            "name": "__init__",
            "qualified_name": "Weapon.__init__",
            "type": "function_definition",
            "parent_index": 0,
            ...
        }
    ]
    """

    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(language)
    except Exception:
        return []

    tree = parser.parse(text.encode("utf-8"))

    symbols = []

    symbol_types = {
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declaration",
        "class_definition",
        "class_declaration",
        "struct_definition",
        "struct_declaration",
        "enum_definition",
        "enum_declaration",
        "interface_declaration",
    }

    def walk(node, parent_index=None):
        current_parent = parent_index

        if node.type in symbol_types:
            name = get_symbol_name(node, text)

            if name:
                if parent_index is not None:
                    parent = symbols[parent_index]
                    qualified_name = f"{parent['qualified_name']}.{name}"
                else:
                    qualified_name = name

                symbol = {
                    "name": name,
                    "qualified_name": qualified_name,
                    "type": node.type,
                    "language": language,
                    "parent_index": parent_index,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "code": get_node_code(text, node),
                    "node": node,
                }

                symbols.append(symbol)

                current_parent = len(symbols) - 1

        for child in node.children:
            walk(child, current_parent)

    walk(tree.root_node)

    return symbols


def get_call_name(text, node):
    """
    Extract the callable name from a Tree-sitter call node.
    """

    function_node = node.child_by_field_name("function")

    if function_node is not None:
        return text[
            function_node.start_byte:function_node.end_byte
        ]

    for child in node.children:
        if child.type in {
            "identifier",
            "attribute",
            "field_expression",
        }:
            return text[
                child.start_byte:child.end_byte
            ]

    return None


def find_containing_symbol(node, symbols):
    """
    Find the smallest symbol containing this AST node.
    """

    candidates = []

    for index, symbol in enumerate(symbols):
        symbol_node = symbol["node"]

        if (
            symbol_node.start_byte <= node.start_byte
            and symbol_node.end_byte >= node.end_byte
        ):
            candidates.append(
                (
                    symbol_node.end_byte - symbol_node.start_byte,
                    index,
                )
            )

    if not candidates:
        return None

    candidates.sort()

    return candidates[0][1]


def extract_calls(text, language, symbols):
    """
    Extract function calls.

    Each result:
    {
        source_index,
        target_name,
        file_line,
    }
    """

    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(language)
    except Exception:
        return []

    tree = parser.parse(text.encode("utf-8"))

    call_types = {
        "call",
        "call_expression",
        "function_call",
    }

    references = []

    def walk(node):
        if node.type in call_types:
            target_name = get_call_name(text, node)

            if target_name:
                source_index = find_containing_symbol(
                    node,
                    symbols,
                )

                if source_index is not None:
                    references.append(
                        {
                            "source_index": source_index,
                            "target_name": target_name,
                            "line": node.start_point[0] + 1,
                        }
                    )

        for child in node.children:
            walk(child)

    walk(tree.root_node)

    return references


def ensure_collection():
    collections = qdrant.get_collections().collections

    if any(
        collection.name == COLLECTION_NAME
        for collection in collections
    ):
        return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


def delete_project_points(project_id):
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id),
                    )
                ]
            ),
        )
    except Exception:
        pass


def save_structural_analysis(
    project_id,
    structural_symbols,
    structural_references,
):
    """
    Save symbols first so that parent_symbol_id and target_symbol_id
    can reference actual database IDs.
    """

    clear_project_code(project_id)

    db_ids = {}

    # ---------------------------------------------------------
    # 1. Insert symbols
    # ---------------------------------------------------------

    for index, symbol in enumerate(structural_symbols):

        parent_db_id = None

        if symbol["parent_index"] is not None:
            parent_db_id = db_ids.get(
                symbol["parent_index"]
            )

        db_id = insert_symbol(
            project_id=project_id,
            file=symbol["file"],
            name=symbol["name"],
            qualified_name=symbol["qualified_name"],
            symbol_type=symbol["type"],
            language=symbol["language"],
            parent_symbol_id=parent_db_id,
            start_line=symbol["start_line"],
            end_line=symbol["end_line"],
            code=symbol["code"],
        )

        db_ids[index] = db_id

    # ---------------------------------------------------------
    # 2. Build symbol lookup
    # ---------------------------------------------------------

    by_qualified_name = {}
    by_name = {}

    for index, symbol in enumerate(structural_symbols):

        db_id = db_ids[index]

        by_qualified_name[
            symbol["qualified_name"]
        ] = db_id

        by_name.setdefault(
            symbol["name"],
            []
        ).append(db_id)

    # ---------------------------------------------------------
    # 3. Insert references
    # ---------------------------------------------------------

    for reference in structural_references:

        source_index = reference["source_index"]

        source_db_id = db_ids.get(source_index)

        if source_db_id is None:
            continue

        target_name = reference["target_name"]

        target_db_id = None

        # Exact qualified name.
        if target_name in by_qualified_name:
            target_db_id = by_qualified_name[target_name]

        # Bare name is only resolved if it is unambiguous.
        elif target_name in by_name:
            candidates = by_name[target_name]

            if len(candidates) == 1:
                target_db_id = candidates[0]

        insert_reference(
            project_id=project_id,
            source_symbol_id=source_db_id,
            target_symbol_id=target_db_id,
            target_name=target_name,
            reference_type="calls",
            file=reference["file"],
            line=reference["line"],
        )

    return db_ids


def index_project(project_id, project_path):
    ensure_collection()

    delete_project_points(project_id)

    all_documents = []
    structural_symbols = []
    structural_references = []

    files_count = 0

    project_root = Path(project_path)

    if not project_root.exists():
        raise FileNotFoundError(
            f"Project path does not exist: {project_path}"
        )

    # ---------------------------------------------------------
    # Walk project
    # ---------------------------------------------------------

    for root, dirs, files in os.walk(project_root):

        # Ignore common generated / VCS directories.
        dirs[:] = [
            d
            for d in dirs
            if d not in {
                ".git",
                "__pycache__",
                "node_modules",
                ".venv",
                "venv",
                "build",
                "dist",
            }
        ]

        for filename in files:

            path = Path(root) / filename

            language = get_language(path)

            if language is None:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            relative_file = str(
                path.relative_to(project_root)
            )

            files_count += 1

            # -------------------------------------------------
            # AST symbols
            # -------------------------------------------------

            symbols = extract_code_symbols(
                text,
                language,
            )

            # Store temporary indexes so references can
            # point to symbols belonging to this file.
            local_to_global = {}

            for local_index, symbol in enumerate(symbols):

                global_index = len(structural_symbols)

                symbol["file"] = relative_file

                structural_symbols.append(symbol)

                local_to_global[local_index] = global_index

            # -------------------------------------------------
            # Calls
            # -------------------------------------------------

            calls = extract_calls(
                text,
                language,
                symbols,
            )

            for call in calls:

                source_local = call["source_index"]

                source_global = local_to_global.get(
                    source_local
                )

                if source_global is None:
                    continue

                structural_references.append(
                    {
                        "source_index": source_global,
                        "target_name": call["target_name"],
                        "reference_type": "calls",
                        "file": relative_file,
                        "line": call["line"],
                    }
                )

            # -------------------------------------------------
            # Qdrant documents
            # -------------------------------------------------

            for symbol in symbols:

                all_documents.append(
                    {
                        "file": relative_file,
                        "symbol": symbol["qualified_name"],
                        "symbol_type": symbol["type"],
                        "language": language,
                        "start_line": symbol["start_line"],
                        "end_line": symbol["end_line"],
                        "text": symbol["code"],
                    }
                )

    # ---------------------------------------------------------
    # Save SQLite structure
    # ---------------------------------------------------------

    db_ids = save_structural_analysis(
        project_id,
        structural_symbols,
        structural_references,
    )

    # ---------------------------------------------------------
    # Save Qdrant vectors
    # ---------------------------------------------------------

    if all_documents:

        texts = [
            document["text"]
            for document in all_documents
        ]

        vectors = embedding_model.encode(texts)

        points = []

        for index, (document, vector) in enumerate(
            zip(all_documents, vectors)
        ):
            points.append(
                PointStruct(
                    id=index + 1,
                    vector=vector.tolist(),
                    payload={
                        "project_id": project_id,
                        **document,
                    },
                )
            )

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    return {
        "files": files_count,
        "documents": len(all_documents),
        "symbols": len(structural_symbols),
        "references": len(structural_references),
    }
