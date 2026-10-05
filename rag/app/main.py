from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .database import (
    init_database,
    create_project,
    get_project,
    list_projects,
    list_symbols,
    get_symbol,
    get_callees,
    get_callers,
)

from .indexer import index_project

from .rag import (
    search,
    expand_graph,
    ask_bonsai,
)

@asynccontextmanager
async def lifespan(app):

    init_database()

    yield


app = FastAPI(
    title="Bonsai RAG API",
    version="0.2.0",
    lifespan=lifespan,
)


class ProjectCreate(BaseModel):

    name: str

    path: str

    mode: str = "automatic"


class SearchRequest(BaseModel):

    query: str

    limit: int = 8


class ChatRequest(BaseModel):

    project: int

    mode: str = "automatic"

    message: str

    context_limit: int = 8


@app.get("/")
def root():

    return {
        "service": "bonsai-rag",
        "status": "ok",
        "version": "0.2.0",
    }


@app.get("/projects")
def projects():

    return [
        dict(row)
        for row in list_projects()
    ]


@app.post("/projects")
def create(data: ProjectCreate):

    project_id = create_project(
        data.name,
        data.path,
        data.mode,
    )

    return {
        "id": project_id,
        "name": data.name,
    }


@app.post("/projects/{project_id}/index")
def index(project_id: int):

    project = get_project(
        project_id
    )

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    result = index_project(
        project_id,
        project["path"],
    )

    return {
        "status": "indexed",
        "project_id": project_id,
        **result,
    }


@app.post("/projects/{project_id}/search")
def search_project(
    project_id: int,
    data: SearchRequest,
):

    project = get_project(
        project_id
    )

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    results = search(
        project_id,
        data.query,
        data.limit,
    )

    return {
        "results": results
    }


@app.get("/projects/{project_id}/symbols")
def project_symbols(
    project_id: int,
):

    project = get_project(
        project_id
    )

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    return {
        "symbols": [
            dict(row)
            for row in list_symbols(
                project_id
            )
        ]
    }


@app.get(
    "/projects/{project_id}/symbols/{symbol_name}"
)
def symbol_info(
    project_id: int,
    symbol_name: str,
):

    project = get_project(
        project_id
    )

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    symbol = get_symbol(
        project_id,
        symbol_name,
    )

    if not symbol:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    return {
        "symbol": dict(symbol),
        "callees": [
            dict(row)
            for row in get_callees(
                project_id,
                symbol_name,
            )
        ],
        "callers": [
            dict(row)
            for row in get_callers(
                project_id,
                symbol_name,
            )
        ],
    }


@app.get(
    "/projects/{project_id}/symbols/{symbol_name}/callees"
)
def symbol_callees(
    project_id: int,
    symbol_name: str,
):

    return {
        "symbol": symbol_name,
        "callees": [
            dict(row)
            for row in get_callees(
                project_id,
                symbol_name,
            )
        ],
    }


@app.get(
    "/projects/{project_id}/symbols/{symbol_name}/callers"
)
def symbol_callers(
    project_id: int,
    symbol_name: str,
):

    return {
        "symbol": symbol_name,
        "callers": [
            dict(row)
            for row in get_callers(
                project_id,
                symbol_name,
            )
        ],
    }

@app.get("/projects/{project_id}/symbols/{symbol_name}/graph")
def symbol_graph(project_id: int, symbol_name: str):
    project = get_project(project_id)

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    from .rag import get_graph_context

    graph = get_graph_context(
        project_id,
        symbol_name
    )

    if not graph:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found"
        )

    return graph

@app.post("/chat")
async def chat(data: ChatRequest):

    project = get_project(
        data.project
    )

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    # -----------------------------------------------------
    # 1. Semantic search
    # -----------------------------------------------------

    search_results = search(
        data.project,
        data.message,
        data.context_limit,
    )

    # -----------------------------------------------------
    # 2. Expand search results through code graph
    # -----------------------------------------------------

    graph_context = expand_graph(
        data.project,
        search_results,
    )

    # -----------------------------------------------------
    # 3. Ask LLM using graph-aware context
    # -----------------------------------------------------

    answer = await ask_bonsai(
        data.message,
        graph_context,
    )

    return {
        "answer": answer["content"],
        "reasoning": answer["reasoning"],
        "context": graph_context,
    }
