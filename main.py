from contextlib import asynccontextmanager
from fastapi import FastAPI
from database import init_db
from routers import documents, ask, conversations, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Document Intelligence API",
    description=(
        "RAG-based document Q&A as a REST API. "
        "Upload documents, then ask questions across them with full conversation history."
    ),
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api",
)

app.include_router(documents.router)
app.include_router(ask.router)
app.include_router(conversations.router)
app.include_router(admin.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
