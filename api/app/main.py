from app.routes import vision
from app.routes import function_calling
from fastapi import FastAPI

from app.routes import busqueda, documentos, rag, rag_memory, rag_re_ranker

app = FastAPI()

app.include_router(function_calling.router, prefix="/function_calling")
app.include_router(rag.router, prefix="/rag")
app.include_router(documentos.router, prefix="/documentos")
app.include_router(busqueda.router, prefix="/buscar")
