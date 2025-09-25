from fastapi import APIRouter, HTTPException

from app.core.embedding import generar_embedding
from app.core.supabase_client import supabase
from app.models.schemas import BusquedaRequest

router = APIRouter()

@router.post("/")
def buscar_documentos(payload: BusquedaRequest):
    try:
        vector = generar_embedding(payload.consulta)

        resultado = supabase.rpc(
            "buscar_similares",
            {"query": vector, "top_k": payload.top_k}
        ).execute()

        return {"resultados": resultado.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))