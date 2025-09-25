# app/routes/documentos.py
import json
import uuid

from fastapi import APIRouter, HTTPException

from app.core.embedding import generar_embedding
from app.core.supabase_client import supabase
from app.models.schemas import DocumentoRequest

router = APIRouter()

@router.post("/")
def insertar_documento(payload: DocumentoRequest):
    try:
        vector = generar_embedding(payload.texto)
        doc_id = str(uuid.uuid4())

        data = {
            "id": doc_id,
            "texto": payload.texto,
            "metadatos": json.dumps(payload.metadatos),
            "embedding": vector
        }

        supabase.table("documentos").insert(data).execute()
        return {"id": doc_id, "mensaje": "Documento insertado correctamente"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))