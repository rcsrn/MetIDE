from langchain_core.tools import tool
import requests

from app.routes.busqueda import buscar_documentos
from sentence_transformers.cross_encoder import CrossEncoder
from datetime import datetime
from app.models.schemas import BusquedaRequest

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@tool
def buscar_contexto_en_documentos(consulta: str) -> str:
    """Útil para buscar información en documentos. Devuelve el contexto relevante para responder una pregunta."""
    print(f"--- Herramienta RAG buscando contexto para: {consulta} ---")
    
    payload_busqueda = BusquedaRequest(consulta=consulta, top_k=10)
    contexto_chunks = buscar_documentos(payload_busqueda)

    if not contexto_chunks.get('resultados'):
        return "No se encontró contexto relevante en los documentos."
    
    pares_para_rerank = [[consulta, chunk['texto']] for chunk in contexto_chunks['resultados']]
    puntajes = cross_encoder.predict(pares_para_rerank)
    
    for i, chunk in enumerate(contexto_chunks['resultados']):
        chunk['relevance_score'] = puntajes[i]
        
    chunks_reordenados = sorted(contexto_chunks['resultados'], key=lambda x: x['relevance_score'], reverse=True)
    
    contexto_final_chunks = chunks_reordenados[:3]
    
    contexto_str = "".join(chunk['texto'] for chunk in contexto_final_chunks)
    return contexto_str
