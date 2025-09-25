from fastapi import APIRouter

from app.routes.busqueda import buscar_documentos
from app.models.schemas import BusquedaRequest
from app.core.config import GEMINI_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from sentence_transformers.cross_encoder import CrossEncoder
from app.core.memory import SupabaseChatMessageHistory

router = APIRouter()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

memorias_de_sesion = {}

def obtener_historial_de_mensajes(session_id: str):
    return SupabaseChatMessageHistory(session_id)

reescritor_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("user", "Dada la conversación anterior, genera una consulta de búsqueda que sea autónoma y pueda ser entendida sin el historial. La consulta debe ser sobre el último tema discutido. No respondas la pregunta, solo reformúlala."),
    ("user", "{input}")
])

cadena_reescritura = reescritor_prompt | llm

prompt_principal = ChatPromptTemplate.from_messages([
    ("system", '''Eres un asistente de IA experto y servicial. Tu tarea es responder a la pregunta del usuario basándote en dos fuentes de información:

    1.  El contexto relevante extraído de un documento que se te proporciona.
    2.  El historial de la conversación que has tenido con el usuario.

    Responde de forma clara, concisa y amigable. Si la respuesta no se encuentra en el contexto o en el historial, indica amablemente que no tienes esa información.

    Contexto del documento:
    ---
    {contexto}
    ---'''),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

cadena_conversacional = prompt_principal | llm

cadena_con_historial = RunnableWithMessageHistory(
    cadena_conversacional,
    obtener_historial_de_mensajes,
    input_messages_key="input",
    history_messages_key="history",
)

@router.post('/')
async def responder_con_rag_y_memoria(payload: BusquedaRequest):
    historial = obtener_historial_de_mensajes(payload.session_id)

    consulta_reescrita_response = await cadena_reescritura.ainvoke({
        "history": historial.messages,
        "input": payload.consulta
    })
    consulta_reescrita = consulta_reescrita_response.content
    print(f"Consulta original: '{payload.consulta}' -> Consulta reescrita: '{consulta_reescrita}'")

    payload_busqueda = BusquedaRequest(
        consulta=consulta_reescrita,
        session_id=payload.session_id,
        top_k=10
    )
    contexto_chunks = buscar_documentos(payload_busqueda)
    print("--- Contenido de los Chunks Recuperados ---")
    for i, chunk in enumerate(contexto_chunks['resultados']):
        print(f"--- Chunk {i + 1} ---")
        print(chunk['texto'])
        print("\n")
    
    pares_para_rerank = []
    for chunk in contexto_chunks['resultados']:
        pares_para_rerank.append([consulta_reescrita, chunk['texto']])
    
    puntajes = cross_encoder.predict(pares_para_rerank)
    
    for i in range(len(contexto_chunks['resultados'])):
        contexto_chunks['resultados'][i]['relevance_score'] = puntajes[i]
        
    chunks_reordenados = sorted(contexto_chunks['resultados'], key=lambda x: x['relevance_score'], reverse=True)
    
    contexto_final_chunks = chunks_reordenados[:3]
    
    print("--- Documentos Re-ordenados ---")
    for chunk in contexto_final_chunks:
        print(f"--- Chunk ---")
        print(f"Puntaje: {chunk['relevance_score']:.4f}")
        print(chunk['texto'])
        print("\n")
    
    contexto_str = "".join(chunk['texto'] for chunk in contexto_final_chunks)

    config = {"configurable": {"session_id": payload.session_id}}
    
    respuesta_generada = await cadena_con_historial.ainvoke({
        "input": payload.consulta,
        "contexto": contexto_str,
    }, config=config)

    return {'respuesta': respuesta_generada.content}