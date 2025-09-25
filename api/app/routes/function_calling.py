from fastapi import APIRouter
from app.models.schemas import BusquedaRequest
from app.tools.agent_tools import buscar_contexto_en_documentos, buscar_info_cliente, registrar_cliente, editar_cliente, eliminar_cliente
from app.core.memory import SupabaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.core.config import GEMINI_API_KEY

router = APIRouter()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
tools = [buscar_contexto_en_documentos, buscar_info_cliente, registrar_cliente, editar_cliente, eliminar_cliente]

def obtener_historial_de_mensajes(session_id: str):
    return SupabaseChatMessageHistory(session_id)

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de IA. Tu única fuente de conocimiento es la información que te proporcionan tus herramientas. Tu proceso es el siguiente:
        1.  Analiza la pregunta del usuario y el historial para reescribirla y que sea una consulta autónoma.
        2.  Decide qué herramienta usar para obtener la información.
        3.  Invoca la herramienta.
        4.  Responde de forma clara, concisa, corta y amigable. Si la respuesta no se encuentra en el contexto o en el historial, indica amablemente que no tienes esa información.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_con_memoria = RunnableWithMessageHistory(
    agent_executor,
    obtener_historial_de_mensajes,
    input_messages_key="input",
    history_messages_key="history",
)

@router.post('/')
async def master_agent_endpoint(payload: BusquedaRequest):
    response = await agent_con_memoria.ainvoke(
        {"input": payload.consulta},
        config={"configurable": {"session_id": payload.session_id}}
    )
    return {"respuesta": response.get("output")}

