import os
import time
import uuid
import importlib
from typing import List, Optional
from fastapi import FastAPI, Header, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# LangGraph
from langgraph.graph import StateGraph, END

# LangSmith (tracking automático)
from langsmith import Client

# Pipelines
from pipelines import get_pipelines

# ======================
# Cargar variables de entorno (.env)
# ======================
load_dotenv()

API_KEY = os.getenv("API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OPENAI").upper()
MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",")]

# LangSmith
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
if LANGCHAIN_PROJECT:
    client = Client()
    print(f"[LangSmith] Tracking habilitado en el proyecto: {LANGCHAIN_PROJECT}")

# ======================
# Crear app FastAPI
# ======================
app = FastAPI(title="Backend LangChain + LangGraph + LangSmith para Open WebUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Schemas OpenAI-like
# ======================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None

# ======================
# Funciones helper
# ======================
def require_auth(authorization: Optional[str]):
    # Desarrollo local: no requiere API key
    return

def to_lc_messages(msgs: List[ChatMessage]):
    out = []
    for m in msgs:
        role = m.role.lower()
        if role == "system":
            out.append(SystemMessage(content=m.content))
        elif role == "user":
            out.append(HumanMessage(content=m.content))
        elif role == "assistant":
            out.append(AIMessage(content=m.content))
        else:
            out.append(HumanMessage(content=m.content))
    return out

def get_llm(model_override: Optional[str] = None, temperature: Optional[float] = None):
    model = model_override or MODEL_ID
    temp = TEMPERATURE if temperature is None else temperature
    if LLM_PROVIDER == "OPENAI":
        return ChatOpenAI(
            model=model,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=temp,
        )
    else:
        raise Exception(f"LLM_PROVIDER no soportado: {LLM_PROVIDER}")

def build_graph(llm):
    graph = StateGraph(dict)

    def step(state):
        result = llm.invoke(state["messages"])
        return {"response": result.content}

    graph.add_node("chat", step)
    graph.set_entry_point("chat")
    graph.set_finish_point("chat")
    return graph.compile()

# ======================
# Endpoints
# ======================
@app.get("/health")
def health():
    return {"status": "ok", "provider": LLM_PROVIDER, "model": MODEL_ID}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time())}]
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    llm = get_llm(model_override=req.model, temperature=req.temperature)
    lc_messages = to_lc_messages(req.messages)

    graph = build_graph(llm)
    output = graph.invoke({"messages": lc_messages})
    full_text = output["response"]

    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
    }
    return JSONResponse(content=resp)

# ======================
# Endpoints pipelines
# ======================
@app.get("/v1/pipelines")
def list_pipelines():
    pipelines_dict = get_pipelines()
    return {
        "pipelines": [
            {
                "name": name,
                "description": f"Pipeline {name} para Open WebUI",
                "endpoint": f"http://127.0.0.1:8000/v1/run_pipeline?name={name}",
                "method": "POST"  # <-- Agregado para Open Web UI
            }
            for name in pipelines_dict.keys()
        ]
    }

@app.post("/v1/run_pipeline")
def run_pipeline(name: str = Query(...), data: Optional[dict] = None):
    pipelines_dict = get_pipelines()
    if name not in pipelines_dict:
        return {"error": f"Pipeline {name} no encontrado"}

    pipeline_func = pipelines_dict[name]
    pipeline_func(data or {})
    return {"status": "ok", "pipeline": name, "data": data or {}}

# ======================
# Cargar automáticamente pipelines
# ======================
PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "pipelines")
loaded_pipelines = []

for file in os.listdir(PIPELINES_DIR):
    if file.endswith(".py") and file != "__init__.py":
        module_name = f"pipelines.{file[:-3]}"
        importlib.import_module(module_name)
        loaded_pipelines.append(file[:-3])
        print(f"[Main] Pipeline {file[:-3]} cargado")

print(f"[Main] Pipelines cargados: {len(loaded_pipelines)}")
