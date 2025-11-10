import os
import time
import uuid
import json
import asyncio
import importlib
import re
from typing import List, Optional, Dict, Any, Generator, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# LangGraph
from langgraph.graph import StateGraph

# OpenAI (nuevo estilo >=1.0.0)
from openai import OpenAI

# LangSmith (opcional)
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# ======================
# Cargar variables de entorno
# ======================
load_dotenv()

API_KEY = os.getenv("API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OPENAI").upper()
MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
TOMORROW_API_KEY = os.getenv("TOMORROW_API_KEY")

# Twilio / WhatsApp configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")
TWILIO_VERIFY_TOKEN = os.getenv("TWILIO_VERIFY_TOKEN", "")
TWILIO_BASE_URL = os.getenv("TWILIO_BASE_URL", "https://api.twilio.com/2010-04-01")
_twilio_allowed_raw = os.getenv("TWILIO_WHATSAPP_ALLOWED_RECIPIENTS", "")
TWILIO_ALLOWED_RECIPIENTS = {
    number.strip(): True for number in _twilio_allowed_raw.split(",") if number.strip()
}

WHATSAPP_HISTORY_MAX_MESSAGES = int(os.getenv("WHATSAPP_HISTORY_MAX_MESSAGES", "12"))
whatsapp_conversations: Dict[str, List[Dict[str, str]]] = {}

# Cliente OpenAI
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
else:
    client = None

# CORS configurado para Open WebUI
ALLOWED_ORIGINS = ["*"]

LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
if LANGCHAIN_PROJECT and LANGSMITH_AVAILABLE:
    client_ls = Client()
    print(f"[LangSmith] Tracking habilitado en el proyecto: {LANGCHAIN_PROJECT}")

# ======================
# Crear app FastAPI
# ======================
app = FastAPI(
    title="Backend LangChain para Open WebUI",
    description="Backend con pipelines LangChain compatible con Open WebUI - Versión Optimizada",
    version="4.1.0"
)

# Configuración CORS ultra permisiva
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ======================
# Schemas flexibles
# ======================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

# ======================
# Funciones helper mejoradas
# ======================
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

def create_chat_completion_response(content: str, model: str, request_id: str = None) -> Dict[str, Any]:
    """Función helper para crear respuestas en formato OpenAI compatible"""
    return {
        "id": request_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(content.split()) * 2,
            "completion_tokens": len(content.split()),
            "total_tokens": len(content.split()) * 3
        }
    }

def create_streaming_chunk(content: str, model: str, request_id: str, is_final: bool = False) -> str:
    """Crea un chunk de streaming en formato OpenAI"""
    if is_final:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
    else:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        }
    return f"data: {json.dumps(chunk)}\n\n"

async def stream_content_async(content: str, model: str, request_id: str) -> Generator[str, None, None]:
    """Genera streaming de contenido de manera asíncrona"""
    yield create_streaming_chunk("", model, request_id)
    
    words = content.split()
    chunk_size = 3
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " ".join(chunk_words)
        if i + chunk_size < len(words):
            chunk_content += " "
        
        yield create_streaming_chunk(chunk_content, model, request_id)
        await asyncio.sleep(0.05)
    
    yield create_streaming_chunk("", model, request_id, is_final=True)
    yield "data: [DONE]\n\n"

def stream_content_sync(content: str, model: str, request_id: str) -> Generator[str, None, None]:
    """Genera streaming de contenido de manera síncrona"""
    yield create_streaming_chunk("", model, request_id)
    
    words = content.split()
    chunk_size = 3
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " ".join(chunk_words)
        if i + chunk_size < len(words):
            chunk_content += " "
        
        yield create_streaming_chunk(chunk_content, model, request_id)
        time.sleep(0.02)
    
    yield create_streaming_chunk("", model, request_id, is_final=True)
    yield "data: [DONE]\n\n"

def sanitize_for_whatsapp(text: str) -> str:
    """Convierte Markdown básico a texto plano legible en WhatsApp"""
    if not text:
        return ""

    sanitized = re.sub(r"\|.*\|", "", text)
    sanitized = re.sub(r"^-{3,}$", "", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"^#+\s*", "", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"^[\s\-*]+", "- ", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)

    return sanitized.strip()

def split_whatsapp_message(text: str, limit: int = 1500) -> List[str]:
    """Divide el texto en segmentos seguros para Twilio (máx 1600 caracteres)."""
    content = (text or "").strip()
    if not content:
        return [""]

    segments: List[str] = []
    remaining = content

    while remaining:
        if len(remaining) <= limit:
            segments.append(remaining)
            break

        split_idx = remaining.rfind("\n", 0, limit)
        if split_idx == -1 or split_idx < int(limit * 0.6):
            split_idx = remaining.rfind(" ", 0, limit)
        if split_idx == -1 or split_idx < int(limit * 0.6):
            split_idx = limit

        segment = remaining[:split_idx].strip()
        if segment:
            segments.append(segment)

        remaining = remaining[split_idx:].strip()

        if not remaining:
            break

    return segments if segments else [content]

def extract_pipeline_output(result: Any) -> str:
    """✅ MEJORADO: Extrae el contenido de la respuesta del pipeline"""
    if isinstance(result, dict):
        # ✅ Formato OpenAI estándar (PRIORIDAD)
        if "choices" in result and isinstance(result["choices"], list):
            try:
                return result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass
        
        # Formato legacy "reply"
        if "reply" in result:
            return str(result["reply"])
        
        # Otros campos comunes
        for key in ["output", "response", "result", "content", "message", "text"]:
            if key in result:
                return str(result[key])
        
        return str(result)
    else:
        return str(result)

def extract_user_message_from_request(request: Union[Dict, Any]) -> str:
    """Extrae mensaje del usuario de cualquier formato de request"""
    if isinstance(request, dict):
        if "body" in request:
            body = request["body"]
            if isinstance(body, dict) and "messages" in body:
                messages = body["messages"]
                if messages and isinstance(messages, list):
                    last_message = messages[-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        return last_message["content"]
        
        if "messages" in request:
            messages = request["messages"]
            if messages and isinstance(messages, list):
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"]
        
        if "input" in request:
            return str(request["input"])
    
    return ""

def is_streaming_request(request: Union[Dict, Any]) -> bool:
    """Detecta si el request solicita streaming"""
    if isinstance(request, dict):
        if request.get("stream", False):
            return True
        
        if "body" in request and isinstance(request["body"], dict):
            if request["body"].get("stream", False):
                return True
    
    return False

SUPPORTED_PIPELINES = {
    "wheat": "wheat_expert_complete",
    "trigo": "wheat_expert_complete",
    "rice": "rice_expert_complete",
    "arroz": "rice_expert_complete",
    "lentil": "lentil_expert_complete",
    "lenteja": "lentil_expert_complete",
    "lentejas": "lentil_expert_complete",
    "porotos": "poroto_expert_complete",
}

DEFAULT_WHATSAPP_REPLY = os.getenv(
    "WHATSAPP_DEFAULT_REPLY",
    "Por ahora puedo ayudarte con consultas de trigo, arroz, porotos o lentejas. Indica el cultivo para continuar."
)

def detect_pipeline_from_text(message: str) -> Optional[str]:
    """Selecciona pipeline válido solo para trigo, arroz, porotos o lentejas."""
    text = (message or "").lower()

    for keyword, pipeline_id in SUPPORTED_PIPELINES.items():
        if keyword in text:
            if pipeline_id in pipelines_dict:
                return pipeline_id

    return None

def run_pipeline_request(model_id: str, messages: List[Dict[str, str]]) -> str:
    if model_id not in pipelines_dict:
        raise ValueError(f"Pipeline {model_id} no cargado")

    pipeline_func = pipelines_dict[model_id]["func"]

    prompt_request = {
        "messages": messages,
        "model": model_id,
        "temperature": TEMPERATURE,
        "max_tokens": 4000,
        "stream": False
    }

    result = pipeline_func({
        "messages": prompt_request["messages"],
        "temperature": prompt_request["temperature"],
        "max_tokens": prompt_request["max_tokens"],
        "model": model_id,
        "stream": False,
        "body": prompt_request
    })

    return extract_pipeline_output(result)
    prompt_request = {
        "messages": messages,
        "model": model_id,
        "temperature": TEMPERATURE,
        "max_tokens": 4000,
        "stream": False
    }

    result = pipeline_func({
        "messages": prompt_request["messages"],
        "temperature": prompt_request["temperature"],
        "max_tokens": prompt_request["max_tokens"],
        "model": model_id,
        "stream": False,
        "body": prompt_request
    })

    return extract_pipeline_output(result)

def send_whatsapp_message(to_number: str, content: str) -> Dict[str, Any]:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM):
        raise RuntimeError("Credenciales de Twilio incompletas")

    url = f"{TWILIO_BASE_URL}/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    segments = split_whatsapp_message(content)
    responses: List[Dict[str, Any]] = []

    for idx, segment in enumerate(segments, start=1):
        body_text = segment
        if len(segments) > 1:
            body_text = f"[{idx}/{len(segments)}]\n{segment}"

        payload = {
            "From": TWILIO_WHATSAPP_FROM,
            "To": to_number,
            "Body": body_text
        }

        response = requests.post(
            url,
            data=payload,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=15
        )
        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text}

        responses.append({
            "segment": idx,
            "status_code": response.status_code,
            "data": data
        })

        if response.status_code >= 400:
            break

    return {
        "segments_sent": len(responses),
        "responses": responses
    }

# ======================
# Diccionario global de pipelines
# ======================
pipelines_dict = {}

# ======================
# Cargar pipelines automáticamente
# ======================
def load_pipelines():
    global pipelines_dict
    pipelines_dict.clear()
    
    PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "pipelines")
    
    if not os.path.exists(PIPELINES_DIR):
        print(f"[Warning] Directorio {PIPELINES_DIR} no existe")
        return
    
    # Directorios a excluir del cargado automático
    EXCLUDED_DIRS = {"failed", "pipelines", "__pycache__", "documents", "example", "test_response_format", "main", "master_pipeline", "example_pipeline"}
    
    for root, dirs, files in os.walk(PIPELINES_DIR):
        # Filtrar directorios excluidos para evitar que os.walk() entre en ellos
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Solo cargar archivos del directorio raíz de pipelines
                if root == PIPELINES_DIR:
                    rel_path = os.path.relpath(os.path.join(root, file), os.path.dirname(__file__))
                    module_name = rel_path.replace(os.sep, ".")[:-3]
                    
                    try:
                        module = importlib.import_module(module_name)
                        
                        if hasattr(module, "pipeline") and hasattr(module, "metadata"):
                            key_name = module.metadata.get("name", file[:-3])
                            pipelines_dict[key_name] = {
                                "func": module.pipeline,
                                "metadata": module.metadata
                            }
                            print(f"✅ Pipeline {key_name} cargado: {module.metadata.get('description', 'Sin descripción')}")
                        else:
                            print(f"⚠️  {file} no tiene 'pipeline' o 'metadata'")
                            
                    except Exception as e:
                        print(f"❌ Error cargando {file}: {e}")
                else:
                    print(f"🔄 Saltando archivo en subdirectorio: {os.path.join(root, file)}")

load_pipelines()
print(f"🔄 Total pipelines cargados: {len(pipelines_dict)}")
print(f"📋 Pipelines disponibles: {list(pipelines_dict.keys())}")
# ======================
# Middleware de logging mejorado
# ======================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    print(f"[Request] {request.method} {request.url.path}")
    if request.headers.get("content-type") == "application/json":
        print(f"[Request] Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    print(f"[Response] {response.status_code} - {process_time:.3f}s")
    
    return response

# ======================
# Endpoints básicos
# ======================
@app.get("/")
def root():
    return {
        "message": "🚀 Backend LangChain para Open WebUI v4.1 - Optimizado para Visualización",
        "status": "running",
        "pipelines_count": len(pipelines_dict),
        "pipelines": list(pipelines_dict.keys()),
        "version": "4.1.0",
        "weather_api": "Tomorrow.io" if TOMORROW_API_KEY else "No configurada",
        "features": [
            "streaming_async", "streaming_sync", "chat_completions", 
            "pipelines", "weather", "cors_permissive", "logging_enhanced",
            "openai_format_responses"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pipelines_loaded": len(pipelines_dict),
        "model": MODEL_ID,
        "weather_api": bool(TOMORROW_API_KEY),
        "openai_api": bool(OPENAI_API_KEY),
        "streaming_support": True,
        "cors_enabled": True,
        "timestamp": int(time.time())
    }

# ======================
# Endpoint de clima directo
# ======================
@app.get("/v1/clima/{ciudad}")
def obtener_clima(ciudad: str):
    """Endpoint directo para obtener clima"""
    if not TOMORROW_API_KEY:
        raise HTTPException(status_code=500, detail="API Key de Tomorrow.io no configurada")
    
    try:
        url = "https://api.tomorrow.io/v4/weather/realtime"
        
        coordinates_map = {
            "santiago": "-33.4489,-70.6693",
            "madrid": "40.4168,-3.7038",
            "barcelona": "41.3851,2.1734",
            "buenos aires": "-34.6118,-58.3960"
        }
        
        coords = coordinates_map.get(ciudad.lower(), ciudad)
        
        querystring = {
            "location": coords,
            "apikey": TOMORROW_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(url, params=querystring, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            weather_data = data['data']
            values = weather_data.get('values', {})
            
            return {
                "ciudad": ciudad,
                "temperatura": values.get('temperature'),
                "humedad": values.get('humidity'),
                "precipitacion": values.get('precipitationType', 0),
                "viento": values.get('windSpeed'),
                "timestamp": int(time.time()),
                "success": True
            }
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Error de API Tomorrow.io: {response.text}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ======================
# Endpoints OpenAI Compatible
# ======================
@app.get("/v1/models")
def list_models():
    models = []
    
    if client:
        try:
            response = client.models.list()
            for m in response.data:
                models.append({
                    "id": m.id,
                    "object": m.object,
                    "created": getattr(m, "created", int(time.time())),
                    "owned_by": getattr(m, "owned_by", "openai")
                })
        except Exception as e:
            print(f"[Warning] No se pudieron obtener modelos de OpenAI: {e}")
            models.append({
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            })
    else:
        models.append({
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai"
        })
    
    for name, data in pipelines_dict.items():
        models.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "langchain-backend",
            "pipeline": {
                "type": data["metadata"].get("type", "llm"),
                "name": data["metadata"].get("name", name),
                "description": data["metadata"].get("description", "")
            }
        })
    
    return {"object": "list", "data": models}

@app.get("/models")
def list_models_alt():
    return list_models()

# ======================
# ENDPOINTS ADICIONALES PARA RESOLVER ERRORES 405
# ======================
@app.get("/v1/models/models")
def models_models_endpoint():
    """Endpoint duplicado que Open WebUI busca"""
    print("[Models/Models] GET endpoint duplicado llamado")
    return list_models()

@app.post("/v1/models/models")
def models_models_post():
    """Endpoint POST para /v1/models/models"""
    print("[Models/Models] POST endpoint llamado")
    return list_models()

# ======================
# ENDPOINT ESPECÍFICO PARA CHAT COMPLETIONS MALFORMADO
# ======================
@app.post("/v1/models/chat/completions")
async def models_chat_completions(request: Request):
    """Endpoint específico para /v1/models/chat/completions"""
    print(f"[Models Chat Completions] Request interceptado - redirigiendo")
    return await chat_completions(request)

# ======================
# ENDPOINTS ESPECÍFICOS PARA PIPELINES MALFORMADOS
# ======================
@app.post("/v1/models/{pipeline_id}/filter/inlet")
async def models_pipeline_inlet(pipeline_id: str, request: Request):
    """Endpoint específico para /v1/models/{pipeline_id}/filter/inlet"""
    print(f"[Models Pipeline Inlet] Request interceptado para {pipeline_id}")
    return await pipeline_filter_inlet(pipeline_id, request)

@app.post("/v1/models/{pipeline_id}/filter/outlet")
async def models_pipeline_outlet(pipeline_id: str, request: Request):
    """Endpoint específico para /v1/models/{pipeline_id}/filter/outlet"""
    print(f"[Models Pipeline Outlet] Request interceptado para {pipeline_id}")
    return await pipeline_filter_outlet(pipeline_id, request)

# ======================
# CATCH-ALL MODELS
# ======================
@app.api_route("/v1/models/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def catch_models_paths(path: str, request: Request):
    """Catch-all para modelos"""
    print(f"[Models Catch-All] {request.method} /v1/models/{path}")
    
    if 'filter' in path:
        print(f"[Models Catch-All] IGNORANDO pipeline request: {path}")
        raise HTTPException(status_code=404, detail="Endpoint not handled by models catch-all")
    
    if path == 'chat/completions':
        print(f"[Models Catch-All] ERROR: chat/completions llegó al catch-all")
        return JSONResponse({
            "error": "chat/completions should be handled by specific endpoint",
            "redirect_to": "/v1/chat/completions"
        }, status_code=500)
    
    return list_models()

# ======================
# Endpoints Chat - MAXIMA COMPATIBILIDAD
# ======================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Endpoint de chat con máxima compatibilidad"""
    
    try:
        request_data = await request.json()
        
        print(f"[Chat Completions] Request data: {request_data}")
        
        model = request_data.get("model", MODEL_ID)
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", TEMPERATURE)
        max_tokens = request_data.get("max_tokens", 4000)
        stream = request_data.get("stream", False)
        
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        print(f"[Chat Completions] Model: {model}, Stream: {stream}")
        print(f"[Chat Completions] Messages count: {len(messages)}")
        
        # Si el modelo es un pipeline
        if model in pipelines_dict:
            try:
                pipeline_func = pipelines_dict[model]["func"]
                
                pipeline_input = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model": model,
                    "stream": stream,
                    "body": {
                        "messages": messages,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": stream
                    }
                }
                
                if "choices" in request_data:
                    pipeline_input["choices"] = request_data["choices"]
                
                result = pipeline_func(pipeline_input)
                content = extract_pipeline_output(result)
                
                print(f"[Chat Completions] Pipeline result: {content[:100]}...")
                
                if stream:
                    return StreamingResponse(
                        stream_content_sync(content, model, request_id),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Content-Type": "text/event-stream",
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Headers": "*",
                            "X-Accel-Buffering": "no"
                        }
                    )
                else:
                    return create_chat_completion_response(content, model, request_id)
                
            except Exception as e:
                print(f"[Error] Pipeline {model}: {e}")
                error_content = f"Error en pipeline {model}: {str(e)}"
                
                if stream:
                    return StreamingResponse(
                        stream_content_sync(error_content, model, request_id),
                        media_type="text/event-stream"
                    )
                else:
                    return create_chat_completion_response(error_content, model, request_id)
        
        # LLM estándar
        try:
            llm = get_llm(model, temperature)
            lc_messages = []
            
            for msg in messages:
                role = msg.get("role", "user").lower()
                content = msg.get("content", "")
                
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))
            
            graph = build_graph(llm)
            output = graph.invoke({"messages": lc_messages})
            content = output["response"]
            
            if stream:
                return StreamingResponse(
                    stream_content_sync(content, model, request_id),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            else:
                return create_chat_completion_response(content, model, request_id)
            
        except Exception as e:
            print(f"[Error] LLM estándar: {e}")
            error_content = f"Error en chat: {str(e)}"
            
            if stream:
                return StreamingResponse(
                    stream_content_sync(error_content, model, request_id),
                    media_type="text/event-stream"
                )
            else:
                return create_chat_completion_response(error_content, model, request_id)
        
    except Exception as e:
        print(f"[Error] Chat completions general: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error procesando request: {str(e)}"}
        )

# ======================
# ENDPOINT DEFINITIVO - Filter Inlet
# ======================
@app.post("/v1/{pipeline_id}/filter/inlet")
async def pipeline_filter_inlet(pipeline_id: str, request: Request):
    """✅ OPTIMIZADO: Endpoint para Open WebUI con formato OpenAI"""
    if pipeline_id not in pipelines_dict:
        print(f"[Filter Inlet] Pipeline {pipeline_id} no encontrado")
        return JSONResponse({"bypass": True, "message": "Pipeline not found"})
    
    try:
        request_data = await request.json()
        
        print(f"[Filter Inlet] Pipeline: {pipeline_id}")
        print(f"[Filter Inlet] Request data keys: {list(request_data.keys())}")
        
        pipeline_func = pipelines_dict[pipeline_id]["func"]
        result = pipeline_func(request_data)
        
        # ✅ NUEVO: Verificar si ya viene en formato OpenAI
        if isinstance(result, dict) and "choices" in result:
            # Ya está en formato correcto, usar directamente
            print(f"[Filter Inlet] Pipeline devolvió formato OpenAI nativo")
            content = extract_pipeline_output(result)
            response_data = result
        else:
            # Convertir a formato OpenAI
            content = extract_pipeline_output(result)
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            response_data = create_chat_completion_response(content, pipeline_id, request_id)
        
        is_streaming = is_streaming_request(request_data)
        
        print(f"[Filter Inlet] Streaming: {is_streaming}")
        print(f"[Filter Inlet] Content length: {len(content)}")
        print(f"[Filter Inlet] Content preview: {content[:100]}...")
        
        if is_streaming:
            print(f"[Filter Inlet] Iniciando streaming response")
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            
            return StreamingResponse(
                stream_content_sync(content, pipeline_id, request_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            print(f"[Filter Inlet] Respuesta normal enviada en formato OpenAI")
            
            return JSONResponse(
                content=response_data,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        
    except Exception as e:
        print(f"[Error] Filter inlet {pipeline_id}: {e}")
        print(f"[Error] Exception type: {type(e)}")
        
        error_content = f"Error ejecutando {pipeline_id}: {str(e)}"
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        return JSONResponse(
            content=create_chat_completion_response(error_content, pipeline_id, request_id),
            status_code=500
        )

@app.post("/v1/{pipeline_id}/filter/outlet")
async def pipeline_filter_outlet(pipeline_id: str, request: Request):
    """Endpoint outlet para Open WebUI"""
    try:
        request_data = await request.json()
        print(f"[Filter Outlet] Pipeline: {pipeline_id}")
        print(f"[Filter Outlet] Request data: {request_data}")
        
        return JSONResponse({
            "bypass": True,
            "message": "Filter outlet processed"
        })
        
    except Exception as e:
        print(f"[Error] Filter outlet {pipeline_id}: {e}")
        return JSONResponse({
            "bypass": True,
            "error": str(e)
        })

@app.api_route("/v1/{pipeline_id}/filter/{filter_type:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def catch_filter_paths(pipeline_id: str, filter_type: str, request: Request):
    """Catch-all para endpoints de filter"""
    print(f"[Filter Catch-All] {request.method} /v1/{pipeline_id}/filter/{filter_type}")
    
    try:
        if request.method == "POST":
            request_data = await request.json()
        else:
            request_data = {}
        
        return JSONResponse({
            "bypass": True,
            "message": f"Filter {filter_type} processed for {pipeline_id}",
            "method": request.method
        })
        
    except Exception as e:
        return JSONResponse({
            "bypass": True,
            "error": str(e)
        })

# ======================
# Endpoints adicionales
# ======================
@app.get("/v1/pipelines")
def list_pipelines():
    pipeline_list = []
    
    for name, data in pipelines_dict.items():
        metadata = data["metadata"]
        pipeline_list.append({
            "id": name,
            "name": metadata.get("name", name),
            "description": metadata.get("description", ""),
            "type": metadata.get("type", "llm"),
            "object": "pipeline",
            "created": int(time.time()),
            "status": "active"
        })
    
    return {"object": "list", "data": pipeline_list}

@app.get("/pipelines")
def list_pipelines_alt():
    return list_pipelines()

@app.post("/v1/pipelines/reload")
def reload_pipelines():
    load_pipelines()
    return {
        "message": "Pipelines recargados",
        "count": len(pipelines_dict),
        "pipelines": list(pipelines_dict.keys()),
        "timestamp": int(time.time())
    }

# ======================
# Options para CORS
# ======================
@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )

# ======================
# Webhook WhatsApp (Twilio)
# ======================
@app.get("/webhook")
async def verify_webhook(request: Request):
    """Verificación del webhook (compatibilidad Meta/Twilio)."""
    mode = request.query_params.get("hub.mode") or request.query_params.get("mode")
    token = request.query_params.get("hub.verify_token") or request.query_params.get("token")
    challenge = request.query_params.get("hub.challenge") or request.query_params.get("challenge")

    if mode == "subscribe" and token == TWILIO_VERIFY_TOKEN:
        if challenge:
            try:
                return int(challenge)
            except ValueError:
                return challenge
        return "OK"

    return JSONResponse(status_code=403, content={"error": "Verification token mismatch"})

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """Procesa mensajes entrantes de Twilio WhatsApp."""
    try:
        form = await request.form()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Invalid payload: {exc}"})

    payload = dict(form)
    print(f"[WhatsApp] Payload recibido: {payload}")

    from_number = payload.get("From")
    message_body = payload.get("Body", "").strip()

    if not from_number or not message_body:
        return JSONResponse(status_code=200, content={"status": "ignored"})

    if TWILIO_ALLOWED_RECIPIENTS and from_number not in TWILIO_ALLOWED_RECIPIENTS:
        print(f"[WhatsApp] Número {from_number} no autorizado")
        return JSONResponse(status_code=200, content={"status": "unauthorized"})

    pipeline_id = detect_pipeline_from_text(message_body)

    if not pipeline_id:
        print(f"[WhatsApp] Mensaje sin pipeline válido: {message_body}")
        try:
            send_result = send_whatsapp_message(from_number, DEFAULT_WHATSAPP_REPLY)
            print(f"[WhatsApp] Respuesta default enviada: {send_result}")
        except Exception as exc:
            print(f"[WhatsApp] Error enviando respuesta default: {exc}")
            return JSONResponse(status_code=500, content={"error": "Failed to send default reply"})

        return JSONResponse({
            "status": "unsupported",
            "message": "No pipeline keyword detected"
        })

    print(f"[WhatsApp] Mensaje → pipeline {pipeline_id}")

    conversation_history = whatsapp_conversations.get(from_number, []).copy()
    conversation_messages = conversation_history + [{"role": "user", "content": message_body}]

    raw_response = ""

    try:
        raw_response = run_pipeline_request(pipeline_id, conversation_messages)
        reply_text = sanitize_for_whatsapp(raw_response)
    except Exception as exc:
        print(f"[WhatsApp] Error ejecutando pipeline: {exc}")
        reply_text = "Lo siento, hubo un error procesando tu consulta."
    
    if not reply_text:
        reply_text = "No pude generar una respuesta en este momento."

    try:
        send_result = send_whatsapp_message(from_number, reply_text)
        print(f"[WhatsApp] Respuesta enviada: {send_result}")

        if raw_response:
            conversation_messages.append({"role": "assistant", "content": reply_text})
            if len(conversation_messages) > WHATSAPP_HISTORY_MAX_MESSAGES:
                conversation_messages = conversation_messages[-WHATSAPP_HISTORY_MAX_MESSAGES:]
            whatsapp_conversations[from_number] = conversation_messages
    except Exception as exc:
        print(f"[WhatsApp] Error enviando respuesta: {exc}")
        return JSONResponse(status_code=500, content={"error": "Failed to send response"})

    return JSONResponse({
        "status": "processed",
        "pipeline": pipeline_id
    })

# ======================
# ENDPOINT CATCH-ALL GENERAL
# ======================
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def catch_all_endpoint(path: str, request: Request):
    """Catch-all final para cualquier endpoint no manejado"""
    if path.startswith("v1/"):
        print(f"[Catch-All] {request.method} /{path} - Endpoint no encontrado")
        
        if "models" in path and "filter" not in path:
            return list_models()
        
        if any(pipeline_id in path for pipeline_id in pipelines_dict.keys()):
            return JSONResponse({
                "bypass": True,
                "message": f"Pipeline endpoint /{path} processed"
            })
        
        return JSONResponse({
            "error": f"Endpoint /{path} not found",
            "available_endpoints": [
                "/v1/models",
                "/v1/chat/completions",
                "/v1/pipelines",
                "/v1/{pipeline_id}/filter/inlet",
                "/v1/{pipeline_id}/filter/outlet"
            ]
        }, status_code=404)
    
    raise HTTPException(status_code=404, detail=f"Endpoint /{path} not found")

# ======================
# Eventos de startup/shutdown
# ======================
@app.on_event("startup")
async def startup():
    print("🚀 Backend v4.1 - OPTIMIZADO PARA VISUALIZACIÓN iniciado")
    print(f"📊 Pipelines: {len(pipelines_dict)}")
    print(f"📋 Lista: {list(pipelines_dict.keys())}")
    print(f"🌤️  API clima: {'✅' if TOMORROW_API_KEY else '❌'}")
    print(f"🔑 OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")
    print(f"🎥 Streaming: ✅ (sync + async)")
    print(f"🌐 CORS: ✅ (ultra permisivo)")
    print(f"📝 Logging: ✅ (enhanced)")
    print(f"✨ Formato OpenAI: ✅ (respuestas optimizadas)")
    print(f"🔧 Endpoints críticos:")
    print(f"   - /v1/{{pipeline_id}}/filter/inlet (formato OpenAI)")
    print(f"   - /v1/{{pipeline_id}}/filter/outlet")
    print(f"   - /v1/chat/completions")
    print(f"   - /v1/models (+ catch-all)")
    print(f"✅ MEJORAS APLICADAS:")
    print(f"   - Respuestas en formato OpenAI nativo")
    print(f"   - extract_pipeline_output optimizado")
    print(f"   - Detección automática de formato")

@app.on_event("shutdown")
async def shutdown():
    print("🔄 Backend cerrando...")

if __name__ == "__main__":
    import uvicorn
    print("🌟 Iniciando servidor v4.1 - OPTIMIZADO PARA VISUALIZACIÓN...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="info")