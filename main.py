import os
import time
import uuid
import json
import asyncio
import importlib
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
    description="Backend con pipelines LangChain compatible con Open WebUI - Versión Definitiva",
    version="4.0.0"
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
            "prompt_tokens": len(content.split()) * 2,  # Estimación
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
    # Chunk inicial
    yield create_streaming_chunk("", model, request_id)
    
    # Dividir en palabras para streaming natural
    words = content.split()
    chunk_size = 3  # Enviar 3 palabras por chunk
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " ".join(chunk_words)
        if i + chunk_size < len(words):
            chunk_content += " "
        
        yield create_streaming_chunk(chunk_content, model, request_id)
        await asyncio.sleep(0.05)  # Delay para streaming natural
    
    # Chunk final
    yield create_streaming_chunk("", model, request_id, is_final=True)
    yield "data: [DONE]\n\n"

def stream_content_sync(content: str, model: str, request_id: str) -> Generator[str, None, None]:
    """Genera streaming de contenido de manera síncrona"""
    # Chunk inicial
    yield create_streaming_chunk("", model, request_id)
    
    # Dividir en palabras
    words = content.split()
    chunk_size = 3
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " ".join(chunk_words)
        if i + chunk_size < len(words):
            chunk_content += " "
        
        yield create_streaming_chunk(chunk_content, model, request_id)
        time.sleep(0.02)  # Delay mínimo
    
    # Chunk final
    yield create_streaming_chunk("", model, request_id, is_final=True)
    yield "data: [DONE]\n\n"

def extract_pipeline_output(result: Any) -> str:
    """Extrae el contenido de la respuesta del pipeline"""
    if isinstance(result, dict):
        for key in ["output", "response", "result", "content", "message", "text"]:
            if key in result:
                return str(result[key])
        return str(result)
    else:
        return str(result)

def extract_user_message_from_request(request: Union[Dict, Any]) -> str:
    """Extrae mensaje del usuario de cualquier formato de request"""
    if isinstance(request, dict):
        # Formato Open WebUI
        if "body" in request:
            body = request["body"]
            if isinstance(body, dict) and "messages" in body:
                messages = body["messages"]
                if messages and isinstance(messages, list):
                    last_message = messages[-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        return last_message["content"]
        
        # Formato estándar
        if "messages" in request:
            messages = request["messages"]
            if messages and isinstance(messages, list):
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"]
        
        # Formato simple
        if "input" in request:
            return str(request["input"])
    
    return ""

def is_streaming_request(request: Union[Dict, Any]) -> bool:
    """Detecta si el request solicita streaming"""
    if isinstance(request, dict):
        # Nivel superior
        if request.get("stream", False):
            return True
        
        # En body
        if "body" in request and isinstance(request["body"], dict):
            if request["body"].get("stream", False):
                return True
    
    return False

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
    
    for root, dirs, files in os.walk(PIPELINES_DIR):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
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

# Cargar pipelines al inicio
load_pipelines()
print(f"🔄 Total pipelines cargados: {len(pipelines_dict)}")
print(f"📋 Pipelines disponibles: {list(pipelines_dict.keys())}")

# ======================
# Middleware de logging mejorado
# ======================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    print(f"[Request] {request.method} {request.url.path}")
    if request.headers.get("content-type") == "application/json":
        print(f"[Request] Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    print(f"[Response] {response.status_code} - {process_time:.3f}s")
    
    return response

# ======================
# Endpoints básicos
# ======================
@app.get("/")
def root():
    return {
        "message": "🚀 Backend LangChain para Open WebUI v4.0 - Funcionando Correctamente",
        "status": "running",
        "pipelines_count": len(pipelines_dict),
        "pipelines": list(pipelines_dict.keys()),
        "version": "4.0.0",
        "weather_api": "Tomorrow.io" if TOMORROW_API_KEY else "No configurada",
        "features": [
            "streaming_async", "streaming_sync", "chat_completions", 
            "pipelines", "weather", "cors_permissive", "logging_enhanced"
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
    
    # Modelos de OpenAI
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
    
    # Pipelines como modelos
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
    """Endpoint duplicado que Open WebUI busca - resuelve error 405"""
    print("[Models/Models] GET endpoint duplicado llamado")
    return list_models()

@app.post("/v1/models/models")
def models_models_post():
    """Endpoint POST para /v1/models/models"""
    print("[Models/Models] POST endpoint llamado")
    return list_models()

# ======================
# ⚠️ CRITICAL FIX: ENDPOINT ESPECÍFICO PARA CHAT COMPLETIONS MALFORMADO
# Debe ir ANTES del catch-all para que tenga prioridad
# ======================
@app.post("/v1/models/chat/completions")
async def models_chat_completions(request: Request):
    """Endpoint específico para /v1/models/chat/completions que Open WebUI usa"""
    print(f"[Models Chat Completions] Request interceptado - redirigiendo a chat completions normal")
    
    # Redirigir al endpoint correcto
    return await chat_completions(request)

# ======================
# ⚠️ CRITICAL FIX: ENDPOINTS ESPECÍFICOS PARA PIPELINES MALFORMADOS
# Open WebUI envía a /v1/models/{pipeline_id}/filter/{type} en lugar de /v1/{pipeline_id}/filter/{type}
# ======================
@app.post("/v1/models/{pipeline_id}/filter/inlet")
async def models_pipeline_inlet(pipeline_id: str, request: Request):
    """Endpoint específico para /v1/models/{pipeline_id}/filter/inlet que Open WebUI usa"""
    print(f"[Models Pipeline Inlet] Request interceptado para {pipeline_id} - redirigiendo a pipeline inlet normal")
    
    # Redirigir al endpoint correcto
    return await pipeline_filter_inlet(pipeline_id, request)

@app.post("/v1/models/{pipeline_id}/filter/outlet")
async def models_pipeline_outlet(pipeline_id: str, request: Request):
    """Endpoint específico para /v1/models/{pipeline_id}/filter/outlet que Open WebUI usa"""
    print(f"[Models Pipeline Outlet] Request interceptado para {pipeline_id} - redirigiendo a pipeline outlet normal")
    
    # Redirigir al endpoint correcto
    return await pipeline_filter_outlet(pipeline_id, request)

# ======================
# ⚠️ CRITICAL FIX: CATCH-ALL CORREGIDO - AHORA VA DESPUÉS
# ======================
@app.api_route("/v1/models/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def catch_models_paths(path: str, request: Request):
    """Catch-all CORREGIDO - Solo para modelos reales, NO pipelines ni chat"""
    print(f"[Models Catch-All] {request.method} /v1/models/{path}")
    
    # CRITICAL: Si el path contiene 'filter', es un pipeline - NO interceptar
    if 'filter' in path:
        print(f"[Models Catch-All] IGNORANDO pipeline request: {path}")
        print(f"[Models Catch-All] Request debe ir a endpoint correcto de pipeline")
        raise HTTPException(status_code=404, detail="Endpoint not handled by models catch-all")
    
    # Para chat/completions, debería haber sido manejado por el endpoint específico arriba
    if path == 'chat/completions':
        print(f"[Models Catch-All] ERROR: chat/completions llegó al catch-all (no debería pasar)")
        return JSONResponse({
            "error": "chat/completions should be handled by specific endpoint",
            "redirect_to": "/v1/chat/completions"
        }, status_code=500)
    
    # Solo para rutas de modelos normales
    return list_models()

# ======================
# Endpoints Chat - MAXIMA COMPATIBILIDAD
# ======================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Endpoint de chat con máxima compatibilidad y flexibilidad"""
    
    try:
        # Obtener datos del request de manera flexible
        request_data = await request.json()
        
        print(f"[Chat Completions] Request data: {request_data}")
        
        # Extraer campos
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
                
                # CORRECCIÓN: Agregar formato body para compatibilidad con pipelines
                pipeline_input = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model": model,
                    "stream": stream,
                    # NUEVO: Agregar formato body que los pipelines esperan
                    "body": {
                        "messages": messages,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": stream
                    }
                }
                
                # NUEVO: Si hay choices en el request original, incluirlos
                if "choices" in request_data:
                    pipeline_input["choices"] = request_data["choices"]
                    print(f"[Chat Completions] Choices incluidos en pipeline input")
                
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
    """
    Endpoint DEFINITIVO para Open WebUI - TODAS las compatibilidades
    """
    if pipeline_id not in pipelines_dict:
        print(f"[Filter Inlet] Pipeline {pipeline_id} no encontrado")
        return JSONResponse({"bypass": True, "message": "Pipeline not found"})
    
    try:
        # Obtener request data
        request_data = await request.json()
        
        print(f"[Filter Inlet] Pipeline: {pipeline_id}")
        print(f"[Filter Inlet] Request data keys: {list(request_data.keys())}")
        
        # Ejecutar pipeline
        pipeline_func = pipelines_dict[pipeline_id]["func"]
        result = pipeline_func(request_data)
        content = extract_pipeline_output(result)
        
        # Detectar streaming
        is_streaming = is_streaming_request(request_data)
        
        print(f"[Filter Inlet] Streaming: {is_streaming}")
        print(f"[Filter Inlet] Content length: {len(content)}")
        print(f"[Filter Inlet] Content preview: {content[:100]}...")
        
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        # Respuesta con streaming
        if is_streaming:
            print(f"[Filter Inlet] Iniciando streaming response")
            
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
            # Respuesta normal con formato OpenAI (COMO FUNCIONABA ORIGINALMENTE)
            response = create_chat_completion_response(content, pipeline_id, request_id)
            print(f"[Filter Inlet] Respuesta normal enviada")
            
            return JSONResponse(
                content=response,
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

# ======================
# ENDPOINT OUTLET Y OTROS FILTROS
# ======================
@app.post("/v1/{pipeline_id}/filter/outlet")
async def pipeline_filter_outlet(pipeline_id: str, request: Request):
    """
    Endpoint outlet para Open WebUI - resuelve error 405
    """
    try:
        request_data = await request.json()
        print(f"[Filter Outlet] Pipeline: {pipeline_id}")
        print(f"[Filter Outlet] Request data: {request_data}")
        
        # Open WebUI espera que devolvamos el contenido sin modificar
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
    """
    Catch-all para cualquier endpoint de filter que no exista
    """
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
# ENDPOINT CATCH-ALL GENERAL
# ======================
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def catch_all_endpoint(path: str, request: Request):
    """
    Catch-all final para cualquier endpoint no manejado
    """
    if path.startswith("v1/"):
        print(f"[Catch-All] {request.method} /{path} - Endpoint no encontrado")
        
        # Si es un endpoint de modelos, redirigir
        if "models" in path and "filter" not in path:
            return list_models()
        
        # Si es un endpoint de pipeline mal dirigido
        if any(pipeline_id in path for pipeline_id in pipelines_dict.keys()):
            return JSONResponse({
                "bypass": True,
                "message": f"Pipeline endpoint /{path} processed"
            })
        
        # Para otros endpoints v1
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
    
    # Para endpoints que no son v1, return 404 normal
    raise HTTPException(status_code=404, detail=f"Endpoint /{path} not found")

# ======================
# Eventos de startup/shutdown
# ======================
@app.on_event("startup")
async def startup():
    print("🚀 Backend v4.0 - FUNCIONANDO CORRECTAMENTE iniciado")
    print(f"📊 Pipelines: {len(pipelines_dict)}")
    print(f"📋 Lista: {list(pipelines_dict.keys())}")
    print(f"🌤️  API clima: {'✅' if TOMORROW_API_KEY else '❌'}")
    print(f"🔑 OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")
    print(f"🎥 Streaming: ✅ (sync + async)")
    print(f"🌐 CORS: ✅ (ultra permisivo)")
    print(f"📝 Logging: ✅ (enhanced)")
    print(f"🔧 Endpoints críticos:")
    print(f"   - /v1/{{pipeline_id}}/filter/inlet")
    print(f"   - /v1/{{pipeline_id}}/filter/outlet")
    print(f"   - /v1/chat/completions")
    print(f"   - /v1/models (+ catch-all CORREGIDO)")
    print(f"✅ FIX APLICADO: Catch-all ya NO intercepta pipelines")
    print(f"✅ FORMATO ORIGINAL: Respuestas OpenAI mantenidas")

@app.on_event("shutdown")
async def shutdown():
    print("🔄 Backend cerrando...")

if __name__ == "__main__":
    import uvicorn
    print("🌟 Iniciando servidor v4.0 - FUNCIONANDO CORRECTAMENTE...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="info")