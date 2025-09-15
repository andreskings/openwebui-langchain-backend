# langchain_pipeline.py - Pipeline completo con LangChain + LangGraph + LangSmith
import os
from typing import Dict, Any, List, Union
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph

# LangSmith (opcional)
try:
    from langsmith import Client, traceable
    LANGSMITH_AVAILABLE = True
    client_ls = Client()
    print("[LangSmith] Cliente inicializado correctamente")
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("[LangSmith] No disponible - continuando sin trazas")

# ======================
# Metadata del pipeline
# ======================
metadata = {
       "name": "langchain_v3",  # Cambiar esto
       "description": "Pipeline LangChain v3.0 funcional",
       "type": "llm",
       "version": "3.0.0"
   }

class LangChainPipeline:
    """Clase principal para el pipeline de LangChain"""
    
    def __init__(self):
        """Inicializa el pipeline con configuración"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.default_model = os.getenv("MODEL_ID", "gpt-4o-mini")
        self.default_temperature = float(os.getenv("TEMPERATURE", 0.2))
        
        # Configuración de LangSmith
        self.langsmith_project = os.getenv("LANGCHAIN_PROJECT")
        self.langsmith_enabled = LANGSMITH_AVAILABLE and bool(self.langsmith_project)
        
        if self.langsmith_enabled:
            print(f"[LangChain Pipeline] LangSmith habilitado - Proyecto: {self.langsmith_project}")
        else:
            print("[LangChain Pipeline] LangSmith deshabilitado")
    
    def extract_user_message_robust(self, request: Dict[str, Any]) -> str:
        """Extrae el mensaje del usuario del request con máxima robustez"""
        
        print(f"[LangChain Extract] Request structure: {list(request.keys())}")
        
        # CASO 1: Si Open WebUI envía una respuesta previa como input (bucle)
        if "choices" in request and request["choices"]:
            choice = request["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                print(f"[LangChain Extract] Detectado bucle de respuesta: '{content[:50]}...'")
                # Si es una respuesta del sistema, ignorar y generar nueva
                if "Pipeline" in content or "ejecutado" in content:
                    return ""
                return content
        
        # CASO 2: Formato Open WebUI normal
        if "body" in request and isinstance(request["body"], dict):
            body = request["body"]
            if "messages" in body and body["messages"]:
                messages = body["messages"]
                if isinstance(messages, list) and messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        content = last_message["content"]
                        print(f"[LangChain Extract] De body.messages: '{content}'")
                        return content
        
        # CASO 3: Formato estándar
        if "messages" in request and request["messages"]:
            messages = request["messages"]
            if isinstance(messages, list) and messages:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    content = last_message["content"]
                    print(f"[LangChain Extract] De messages: '{content}'")
                    return content
        
        # CASO 4: Formato simple
        if "input" in request:
            content = str(request["input"])
            print(f"[LangChain Extract] De input: '{content}'")
            return content
        
        # CASO 5: Buscar recursivamente cualquier contenido de usuario
        def find_user_content(obj, depth=0):
            if depth > 3:  # Evitar recursión infinita
                return None
            
            if isinstance(obj, dict):
                # Buscar patrones comunes
                for key in ["content", "text", "message", "query", "prompt"]:
                    if key in obj and isinstance(obj[key], str) and obj[key].strip():
                        return obj[key].strip()
                
                # Buscar en estructuras anidadas
                for value in obj.values():
                    result = find_user_content(value, depth + 1)
                    if result:
                        return result
            
            elif isinstance(obj, list):
                for item in obj:
                    result = find_user_content(item, depth + 1)
                    if result:
                        return result
            
            return None
        
        recursive_content = find_user_content(request)
        if recursive_content:
            print(f"[LangChain Extract] Búsqueda recursiva: '{recursive_content}'")
            return recursive_content
        
        print("[LangChain Extract] No se encontró contenido del usuario")
        return ""
    
    def extract_messages_robust(self, request: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extrae todos los mensajes del request con robustez mejorada"""
        
        # CASO 1: Respuesta de chat completion (bucle)
        if "choices" in request and request["choices"]:
            # Es una respuesta previa, no mensajes de usuario
            user_msg = self.extract_user_message_robust(request)
            if user_msg and not ("Pipeline" in user_msg or "ejecutado" in user_msg):
                return [{"role": "user", "content": user_msg}]
            else:
                return []
        
        # CASO 2: Formato Open WebUI
        if "body" in request and isinstance(request["body"], dict):
            body = request["body"]
            if "messages" in body and isinstance(body["messages"], list):
                messages = body["messages"]
                # Filtrar mensajes válidos
                valid_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        content = msg["content"].strip()
                        if content and not ("Pipeline" in content or "ejecutado" in content):
                            valid_messages.append({
                                "role": msg["role"],
                                "content": content
                            })
                if valid_messages:
                    return valid_messages
        
        # CASO 3: Formato estándar
        if "messages" in request and isinstance(request["messages"], list):
            messages = request["messages"]
            valid_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    content = msg["content"].strip()
                    if content and not ("Pipeline" in content or "ejecutado" in content):
                        valid_messages.append({
                            "role": msg["role"],
                            "content": content
                        })
            if valid_messages:
                return valid_messages
        
        # CASO 4: Crear mensaje único del input
        user_input = self.extract_user_message_robust(request)
        if user_input and user_input.strip():
            return [{"role": "user", "content": user_input.strip()}]
        
        return []
    
    def convert_to_langchain_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Convierte mensajes a formato LangChain"""
        langchain_messages = []
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()
            
            if not content:
                continue
            
            # Filtrar respuestas del sistema previas
            if "Pipeline" in content or "ejecutado correctamente" in content:
                continue
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                # Por defecto, tratar como mensaje de usuario
                langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages
    
    def get_llm(self, model_id: str = None, temperature: float = None) -> ChatOpenAI:
        """Crea una instancia del LLM"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY no configurada en variables de entorno")
        
        # CORRECCIÓN: Mapear nombre del pipeline a modelo real de OpenAI
        real_model = "gpt-4o-mini" if model_id == "langchain_v3" else (model_id or self.default_model)
        
        return ChatOpenAI(
            model=real_model,  # Usar real_model en lugar de model_id
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=4000,
            timeout=30
        )
    
    def create_conversation_graph(self, llm: ChatOpenAI) -> StateGraph:
        """Crea el grafo de conversación con LangGraph"""
        
        def chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Nodo principal de chat"""
            messages = state.get("messages", [])
            
            if not messages:
                return {"response": "Hola! Soy tu asistente inteligente. ¿En qué puedo ayudarte hoy?"}
            
            try:
                # Invocar el LLM
                result = llm.invoke(messages)
                return {"response": result.content}
            
            except Exception as e:
                print(f"[Chat Node] Error: {e}")
                return {"response": f"Disculpa, ocurrió un error procesando tu consulta: {str(e)}"}
        
        def preprocessing_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Nodo de preprocesamiento"""
            messages = state.get("messages", [])
            
            # Agregar mensaje del sistema si no existe
            has_system = any(isinstance(msg, SystemMessage) for msg in messages)
            
            if not has_system and messages:
                system_prompt = self.get_system_prompt()
                messages.insert(0, SystemMessage(content=system_prompt))
            
            return {"messages": messages}
        
        # Construir el grafo
        graph = StateGraph(dict)
        
        # Agregar nodos
        graph.add_node("preprocessing", preprocessing_node)
        graph.add_node("chat", chat_node)
        
        # Definir flujo
        graph.set_entry_point("preprocessing")
        graph.add_edge("preprocessing", "chat")
        graph.set_finish_point("chat")
        
        return graph.compile()
    
    def get_system_prompt(self) -> str:
        """Genera el prompt del sistema"""
        return f"""Eres un asistente inteligente y útil creado con LangChain. Características:

- Respondes de manera clara, precisa y útil
- Mantienes conversaciones naturales y fluidas  
- Puedes ayudar con una amplia variedad de temas
- Eres amigable pero profesional
- Adaptas tu tono según el contexto
- Proporcionas información precisa y actualizada
- Usas LangChain + LangGraph para procesar consultas

Fecha y hora actual: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Responde siempre en español a menos que el usuario solicite específicamente otro idioma."""
    
    def run_with_tracing(self, graph, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta el grafo con trazas de LangSmith si está habilitado"""
        
        if self.langsmith_enabled:
            @traceable(
                name="LangChainPipelineExecution",
                project_name=self.langsmith_project,
                metadata={
                    "pipeline": metadata["name"],
                    "version": metadata["version"],
                    "model": state.get("model", self.default_model)
                }
            )
            def traced_execution():
                return graph.invoke(state)
            
            return traced_execution()
        else:
            return graph.invoke(state)

def pipeline(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función principal del pipeline de LangChain
    
    Maneja conversaciones usando LangChain + LangGraph con trazas opcionales de LangSmith
    """
    
    try:
        print(f"[LangChain Pipeline] Iniciando procesamiento v3.0")
        print(f"[LangChain Pipeline] Request keys: {list(request.keys())}")
        
        # Debugging: mostrar estructura del request
        if "choices" in request:
            print(f"[LangChain Pipeline] DETECTADO BUCLE - choices presente")
        
        # Inicializar el pipeline
        pipeline_instance = LangChainPipeline()
        
        # Extraer parámetros del request
        model_id = request.get("model", pipeline_instance.default_model)
        temperature = request.get("temperature", pipeline_instance.default_temperature)
        
        print(f"[LangChain Pipeline] Modelo: {model_id}")
        print(f"[LangChain Pipeline] Temperatura: {temperature}")
        
        # Extraer mensajes con método robusto
        messages_raw = pipeline_instance.extract_messages_robust(request)
        
        if not messages_raw:
            print("[LangChain Pipeline] No se encontraron mensajes válidos")
            return {"output": "Hola! Soy tu asistente LangChain. Escribe tu consulta para comenzar nuestra conversación."}
        
        print(f"[LangChain Pipeline] Mensajes extraídos: {len(messages_raw)}")
        for i, msg in enumerate(messages_raw):
            print(f"[LangChain Pipeline] Mensaje {i}: {msg['role']} - '{msg['content'][:50]}...'")
        
        # Convertir a formato LangChain
        langchain_messages = pipeline_instance.convert_to_langchain_messages(messages_raw)
        
        if not langchain_messages:
            print("[LangChain Pipeline] No se generaron mensajes LangChain válidos")
            return {"output": "No pude procesar tu mensaje. Por favor, intenta reformular tu consulta."}
        
        print(f"[LangChain Pipeline] Mensajes LangChain: {len(langchain_messages)}")
        
        # Crear LLM
        llm = pipeline_instance.get_llm(model_id, temperature)
        print(f"[LangChain Pipeline] LLM creado: {llm.model_name}")
        
        # Crear grafo de conversación
        graph = pipeline_instance.create_conversation_graph(llm)
        print(f"[LangChain Pipeline] Grafo creado")
        
        # Preparar estado inicial
        state = {
            "messages": langchain_messages,
            "model": model_id,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat()
        }
        
        # Ejecutar con o sin trazas
        print(f"[LangChain Pipeline] Ejecutando grafo...")
        result = pipeline_instance.run_with_tracing(graph, state)
        
        # Extraer respuesta
        response_content = result.get("response", "No se generó respuesta")
        
        print(f"[LangChain Pipeline] Respuesta generada: {len(response_content)} caracteres")
        print(f"[LangChain Pipeline] Respuesta preview: {response_content[:100]}...")
        
        return {
            "output": response_content,
            "metadata": {
                "pipeline": metadata["name"],
                "model": model_id,
                "temperature": temperature,
                "messages_processed": len(langchain_messages),
                "langsmith_enabled": pipeline_instance.langsmith_enabled,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        error_msg = f"Error en LangChain Pipeline: {str(e)}"
        print(f"[LangChain Pipeline] ❌ {error_msg}")
        print(f"[LangChain Pipeline] ❌ Exception type: {type(e)}")
        print(f"[LangChain Pipeline] ❌ Request completo: {request}")
        
        return {
            "output": f"Disculpa, ocurrió un error procesando tu consulta con LangChain.\n\nDetalle técnico: {str(e)}\n\nPor favor intenta nuevamente o reformula tu consulta.",
            "error": str(e),
            "metadata": {
                "pipeline": metadata["name"],
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
        }