# google_search_ai_pipeline.py - Pipeline híbrido Google Search + OpenAI optimizado para compras
import os
import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import urllib.parse

# LangChain imports para OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ======================
# Metadata del pipeline
# ======================
metadata = {
    "name": "google_search",
    "description": "Pipeline híbrido que busca en Google y responde con OpenAI optimizado para encontrar sitios de compra",
    "type": "search_ai",
    "version": "2.1.0"
}

class GoogleSearchAIPipeline:
    """Pipeline híbrido Google Search + OpenAI optimizado para compras"""
    
    def __init__(self):
        """Inicializa el pipeline con credenciales"""
        # Google Search API
        self.google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        # OpenAI para respuestas
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.default_model = os.getenv("MODEL_ID", "gpt-4o-mini")
        self.default_temperature = float(os.getenv("TEMPERATURE", 0.7))
        
        # URLs y configuración
        self.google_base_url = "https://www.googleapis.com/customsearch/v1"
        self.max_results = 10
        self.timeout = 15
        
        print(f"[Google Search AI] Google API configurada: {bool(self.google_api_key)}")
        print(f"[Google Search AI] OpenAI API configurada: {bool(self.openai_api_key)}")
    
    def extract_user_message_robust(self, request: Dict[str, Any]) -> str:
        """Extrae el mensaje del usuario - COPIADO DE LANGCHAIN"""
        
        print(f"[Google Search AI] Extract - Request structure: {list(request.keys())}")
        
        # CASO 1: Si Open WebUI envía una respuesta previa como input (bucle)
        if "choices" in request and request["choices"]:
            choice = request["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                print(f"[Google Search AI] Extract - Detectado bucle de respuesta: '{content[:50]}...'")
                # Si es una respuesta del sistema, ignorar y generar nueva
                if "Pipeline" in content or "búsqueda" in content or "Google" in content:
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
                        print(f"[Google Search AI] Extract - De body.messages: '{content}'")
                        return content
        
        # CASO 3: Formato estándar
        if "messages" in request and request["messages"]:
            messages = request["messages"]
            if isinstance(messages, list) and messages:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    content = last_message["content"]
                    print(f"[Google Search AI] Extract - De messages: '{content}'")
                    return content
        
        # CASO 4: Formato simple
        if "input" in request:
            content = str(request["input"])
            print(f"[Google Search AI] Extract - De input: '{content}'")
            return content
        
        # CASO 5: Buscar recursivamente cualquier contenido de usuario
        def find_user_content(obj, depth=0):
            if depth > 3:
                return None
            
            if isinstance(obj, dict):
                for key in ["content", "text", "message", "query", "prompt"]:
                    if key in obj and isinstance(obj[key], str) and obj[key].strip():
                        return obj[key].strip()
                
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
            print(f"[Google Search AI] Extract - Búsqueda recursiva: '{recursive_content}'")
            return recursive_content
        
        print("[Google Search AI] Extract - No se encontró contenido del usuario")
        return ""
    
    def detect_commercial_intent(self, query: str) -> bool:
        """Detecta si la consulta tiene intención comercial"""
        commercial_keywords = [
            "comprar", "precio", "venta", "tienda", "donde", "dónde", "adquirir", 
            "distribuidor", "vender", "cotizar", "cotización", "proveedor", 
            "costo", "valor", "conseguir", "encontrar", "busco", "necesito",
            "mercadolibre", "amazon", "ebay", "tienda", "farmacia", "agroquímica"
        ]
        
        return any(keyword in query.lower() for keyword in commercial_keywords)
    
    def enhance_search_query(self, original_query: str) -> str:
        """Mejora la consulta de búsqueda para encontrar sitios de compra"""
        
        # Detectar tipo de producto
        query_lower = original_query.lower()
        
        # Para productos agroquímicos/herbicidas
        if any(word in query_lower for word in ["herbicida", "fungicida", "insecticida", "isoxaflutol", "agroquímica"]):
            enhanced_query = f"{original_query} comprar venta distribuidor agroquímica tienda online precio Chile"
        
        # Para medicamentos
        elif any(word in query_lower for word in ["medicamento", "medicina", "fármaco", "droga"]):
            enhanced_query = f"{original_query} farmacia comprar precio donde conseguir"
        
        # Para productos generales
        elif self.detect_commercial_intent(original_query):
            enhanced_query = f"{original_query} precio tienda online"
        
        else:
            # Agregar términos comerciales automáticamente
            enhanced_query = f"{original_query} comprar venta precio distribuidor tienda"
        
        print(f"[Google Search AI] Query original: '{original_query}'")
        print(f"[Google Search AI] Query mejorada: '{enhanced_query}'")
        
        return enhanced_query
    
    def search_google(self, query: str) -> Dict[str, Any]:
        """Busca en Google con consulta optimizada para compras"""
        
        if not self.google_api_key:
            return {"success": False, "error": "Google API Key no configurada"}
        
        # Mejorar la consulta para encontrar sitios de compra
        enhanced_query = self.enhance_search_query(query)
        
        params = {
            "key": self.google_api_key,
            "cx": self.search_engine_id,
            "q": enhanced_query,
            "num": self.max_results,
            "gl": "cl",  # Geolocalización Chile
            "hl": "es"   # Idioma español
        }
        
        print(f"[Google Search AI] Buscando en Google: '{enhanced_query}'")
        
        try:
            response = requests.get(
                self.google_base_url,
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": "OpenWebUI-GoogleSearch-AI/2.1"}
            )
            
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                error_info = data["error"]
                return {
                    "success": False,
                    "error": f"Error de Google API: {error_info.get('message', 'Error desconocido')}",
                    "query": query
                }
            
            items = data.get("items", [])
            search_info = data.get("searchInformation", {})
            
            print(f"[Google Search AI] Encontrados {len(items)} resultados")
            
            # Extraer información útil para el AI, priorizando sitios comerciales
            search_results = []
            commercial_sites = []
            
            for item in items:
                result_data = {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "displayLink": item.get("displayLink", "")
                }
                
                # Identificar sitios comerciales
                commercial_indicators = [
                    "mercadolibre", "amazon", "ebay", "tienda", "venta", "precio",
                    "comprar", "distribuidor", "agroquímica", "farmacia", "shop",
                    "store", "comercial", "productos", "catalogo"
                ]
                
                is_commercial = any(indicator in item.get("title", "").lower() or 
                                  indicator in item.get("snippet", "").lower() or
                                  indicator in item.get("displayLink", "").lower()
                                  for indicator in commercial_indicators)
                
                if is_commercial:
                    commercial_sites.append(result_data)
                else:
                    search_results.append(result_data)
            
            # Priorizar sitios comerciales
            final_results = commercial_sites + search_results
            final_results = final_results[:8]  # Limitar a 8 resultados
            
            print(f"[Google Search AI] Sitios comerciales encontrados: {len(commercial_sites)}")
            
            return {
                "success": True,
                "query": query,
                "enhanced_query": enhanced_query,
                "total_results": search_info.get("totalResults", "0"),
                "search_time": search_info.get("searchTime", 0),
                "results": final_results,
                "commercial_sites_count": len(commercial_sites)
            }
            
        except Exception as e:
            print(f"[Google Search AI] Error en búsqueda: {e}")
            return {
                "success": False,
                "error": f"Error en búsqueda: {str(e)}",
                "query": query
            }
    
    def get_llm(self, model_id: str = None, temperature: float = None) -> ChatOpenAI:
        """Crea instancia del LLM - COPIADO DE LANGCHAIN"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY no configurada")
        
        # Mapear nombre del pipeline a modelo real
        real_model = "gpt-4o-mini" if model_id == "google_search" else (model_id or self.default_model)
        
        return ChatOpenAI(
            model=real_model,
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=4000,
            timeout=30
        )
    
    def create_search_prompt(self, user_query: str, search_results: List[Dict], search_data: Dict) -> str:
        """Crea prompt optimizado para información comercial"""
        
        if not search_results:
            return f"""El usuario pregunta: "{user_query}"

No se encontraron resultados de búsqueda para esta consulta. Responde basándote en tu conocimiento general y sugiere reformular la consulta o lugares generales donde podrían encontrar el producto."""
        
        # Separar sitios comerciales de informativos
        commercial_sites = []
        info_sites = []
        
        for result in search_results:
            commercial_indicators = [
                "mercadolibre", "amazon", "tienda", "venta", "precio", "comprar", 
                "distribuidor", "shop", "store", "comercial", "catalogo"
            ]
            
            is_commercial = any(indicator in result['title'].lower() or 
                              indicator in result['snippet'].lower() or
                              indicator in result['displayLink'].lower()
                              for indicator in commercial_indicators)
            
            if is_commercial:
                commercial_sites.append(result)
            else:
                info_sites.append(result)
        
        # Construir texto de resultados priorizando comerciales
        results_text = ""
        
        if commercial_sites:
            results_text += "\n**SITIOS DE COMPRA ENCONTRADOS:**\n"
            for i, result in enumerate(commercial_sites, 1):
                results_text += f"""
{i}. **{result['title']}**
   Sitio: {result['displayLink']}
   Descripción: {result['snippet']}
   Enlace directo: {result['link']}
"""
        
        if info_sites:
            results_text += "\n**INFORMACIÓN ADICIONAL:**\n"
            for i, result in enumerate(info_sites, len(commercial_sites) + 1):
                results_text += f"""
{i}. **{result['title']}**
   Fuente: {result['displayLink']}
   Información: {result['snippet']}
   URL: {result['link']}
"""
        
        return f"""El usuario está buscando: "{user_query}"

He encontrado {len(search_results)} resultados, incluyendo {len(commercial_sites)} sitios comerciales:

{results_text}

INSTRUCCIONES ESPECÍFICAS:
1. **PRIORIZA LOS SITIOS DE COMPRA**: Menciona primero donde puede comprar el producto
2. **INCLUYE ENLACES DIRECTOS**: Siempre proporciona los enlaces exactos como enlaces clickeables usando formato [Texto del enlace](URL)
3. **INFORMACIÓN PRÁCTICA**: Incluye precios si están disponibles, formas de contacto, disponibilidad
4. **DISTRIBUIDORES OFICIALES**: Si encuentras distribuidores autorizados, destácalos
5. **ALTERNATIVAS**: Si no hay suficientes opciones, sugiere otros lugares donde buscar
6. **FORMATO CLARO**: Organiza la respuesta con títulos claros (ej: "Dónde comprar:", "Precios:", "Contacto:")
7. **ENLACES FUNCIONALES**: Asegúrate de que todos los enlaces estén en formato [texto](url) para que sean clickeables
8. **INFORMACIÓN LOCAL**: Prioriza opciones disponibles en Chile si es relevante

Genera una respuesta completa y práctica que ayude al usuario a encontrar exactamente donde puede adquirir lo que busca, con enlaces directos y información de contacto."""
    
    def get_help_message(self) -> str:
        """Mensaje de ayuda"""
        return """🔍 **Google Search AI - Búsqueda Inteligente con Enlaces de Compra**

**¿Cómo funciona?**
Busco información actual en Google, encuentro sitios de compra y te doy enlaces directos donde puedes adquirir productos.

**Ejemplos de consultas:**
• `donde comprar Isoxaflutol`
• `precio herbicidas Chile`
• `tiendas de agroquímicos`
• `distribuidores de pesticidas`
• `comprar medicamentos online`

**Lo que obtienes:**
🛒 Enlaces directos a tiendas online
💰 Información de precios cuando está disponible
📞 Datos de contacto de distribuidores
🌐 Sitios web oficiales de productos
📍 Opciones locales en Chile

**Ventajas:**
- Resultados actualizados de Google
- Enlaces clickeables para comprar directamente
- Información de precios y disponibilidad
- Respuestas inteligentes con IA

Escribe lo que necesitas comprar y te ayudo a encontrar donde conseguirlo."""

def pipeline(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline principal híbrido Google Search + OpenAI optimizado para compras
    
    Busca sitios de compra y proporciona enlaces directos
    """
    
    try:
        print(f"[Google Search AI] Iniciando pipeline híbrido v2.1 - Optimizado para compras")
        print(f"[Google Search AI] Request keys: {list(request.keys())}")
        
        # Debugging bucle
        if "choices" in request:
            print(f"[Google Search AI] DETECTADO BUCLE - choices presente")
        
        # Inicializar pipeline
        pipeline_instance = GoogleSearchAIPipeline()
        
        # Extraer parámetros
        model_id = request.get("model", "google_search")
        temperature = request.get("temperature", pipeline_instance.default_temperature)
        
        print(f"[Google Search AI] Modelo: {model_id}")
        print(f"[Google Search AI] Temperatura: {temperature}")
        
        # Extraer consulta del usuario
        user_query = pipeline_instance.extract_user_message_robust(request)
        
        if not user_query:
            print("[Google Search AI] No se encontró consulta válida")
            return {
                "output": pipeline_instance.get_help_message(),
                "metadata": {
                    "pipeline": metadata["name"],
                    "status": "help_shown",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        # Verificar si es solicitud de ayuda
        help_keywords = ["ayuda", "help", "como usar", "cómo usar"]
        if any(word in user_query.lower() for word in help_keywords):
            return {
                "output": pipeline_instance.get_help_message(),
                "metadata": {
                    "pipeline": metadata["name"],
                    "status": "help_requested", 
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        print(f"[Google Search AI] Consulta extraída: '{user_query}'")
        
        # PASO 1: Buscar en Google con optimización comercial
        search_result = pipeline_instance.search_google(user_query)
        
        if not search_result.get("success"):
            # Si Google falla, responder solo con OpenAI
            print(f"[Google Search AI] Google Search falló: {search_result.get('error')}")
            search_results = []
            search_data = {}
        else:
            search_results = search_result.get("results", [])
            search_data = search_result
        
        # PASO 2: Crear prompt optimizado para compras
        search_prompt = pipeline_instance.create_search_prompt(user_query, search_results, search_data)
        print(f"[Google Search AI] Prompt comercial creado: {len(search_prompt)} caracteres")
        
        # PASO 3: Usar OpenAI para generar respuesta enfocada en compras
        try:
            llm = pipeline_instance.get_llm(model_id, temperature)
            
            # Crear mensajes para LangChain
            messages = [
                SystemMessage(content="Eres un asistente especializado en ayudar a encontrar donde comprar productos. Tu objetivo principal es proporcionar enlaces directos, información de precios y datos de contacto de vendedores. Siempre incluye enlaces clickeables en formato [texto](url). Prioriza sitios de compra sobre información general."),
                HumanMessage(content=search_prompt)
            ]
            
            print(f"[Google Search AI] Invocando OpenAI con enfoque comercial...")
            response = llm.invoke(messages)
            ai_response = response.content
            
            print(f"[Google Search AI] Respuesta comercial generada: {len(ai_response)} caracteres")
            print(f"[Google Search AI] Preview: {ai_response[:100]}...")
            
            return {
                "output": ai_response,
                "metadata": {
                    "pipeline": metadata["name"],
                    "query": user_query,
                    "enhanced_query": search_data.get("enhanced_query", user_query),
                    "search_success": search_result.get("success", False),
                    "results_used": len(search_results),
                    "commercial_sites": search_data.get("commercial_sites_count", 0),
                    "total_results": search_result.get("total_results", "0"),
                    "model": model_id,
                    "temperature": temperature,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"[Google Search AI] Error en OpenAI: {e}")
            return {
                "output": f"❌ Error generando respuesta: {str(e)}\n\nIntenta reformular tu consulta.",
                "error": str(e),
                "metadata": {
                    "pipeline": metadata["name"],
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
    except Exception as e:
        error_msg = f"Error en Google Search AI Pipeline: {str(e)}"
        print(f"[Google Search AI] ❌ {error_msg}")
        
        return {
            "output": f"❌ Ocurrió un error en el pipeline.\n\n**Error:** {str(e)}\n\nPor favor intenta nuevamente.",
            "error": str(e),
            "metadata": {
                "pipeline": metadata["name"],
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
        }