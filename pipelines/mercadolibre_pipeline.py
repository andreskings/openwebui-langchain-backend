# mercadolibre_pipeline.py - Pipeline para consultar productos en MercadoLibre
import os
import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import urllib.parse

# ======================
# Metadata del pipeline
# ======================
metadata = {
    "name": "mercadolibre_search",
    "description": "Pipeline para buscar productos en MercadoLibre",
    "type": "search",
    "version": "1.0.0"
}

class MercadoLibrePipeline:
    """Clase principal para el pipeline de MercadoLibre"""
    
    def __init__(self):
        """Inicializa el pipeline con credenciales desde variables de entorno"""
        # Credenciales de MercadoLibre desde variables de entorno
        self.app_id = os.getenv("MERCADOLIBRE_APP_ID")
        self.client_secret = os.getenv("MERCADOLIBRE_CLIENT_SECRET")
        
        # Validar que las credenciales estén configuradas
        if not self.app_id or not self.client_secret:
            print("[ML Pipeline] ⚠️  Credenciales de MercadoLibre no configuradas")
            print("[ML Pipeline] Configurar MERCADOLIBRE_APP_ID y MERCADOLIBRE_CLIENT_SECRET en .env")
            # Para búsquedas básicas no se requieren credenciales, solo para funciones avanzadas
        
        # URLs base de la API
        self.base_url = "https://api.mercadolibre.com"
        self.sites = {
            "argentina": "MLA",
            "brasil": "MLB", 
            "chile": "MLC",
            "colombia": "MCO",
            "mexico": "MLM",
            "peru": "MPE",
            "uruguay": "MLU",
            "venezuela": "MLV"
        }
        self.default_site = "MLA"  # Argentina por defecto
        
        # Configuración
        self.max_results = 20
        self.timeout = 10
    
    def extract_search_query_robust(self, request: Dict[str, Any]) -> str:
        """Extrae la consulta de búsqueda del request con máxima robustez"""
        
        print(f"[ML Pipeline] Extrayendo query del request: {list(request.keys())}")
        print(f"[ML Pipeline] Request completo para debug: {json.dumps(request, indent=2, default=str)[:500]}...")
        
        # CASO 1: Si Open WebUI envía una respuesta previa como input (bucle)
        if "choices" in request and request["choices"]:
            choice = request["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                print(f"[ML Pipeline] Detectado bucle de respuesta: '{content[:50]}...'")
                # Si es una respuesta del sistema, ignorar y generar nueva
                if "MercadoLibre" in content or "Pipeline" in content:
                    return ""
                return content
        
        # CASO 2: Formato estándar de Open WebUI - CORREGIDO
        if "messages" in request and request["messages"]:
            messages = request["messages"]
            print(f"[ML Pipeline] Mensajes encontrados: {len(messages)}")
            if isinstance(messages, list) and messages:
                # Buscar el último mensaje del usuario
                for msg in reversed(messages):  # Empezar desde el final
                    if isinstance(msg, dict):
                        role = msg.get("role", "").lower()
                        content = msg.get("content", "")
                        
                        print(f"[ML Pipeline] Procesando mensaje - Role: {role}, Content: '{content[:50]}...'")
                        
                        # Solo tomar mensajes de usuario
                        if role == "user" and content and content.strip():
                            # Filtrar respuestas del sistema previas
                            if not any(word in content for word in ["MercadoLibre", "Pipeline", "ejecutado"]):
                                content = content.strip()
                                print(f"[ML Pipeline] Mensaje de usuario encontrado: '{content}'")
                                return content
        
        # CASO 3: Formato con body
        if "body" in request and isinstance(request["body"], dict):
            body = request["body"]
            if "messages" in body and body["messages"]:
                messages = body["messages"]
                if isinstance(messages, list) and messages:
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "").lower()
                            content = msg.get("content", "")
                            
                            if role == "user" and content and content.strip():
                                if not any(word in content for word in ["MercadoLibre", "Pipeline"]):
                                    content = content.strip()
                                    print(f"[ML Pipeline] De body.messages: '{content}'")
                                    return content
        
        # CASO 4: Formato simple
        if "input" in request:
            content = str(request["input"])
            print(f"[ML Pipeline] De input: '{content}'")
            return content
        
        # CASO 5: Buscar recursivamente cualquier contenido de usuario
        def find_user_content(obj, depth=0):
            if depth > 3:  # Evitar recursión infinita
                return None
            
            if isinstance(obj, dict):
                # Buscar patrones comunes
                for key in ["content", "text", "message", "query", "prompt", "search"]:
                    if key in obj and isinstance(obj[key], str) and obj[key].strip():
                        candidate = obj[key].strip()
                        # Filtrar respuestas del sistema
                        if not any(word in candidate for word in ["MercadoLibre", "Pipeline", "ejecutado"]):
                            return candidate
                
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
            print(f"[ML Pipeline] Búsqueda recursiva: '{recursive_content}'")
            return recursive_content
        
        print("[ML Pipeline] No se encontró contenido del usuario")
        return ""
    
    def detect_country_from_query(self, query: str) -> str:
        """Detecta el país desde la consulta para elegir el sitio correcto"""
        
        query_lower = query.lower()
        
        # Palabras clave por país
        country_keywords = {
            "MLA": ["argentina", "buenos aires", "peso argentino", "ars"],
            "MLB": ["brasil", "brazil", "sao paulo", "real", "brl"],
            "MLC": ["chile", "santiago", "peso chileno", "clp"],
            "MCO": ["colombia", "bogota", "peso colombiano", "cop"],
            "MLM": ["mexico", "méxico", "ciudad de mexico", "peso mexicano", "mxn"],
            "MPE": ["peru", "perú", "lima", "sol", "pen"],
            "MLU": ["uruguay", "montevideo", "peso uruguayo", "uyu"],
            "MLV": ["venezuela", "caracas", "bolivar", "vef"]
        }
        
        for site_id, keywords in country_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                print(f"[ML Pipeline] País detectado: {site_id}")
                return site_id
        
        return self.default_site
    
    def get_access_token(self) -> Optional[str]:
        """Obtiene token de acceso usando client credentials"""
        if not self.app_id or not self.client_secret:
            print("[ML Pipeline] No hay credenciales configuradas")
            return None
        
        try:
            token_url = f"{self.base_url}/oauth/token"
            data = {
                "grant_type": "client_credentials",
                "client_id": self.app_id,
                "client_secret": self.client_secret
            }
            
            response = requests.post(token_url, data=data, timeout=self.timeout)
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data.get("access_token")
            
            if access_token:
                print("[ML Pipeline] Token de acceso obtenido exitosamente")
                return access_token
            else:
                print("[ML Pipeline] No se obtuvo token de acceso")
                return None
                
        except Exception as e:
            print(f"[ML Pipeline] Error obteniendo token: {e}")
            return None
    
    def search_products(self, query: str, site_id: str = None, limit: int = None) -> Dict[str, Any]:
        """Busca productos en MercadoLibre"""
        
        if not query:
            return {"error": "Query de búsqueda vacío"}
        
        site = site_id or self.detect_country_from_query(query)
        limit = limit or self.max_results
        
        # Construir URL de búsqueda
        search_url = f"{self.base_url}/sites/{site}/search"
        params = {
            "q": query,
            "limit": limit,
            "offset": 0
        }
        
        print(f"[ML Pipeline] Buscando en {site}: '{query}'")
        print(f"[ML Pipeline] URL: {search_url}")
        
        # Preparar headers
        headers = {
            "User-Agent": "OpenWebUI-MercadoLibre-Pipeline/1.0"
        }
        
        # Intentar obtener token de acceso si hay credenciales
        access_token = self.get_access_token()
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
            print("[ML Pipeline] Usando autenticación con token")
        else:
            print("[ML Pipeline] Intentando búsqueda sin autenticación")
        
        try:
            response = requests.get(
                search_url, 
                params=params,
                timeout=self.timeout,
                headers=headers
            )
            
            response.raise_for_status()
            data = response.json()
            
            print(f"[ML Pipeline] Encontrados {len(data.get('results', []))} productos")
            
            return {
                "success": True,
                "query": query,
                "site": site,
                "total_results": data.get("paging", {}).get("total", 0),
                "results": data.get("results", []),
                "filters": data.get("available_filters", []),
                "metadata": {
                    "search_time": datetime.now().isoformat(),
                    "site_id": site,
                    "query_used": query
                }
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error en la búsqueda: {str(e)}"
            print(f"[ML Pipeline] ❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "query": query,
                "site": site
            }
        except json.JSONDecodeError as e:
            error_msg = f"Error decodificando respuesta JSON: {str(e)}"
            print(f"[ML Pipeline] ❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def format_product_results(self, search_result: Dict[str, Any]) -> str:
        """Formatea los resultados de búsqueda para mostrar al usuario"""
        
        if not search_result.get("success"):
            return f"❌ Error en la búsqueda: {search_result.get('error', 'Error desconocido')}"
        
        results = search_result.get("results", [])
        query = search_result.get("query", "")
        site = search_result.get("site", "")
        total = search_result.get("total_results", 0)
        
        if not results:
            return f"🔍 No se encontraron productos para '{query}' en MercadoLibre {site}"
        
        # Mapeo de códigos de país a nombres
        country_names = {
            "MLA": "Argentina", "MLB": "Brasil", "MLC": "Chile", 
            "MCO": "Colombia", "MLM": "México", "MPE": "Perú", 
            "MLU": "Uruguay", "MLV": "Venezuela"
        }
        
        country_name = country_names.get(site, site)
        
        # Construir respuesta formateada
        response = f"🛒 **Resultados de MercadoLibre {country_name}**\n"
        response += f"📝 Búsqueda: *{query}*\n"
        response += f"📊 Mostrando {len(results)} de {total:,} productos encontrados\n\n"
        
        for i, product in enumerate(results[:10], 1):  # Mostrar máximo 10
            title = product.get("title", "Sin título")
            price = product.get("price", 0)
            currency = product.get("currency_id", "")
            condition = product.get("condition", "")
            shipping = product.get("shipping", {})
            free_shipping = "🚚 Envío gratis" if shipping.get("free_shipping") else ""
            permalink = product.get("permalink", "")
            
            # Formatear precio
            if price:
                formatted_price = f"{currency} {price:,.0f}".replace(",", ".")
            else:
                formatted_price = "Consultar precio"
            
            # Formatear condición
            condition_emoji = "🆕" if condition == "new" else "📦" if condition == "used" else "🔄"
            condition_text = "Nuevo" if condition == "new" else "Usado" if condition == "used" else condition.title()
            
            response += f"**{i}. {title}**\n"
            response += f"💰 {formatted_price} {condition_emoji} {condition_text}\n"
            if free_shipping:
                response += f"{free_shipping}\n"
            if permalink:
                response += f"🔗 [Ver producto]({permalink})\n"
            response += "\n"
        
        if len(results) > 10:
            response += f"... y {len(results) - 10} productos más\n\n"
        
        response += f"💡 *Tip: Puedes especificar el país en tu búsqueda (ej: 'notebooks en Chile')*"
        
        return response
    
    def get_help_message(self) -> str:
        """Mensaje de ayuda para el usuario"""
        return """🛒 **Pipeline de MercadoLibre - Ayuda**

**¿Cómo usar?**
Simplemente escribe lo que quieres buscar en MercadoLibre.

**Ejemplos:**
• `buscar laptops gaming`
• `notebooks en Chile`
• `celulares Samsung Argentina`
• `zapatos deportivos`
• `insumos agrícolas Brasil`

**Países disponibles:**
🇦🇷 Argentina • 🇧🇷 Brasil • 🇨🇱 Chile • 🇨🇴 Colombia
🇲🇽 México • 🇵🇪 Perú • 🇺🇾 Uruguay • 🇻🇪 Venezuela

Por defecto busca en Argentina. Puedes especificar otro país en tu consulta.

**¿Necesitas ayuda?** Escribe 'ayuda mercadolibre'"""

def pipeline(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función principal del pipeline de MercadoLibre
    
    Maneja búsquedas de productos en MercadoLibre desde Open WebUI
    """
    
    try:
        print(f"[ML Pipeline] Iniciando búsqueda en MercadoLibre v1.0")
        print(f"[ML Pipeline] Request keys: {list(request.keys())}")
        
        # Debugging: mostrar estructura del request
        if "choices" in request:
            print(f"[ML Pipeline] DETECTADO BUCLE - choices presente")
        
        # Inicializar el pipeline
        ml_pipeline = MercadoLibrePipeline()
        
        # Extraer parámetros del request
        model_id = request.get("model", "mercadolibre_search")
        temperature = request.get("temperature", 0.0)  # No aplica para búsquedas
        
        print(f"[ML Pipeline] Modelo: {model_id}")
        
        # Extraer consulta de búsqueda con método robusto
        search_query = ml_pipeline.extract_search_query_robust(request)
        
        if not search_query:
            print("[ML Pipeline] No se encontró consulta de búsqueda")
            return {
                "output": ml_pipeline.get_help_message(),
                "metadata": {
                    "pipeline": metadata["name"],
                    "status": "help_shown",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        # Verificar si es solicitud de ayuda
        if any(word in search_query.lower() for word in ["ayuda", "help", "como usar", "cómo usar"]):
            return {
                "output": ml_pipeline.get_help_message(),
                "metadata": {
                    "pipeline": metadata["name"],
                    "status": "help_requested",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        print(f"[ML Pipeline] Consulta extraída: '{search_query}'")
        
        # Realizar búsqueda
        search_result = ml_pipeline.search_products(search_query)
        
        # Formatear respuesta
        formatted_response = ml_pipeline.format_product_results(search_result)
        
        print(f"[ML Pipeline] Búsqueda completada")
        print(f"[ML Pipeline] Respuesta generada: {len(formatted_response)} caracteres")
        print(f"[ML Pipeline] Respuesta preview: {formatted_response[:100]}...")
        
        return {
            "output": formatted_response,
            "metadata": {
                "pipeline": metadata["name"],
                "query": search_query,
                "results_count": len(search_result.get("results", [])) if search_result.get("success") else 0,
                "site": search_result.get("site", ""),
                "success": search_result.get("success", False),
                "model": model_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        error_msg = f"Error en MercadoLibre Pipeline: {str(e)}"
        print(f"[ML Pipeline] ❌ {error_msg}")
        print(f"[ML Pipeline] ❌ Exception type: {type(e)}")
        print(f"[ML Pipeline] ❌ Request completo: {request}")
        
        return {
            "output": f"❌ Ocurrió un error buscando en MercadoLibre.\n\n**Error:** {str(e)}\n\nPor favor intenta con otra búsqueda o verifica tu conexión a internet.",
            "error": str(e),
            "metadata": {
                "pipeline": metadata["name"],
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
        }