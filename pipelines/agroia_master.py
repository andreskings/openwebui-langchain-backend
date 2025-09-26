# super_agroia_chat.py - Pipeline Inteligente de Chat Agrícola Completo
import os
import sys
import requests
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import importlib.util

metadata = {
    "name": "super_agroia_chat",
    "description": "Super Chat IA Agrícola - Asistente completo con acceso a clima, precios, RAG y múltiples fuentes",
    "type": "llm", 
    "version": "1.0.0"
}

class SuperAgroIAChat:
    """Super Chat IA que puede consultar múltiples fuentes y razonar como un ingeniero agrónomo"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # Configuración de herramientas disponibles
        self.tools = {
            # Pipelines especializados
            "weather_forecast": {
                "file": "weather_pipeline.py",
                "description": "Pronósticos meteorológicos detallados de Chile",
                "capabilities": ["clima actual", "pronóstico 7 días", "condiciones para aplicaciones", "alertas meteorológicas"]
            },
            "market_prices": {
                "file": "mercadolibre_pipeline.py", 
                "description": "Precios de productos agrícolas y insumos",
                "capabilities": ["precios herbicidas", "fertilizantes", "maquinaria", "semillas", "comparar proveedores"]
            },
            "web_search": {
                "file": "google_pipeline.py",
                "description": "Búsquedas especializadas en web",
                "capabilities": ["información técnica", "plagas y enfermedades", "mejores prácticas", "regulaciones"]
            },
            "document_rag": {
                "file": "langchain_pipeline.py",
                "description": "Análisis de documentos y RAG sobre archivos específicos",
                "capabilities": ["análisis de suelos", "informes técnicos", "manuales", "regulaciones específicas"]
            },
            
            # OpenAI especializado
            "agronomist_expert": {
                "type": "openai",
                "model": "gpt-4o",
                "description": "Conocimiento experto en agronomía y agricultura",
                "system_prompt": """Eres un ingeniero agrónomo experto con 20+ años de experiencia. 
                Conoces sobre: cultivos, suelos, plagas, enfermedades, fertilización, riego, maquinaria agrícola, 
                meteorología aplicada, fitosanidad, y regulaciones agrícolas.
                Siempre das recomendaciones prácticas, considerando factores económicos y ambientales."""
            },
            "technical_advisor": {
                "type": "openai", 
                "model": "gpt-4o-mini",
                "description": "Asesor técnico para procedimientos y cálculos",
                "system_prompt": """Eres un asesor técnico especializado en agricultura de precisión.
                Ayudas con: cálculos de dosificación, calendarios de aplicación, interpretación de análisis,
                optimización de recursos, y recomendaciones técnicas específicas."""
            }
        }
        
        # Contexto de conocimiento agrícola base
        self.agricultural_context = {
            "crops": ["trigo", "maíz", "soja", "avena", "cebada", "arroz", "papa", "tomate", "uva", "palta"],
            "seasons": {
                "spring": ["septiembre", "octubre", "noviembre"],
                "summer": ["diciembre", "enero", "febrero"], 
                "autumn": ["marzo", "abril", "mayo"],
                "winter": ["junio", "julio", "agosto"]
            },
            "critical_periods": {
                "siembra": ["marzo", "abril", "septiembre", "octubre"],
                "aplicaciones": ["octubre", "noviembre", "diciembre", "enero"],
                "cosecha": ["febrero", "marzo", "abril", "mayo"]
            }
        }
    
    def analyze_agricultural_query(self, user_query: str) -> Dict[str, Any]:
        """Análisis inteligente específico para consultas agrícolas"""
        
        analysis_prompt = f"""
Eres un experto en análisis de consultas agrícolas. Analiza esta consulta y determina:

CONSULTA: "{user_query}"

HERRAMIENTAS DISPONIBLES:
- weather_forecast: Clima y pronósticos (Chile)
- market_prices: Precios de insumos agrícolas 
- web_search: Búsquedas técnicas especializadas
- document_rag: Análisis de documentos/RAG específicos
- agronomist_expert: Conocimiento experto en agronomía
- technical_advisor: Asesor técnico y cálculos

PATRONES A DETECTAR:
1. ¿Menciona clima/tiempo/pronóstico? → weather_forecast
2. ¿Menciona precios/costos/comprar? → market_prices  
3. ¿Necesita info técnica/investigación? → web_search
4. ¿Menciona documentos/análisis/fundo específico? → document_rag
5. ¿Requiere conocimiento agronómico profundo? → agronomist_expert
6. ¿Necesita cálculos/procedimientos técnicos? → technical_advisor

CONTEXTO AGRÍCOLA:
- Detecta cultivos mencionados: {', '.join(self.agricultural_context['crops'])}
- Identifica época del año y relación con actividades
- Considera factores como plagas, enfermedades, fertilización, riego

Responde en JSON:
{{
  "query_type": "tipo de consulta (ej: aplicacion_herbicida, analisis_suelo, planificacion_cultivo)",
  "complexity": "simple|medium|complex",
  "tools_needed": ["herramienta1", "herramienta2"],
  "agricultural_factors": ["factor1", "factor2"],
  "reasoning": "explicación del análisis",
  "execution_strategy": "como combinar las respuestas"
}}
"""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": analysis_prompt}],
                    "temperature": 0.2,
                    "max_tokens": 400
                },
                timeout=15
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                
                # Extraer JSON
                if content.startswith('```'):
                    content = content.split('\n', 1)[1].rsplit('\n', 1)[0]
                
                result = json.loads(content)
                
                # Validar herramientas
                valid_tools = [tool for tool in result.get("tools_needed", []) if tool in self.tools]
                result["tools_needed"] = valid_tools
                
                return result
                
        except Exception as e:
            print(f"[SuperAgroIA] Error en análisis inteligente: {e}")
        
        # Fallback a análisis por reglas
        return self.analyze_by_rules(user_query)
    
    def analyze_by_rules(self, user_query: str) -> Dict[str, Any]:
        """Análisis de respaldo usando reglas específicas agrícolas"""
        query_lower = user_query.lower()
        tools = []
        factors = []
        
        # Detección por palabras clave agrícolas
        if any(word in query_lower for word in ["clima", "tiempo", "pronóstico", "lluvia", "temperatura", "viento"]):
            tools.append("weather_forecast")
            factors.append("condiciones_meteorológicas")
        
        if any(word in query_lower for word in ["precio", "costo", "comprar", "herbicida", "fertilizante", "insumo"]):
            tools.append("market_prices") 
            factors.append("factor_económico")
        
        if any(word in query_lower for word in ["buscar", "investigar", "plaga", "enfermedad", "técnica"]):
            tools.append("web_search")
            factors.append("información_técnica")
        
        if any(word in query_lower for word in ["documento", "análisis", "suelo", "fundo", "informe"]):
            tools.append("document_rag")
            factors.append("análisis_específico")
        
        # Siempre incluir experto agrónomo para consultas complejas
        if len(tools) > 1 or any(crop in query_lower for crop in self.agricultural_context["crops"]):
            tools.append("agronomist_expert")
            factors.append("conocimiento_agronómico")
        
        # Default mínimo
        if not tools:
            tools = ["agronomist_expert"]
            factors = ["consulta_general"]
        
        complexity = "simple" if len(tools) == 1 else "medium" if len(tools) == 2 else "complex"
        
        return {
            "query_type": "consulta_agricola",
            "complexity": complexity,
            "tools_needed": tools[:3],  # Máximo 3 herramientas
            "agricultural_factors": factors,
            "reasoning": "Análisis automático por patrones agrícolas",
            "execution_strategy": "consultar_fuentes_y_sintetizar"
        }
    
    def execute_tool(self, tool_name: str, user_query: str, context: str = "") -> str:
        """Ejecuta una herramienta específica"""
        
        if tool_name not in self.tools:
            return f"❌ Herramienta {tool_name} no disponible"
        
        tool_config = self.tools[tool_name]
        
        try:
            if tool_config.get("type") == "openai":
                # Es un modelo de OpenAI especializado
                return self.call_specialized_openai(tool_name, user_query, context)
            else:
                # Es un pipeline local
                return self.call_local_pipeline(tool_name, user_query, context)
                
        except Exception as e:
            print(f"[SuperAgroIA] Error ejecutando {tool_name}: {e}")
            return f"❌ Error en {tool_name}: {str(e)}"
    
    def call_local_pipeline(self, tool_name: str, user_query: str, context: str = "") -> str:
        """Ejecuta un pipeline local específico"""
        
        tool_config = self.tools[tool_name]
        pipeline_file = tool_config["file"]
        
        try:
            # Buscar archivo del pipeline
            possible_paths = [
                os.path.join(os.path.dirname(__file__), pipeline_file),
                os.path.join(os.path.dirname(__file__), "..", pipeline_file),
                os.path.join(os.getcwd(), pipeline_file),
                pipeline_file
            ]
            
            pipeline_path = None
            for path in possible_paths:
                if os.path.exists(os.path.abspath(path)):
                    pipeline_path = os.path.abspath(path)
                    break
            
            if not pipeline_path:
                return f"❌ Pipeline {pipeline_file} no encontrado"
            
            print(f"[SuperAgroIA] Ejecutando {tool_name}: {pipeline_path}")
            
            # Importar dinámicamente
            spec = importlib.util.spec_from_file_location(tool_name, pipeline_path)
            module = importlib.util.module_from_spec(spec)
            
            # Agregar directorio al path
            pipeline_dir = os.path.dirname(pipeline_path)
            if pipeline_dir not in sys.path:
                sys.path.insert(0, pipeline_dir)
            
            spec.loader.exec_module(module)
            
            # Preparar request con contexto adicional
            enhanced_query = user_query
            if context:
                enhanced_query = f"CONTEXTO: {context}\n\nCONSULTA: {user_query}"
            
            request_data = {
                "body": {
                    "messages": [{"role": "user", "content": enhanced_query}],
                    "model": tool_name,
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                "messages": [{"role": "user", "content": enhanced_query}],
                "model": tool_name,
                "user": {"name": "SuperAgroIA", "id": "super_chat"}
            }
            
            # Ejecutar pipeline
            if hasattr(module, 'pipeline'):
                result = module.pipeline(request_data)
                
                # Extraer respuesta
                if isinstance(result, dict):
                    for key in ["output", "response", "result", "content"]:
                        if key in result and result[key]:
                            return str(result[key])
                    return str(result)
                else:
                    return str(result)
            else:
                return f"❌ {pipeline_file} sin función pipeline"
                
        except Exception as e:
            return f"❌ Error ejecutando {tool_name}: {str(e)}"
    
    def call_specialized_openai(self, tool_name: str, user_query: str, context: str = "") -> str:
        """Llama a OpenAI con configuración especializada"""
        
        tool_config = self.tools[tool_name]
        
        try:
            # Construir prompt con contexto
            messages = [
                {"role": "system", "content": tool_config["system_prompt"]}
            ]
            
            if context:
                messages.append({"role": "system", "content": f"CONTEXTO ADICIONAL: {context}"})
            
            messages.append({"role": "user", "content": user_query})
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": tool_config["model"],
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"❌ Error OpenAI {tool_name}: HTTP {response.status_code}"
                
        except Exception as e:
            return f"❌ Error OpenAI {tool_name}: {str(e)}"
    
    def execute_multi_tool_strategy(self, tools_needed: List[str], user_query: str, analysis: Dict) -> Dict[str, str]:
        """Ejecuta múltiples herramientas con estrategia específica"""
        
        results = {}
        execution_context = ""
        
        # Orden de ejecución optimizado para consultas agrícolas
        execution_order = self.optimize_execution_order(tools_needed, analysis)
        
        for i, tool in enumerate(execution_order):
            print(f"[SuperAgroIA] 🔧 Ejecutando {tool} ({i+1}/{len(execution_order)})")
            
            # Construir contexto acumulativo
            if execution_context:
                context = f"INFORMACIÓN PREVIA: {execution_context}"
            else:
                context = f"ANÁLISIS: {analysis['reasoning']}"
            
            # Ejecutar herramienta
            result = self.execute_tool(tool, user_query, context)
            results[tool] = result
            
            # Actualizar contexto para próximas herramientas
            if not result.startswith("❌") and len(result) > 50:
                execution_context += f"\n{tool}: {result[:200]}..."
            
            # Log resultado
            status = "✅" if not result.startswith("❌") else "❌"
            preview = result[:80] + "..." if len(result) > 80 else result
            print(f"[SuperAgroIA] {status} {tool}: {preview}")
        
        return results
    
    def optimize_execution_order(self, tools: List[str], analysis: Dict) -> List[str]:
        """Optimiza el orden de ejecución según el tipo de consulta"""
        
        # Orden preferido para diferentes tipos de consultas
        priority_orders = {
            "weather_first": ["weather_forecast", "agronomist_expert", "market_prices", "web_search", "document_rag", "technical_advisor"],
            "market_first": ["market_prices", "web_search", "agronomist_expert", "weather_forecast", "document_rag", "technical_advisor"],
            "technical_first": ["document_rag", "web_search", "technical_advisor", "agronomist_expert", "weather_forecast", "market_prices"],
            "expert_first": ["agronomist_expert", "weather_forecast", "market_prices", "web_search", "document_rag", "technical_advisor"]
        }
        
        # Determinar estrategia según análisis
        query_type = analysis.get("query_type", "")
        
        if "clima" in query_type or "weather" in analysis.get("agricultural_factors", []):
            order = priority_orders["weather_first"]
        elif "precio" in query_type or "económico" in str(analysis.get("agricultural_factors", [])):
            order = priority_orders["market_first"]
        elif "análisis" in query_type or "documento" in query_type:
            order = priority_orders["technical_first"]
        else:
            order = priority_orders["expert_first"]
        
        # Filtrar solo las herramientas necesarias en el orden optimizado
        return [tool for tool in order if tool in tools]
    
    def synthesize_agricultural_response(self, user_query: str, analysis: Dict, tool_results: Dict[str, str]) -> str:
        """Síntesis especializada para respuestas agrícolas"""
        
        # Filtrar resultados válidos
        valid_results = {
            tool: result for tool, result in tool_results.items()
            if result and not result.startswith("❌") and len(result.strip()) > 30
        }
        
        if not valid_results:
            # Fallback total con experto agrónomo
            print("[SuperAgroIA] Sin resultados válidos, usando fallback de experto")
            fallback = self.execute_tool("agronomist_expert", user_query)
            if not fallback.startswith("❌"):
                return f"{fallback}\n\n---\n*🌱 Respuesta del experto agrónomo (fallback)*"
            else:
                return "❌ **Error en todas las herramientas consultadas.** Por favor, reformula tu consulta o contacta soporte técnico."
        
        # Respuesta única válida
        if len(valid_results) == 1:
            tool, result = list(valid_results.items())[0]
            tool_desc = self.tools[tool]["description"]
            return f"{result}\n\n---\n*🔧 Fuente: {tool_desc}*"
        
        # Múltiples fuentes - síntesis avanzada
        synthesis_prompt = f"""
Eres un experto ingeniero agrónomo que debe sintetizar información de múltiples fuentes para dar la mejor respuesta.

CONSULTA ORIGINAL: "{user_query}"

TIPO DE CONSULTA: {analysis.get('query_type', 'consulta agrícola')}
FACTORES CONSIDERADOS: {', '.join(analysis.get('agricultural_factors', []))}

INFORMACIÓN RECOPILADA:
"""
        
        for tool, result in valid_results.items():
            tool_desc = self.tools[tool]["description"]
            synthesis_prompt += f"\n=== {tool.upper()} ({tool_desc}) ===\n{result}\n"
        
        synthesis_prompt += f"""

INSTRUCCIONES PARA LA SÍNTESIS:
1. Combina toda la información relevante de manera coherente
2. Prioriza recomendaciones prácticas y actionables
3. Considera factores económicos, técnicos y ambientales
4. Incluye advertencias o precauciones importantes
5. Estructura la respuesta de forma clara y profesional
6. Si hay conflictos entre fuentes, explica las diferencias
7. Proporciona pasos concretos cuando sea aplicable

Genera una respuesta final completa y profesional:"""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",  # Usar modelo más potente para síntesis
                    "messages": [{"role": "user", "content": synthesis_prompt}],
                    "temperature": 0.6,
                    "max_tokens": 2500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                final_response = response.json()['choices'][0]['message']['content']
                
                # Agregar metadatos
                tools_used = list(valid_results.keys())
                tools_desc = [self.tools[tool]["description"] for tool in tools_used]
                footer = f"\n\n---\n*🔧 Fuentes consultadas: {', '.join(tools_desc)}*"
                
                return final_response + footer
            else:
                # Fallback: mejor respuesta individual
                return self.get_best_individual_result(valid_results)
                
        except Exception as e:
            print(f"[SuperAgroIA] Error en síntesis: {e}")
            return self.get_best_individual_result(valid_results)
    
    def get_best_individual_result(self, valid_results: Dict[str, str]) -> str:
        """Selecciona la mejor respuesta individual"""
        
        # Priorizar por relevancia agrícola
        priority = ["agronomist_expert", "weather_forecast", "market_prices", "document_rag", "web_search", "technical_advisor"]
        
        for tool in priority:
            if tool in valid_results:
                result = valid_results[tool]
                tool_desc = self.tools[tool]["description"]
                return f"{result}\n\n---\n*📋 Respuesta principal: {tool_desc}*"
        
        # Si no hay prioridades, usar la más larga (más completa)
        best_tool = max(valid_results.keys(), key=lambda k: len(valid_results[k]))
        return f"{valid_results[best_tool]}\n\n---\n*🔧 {self.tools[best_tool]['description']}*"

def extract_user_message(request: Dict[str, Any]) -> str:
    """Extrae el mensaje del usuario del request"""
    try:
        # OpenWebUI format
        if "body" in request and "messages" in request["body"]:
            messages = request["body"]["messages"]
            if messages:
                return messages[-1].get("content", "").strip()
        
        # Direct messages
        if "messages" in request:
            messages = request["messages"]
            if messages:
                return messages[-1].get("content", "").strip()
        
        # Alternative formats
        for key in ["prompt", "query", "input", "text"]:
            if key in request and request[key]:
                return str(request[key]).strip()
                
    except Exception as e:
        print(f"[SuperAgroIA] Error extrayendo mensaje: {e}")
    
    return ""

def pipeline(request: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline principal del Super Chat IA Agrícola"""
    
    try:
        # Extraer consulta del usuario
        user_query = extract_user_message(request)
        
        if not user_query:
            return {"output": (
                "🌱 **Super AgroIA Chat - Tu Ingeniero Agrónomo Virtual**\n\n"
                "¡Hola! Soy tu asistente agrícola inteligente que puede:\n\n"
                "🧠 **Analizar consultas complejas** sobre agricultura\n"
                "🌤️ **Consultar clima y pronósticos** para planificar aplicaciones\n"
                "💰 **Buscar precios** de herbicidas, fertilizantes e insumos\n"
                "📚 **Hacer RAG** sobre documentos específicos de tu fundo\n"
                "🔍 **Investigar** plagas, enfermedades y técnicas\n"
                "⚡ **Sintetizar todo** en recomendaciones prácticas\n\n"
                "**Ejemplos de consultas:**\n"
                "• *\"Necesito aplicar herbicida X el día Y, ¿cómo está el pronóstico?\"*\n"
                "• *\"Análisis del suelo del fundo Z y recomendaciones\"*\n" 
                "• *\"Mejor momento para sembrar trigo según clima y precios\"*\n"
                "• *\"Control de plaga Y en cultivo Z, opciones y costos\"*\n\n"
                "**¿Qué consulta agrícola tienes hoy?** 🚜"
            )}
        
        print(f"[SuperAgroIA] 🌱 Nueva consulta agrícola: '{user_query[:70]}{'...' if len(user_query) > 70 else ''}'")
        
        # Crear instancia del super chat
        super_chat = SuperAgroIAChat()
        
        # Fase 1: Análisis inteligente de la consulta
        print("[SuperAgroIA] 📊 Analizando consulta agrícola...")
        analysis = super_chat.analyze_agricultural_query(user_query)
        
        print(f"[SuperAgroIA] 🎯 Tipo: {analysis['query_type']}")
        print(f"[SuperAgroIA] 🔧 Herramientas: {analysis['tools_needed']}")
        print(f"[SuperAgroIA] 🌾 Factores: {analysis['agricultural_factors']}")
        print(f"[SuperAgroIA] 💭 Complejidad: {analysis['complexity']}")
        
        # Fase 2: Ejecución de herramientas
        print("[SuperAgroIA] 🚜 Ejecutando herramientas especializadas...")
        tool_results = super_chat.execute_multi_tool_strategy(
            analysis["tools_needed"], 
            user_query, 
            analysis
        )
        
        # Fase 3: Síntesis agrícola especializada
        print("[SuperAgroIA] 🌱 Sintetizando respuesta agrícola...")
        final_response = super_chat.synthesize_agricultural_response(
            user_query, 
            analysis, 
            tool_results
        )
        
        # Agregar metadatos finales
        timestamp = datetime.now().strftime('%H:%M')
        complexity_emoji = {"simple": "🟢", "medium": "🟡", "complex": "🔴"}
        
        if len(final_response) < 2000:
            metadata = (f"\n\n🕐 *{timestamp}* • "
                       f"{complexity_emoji.get(analysis['complexity'], '⚫')} *{analysis['complexity'].title()}* • "
                       f"🔧 *{len(analysis['tools_needed'])} herramientas*")
            final_response += metadata
        
        print(f"[SuperAgroIA] ✅ Respuesta completada - {len(final_response)} caracteres")
        
        return {"output": final_response}
        
    except Exception as e:
        error_msg = str(e)
        print(f"[SuperAgroIA] ❌ Error crítico: {error_msg}")
        return {"output": (
            f"❌ **Error en Super AgroIA Chat:** {error_msg}\n\n"
            "🔧 **Soluciones posibles:**\n"
            "• Verifica que tus pipelines estén funcionando\n"
            "• Revisa la configuración de OpenAI API\n"
            "• Intenta reformular tu consulta\n\n"
            "💬 **Contacta soporte técnico si el problema persiste**"
        )}

# Función principal
def main(request):
    """Función principal para compatibilidad"""
    return pipeline(request)
