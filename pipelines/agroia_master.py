# super_agroia_master_integrated.py - AgroIA Master con RAG integrado v2.2
import os
import sys
import requests
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import importlib.util

# Intentar importar el sistema RAG de múltiples formas
RAG_AVAILABLE = False
EnhancedRAGSystem = None

# Método 1: Importación directa
try:
    from pipelines.rag_agricultural_enhanced import EnhancedRAGSystem
    RAG_AVAILABLE = True
    print("[AgroIA] ✓ RAG importado directamente")
except ImportError as e:
    print(f"[AgroIA] Advertencia: Import directo falló: {e}")
    
    # Método 2: Importación dinámica con path
    try:
        possible_paths = [
            "rag_agricultural_enhanced.py",
            "./rag_agricultural_enhanced.py",
            "../rag_agricultural_enhanced.py",
            os.path.join(os.path.dirname(__file__), "rag_agricultural_enhanced.py"),
            os.path.join(os.getcwd(), "rag_agricultural_enhanced.py")
        ]
        
        rag_path = None
        for path in possible_paths:
            full_path = os.path.abspath(path)
            if os.path.exists(full_path):
                rag_path = full_path
                print(f"[AgroIA] Encontrado RAG en: {rag_path}")
                break
        
        if rag_path:
            spec = importlib.util.spec_from_file_location("rag_agricultural_enhanced", rag_path)
            rag_module = importlib.util.module_from_spec(spec)
            sys.modules["rag_agricultural_enhanced"] = rag_module
            spec.loader.exec_module(rag_module)
            EnhancedRAGSystem = rag_module.EnhancedRAGSystem
            RAG_AVAILABLE = True
            print("[AgroIA] ✓ RAG importado dinámicamente")
        else:
            print("[AgroIA] Archivo rag_agricultural_enhanced.py no encontrado")
            print(f"[AgroIA] Buscado en: {os.getcwd()}")
            
    except Exception as e:
        print(f"[AgroIA] Error en import dinámico: {e}")
        import traceback
        traceback.print_exc()

metadata = {
    "name": "super_agroia_master_integrated",
    "description": "Super AgroIA Master - Chat inteligente con RAG, clima, precios y experto agrónomo integrado",
    "type": "llm",
    "version": "2.2.0"
}


class RAGWrapper:
    """Wrapper ligero para el sistema RAG"""
    def __init__(self):
        print(f"\n[RAGWrapper] Inicializando...")
        print(f"[RAGWrapper] RAG_AVAILABLE = {RAG_AVAILABLE}")
        print(f"[RAGWrapper] EnhancedRAGSystem = {EnhancedRAGSystem}")
        
        if RAG_AVAILABLE and EnhancedRAGSystem:
            try:
                print("[RAGWrapper] Creando instancia de EnhancedRAGSystem...")
                self.rag_system = EnhancedRAGSystem()
                docs_count = self.rag_system.vector_store.count()
                self.enabled = True
                print(f"[RAGWrapper] ✓ Inicializado exitosamente - {docs_count} documentos en Chroma")
            except Exception as e:
                print(f"[RAGWrapper] Error inicializando EnhancedRAGSystem: {e}")
                import traceback
                traceback.print_exc()
                self.rag_system = None
                self.enabled = False
        else:
            print("[RAGWrapper] RAG no disponible")
            self.rag_system = None
            self.enabled = False
    
    def query(self, user_query: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Consulta al sistema RAG con formato optimizado"""
        print(f"\n[RAGWrapper.query] Iniciando consulta")
        print(f"[RAGWrapper.query] enabled = {self.enabled}")
        print(f"[RAGWrapper.query] rag_system = {self.rag_system}")
        
        if not self.enabled or not self.rag_system:
            print("[RAGWrapper.query] RAG no habilitado o sin sistema")
            return None
        
        try:
            total_docs = self.rag_system.vector_store.count()
            print(f"[RAGWrapper.query] {total_docs} documentos en base")
            
            if total_docs == 0:
                print("[RAGWrapper.query] Base de datos vacía")
                return {
                    "success": False,
                    "response": None,
                    "metadata": {"chunks_found": 0, "total_chunks": 0}
                }
            
            print(f"[RAGWrapper.query] Buscando chunks para: '{user_query[:50]}...'")
            relevant_chunks = self.rag_system.search_and_rerank(user_query, top_k=12, rerank_k=6)
            
            print(f"[RAGWrapper.query] Encontrados {len(relevant_chunks)} chunks relevantes")
            
            if not relevant_chunks:
                print("[RAGWrapper.query] Sin chunks relevantes")
                return {
                    "success": False,
                    "response": None,
                    "metadata": {"chunks_found": 0, "total_chunks": total_docs}
                }
            
            print(f"[RAGWrapper.query] Generando respuesta con LLM...")
            response_data = self.rag_system.generate_grounded_response(user_query, relevant_chunks)
            
            if response_data.get("error"):
                print(f"[RAGWrapper.query] Error generando respuesta: {response_data.get('error')}")
                return None
            
            final_response = response_data['answer']
            print(f"[RAGWrapper.query] ✓ Respuesta generada: {len(final_response)} caracteres")
            
            # Agregar referencias
            if response_data.get('sources'):
                final_response += "\n\n" + "=" * 78
                final_response += "\n" + " " * 28 + "REFERENCIAS"
                final_response += "\n" + "=" * 78 + "\n"
                
                for ref_id in sorted(response_data['sources'].keys(), key=lambda x: int(x.strip('[]'))):
                    source_info = response_data['sources'][ref_id]
                    final_response += f"\n- Citation {ref_id}"
                    final_response += f"\n  Source: {source_info['source_file']}"
                    final_response += f"\n  Relevance: {source_info['similarity']*100:.2f}%"
                    final_response += f"\n  Section: {source_info['section_type']}"
                    final_response += "\n" + "-" * 78 + "\n"
            
            return {
                "success": True,
                "response": final_response,
                "metadata": {
                    "chunks_found": len(relevant_chunks),
                    "total_chunks": total_docs
                }
            }
            
        except Exception as e:
            print(f"[RAGWrapper.query] ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        return None


class SuperAgroIAMaster:
    """Orquestador inteligente con RAG integrado"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.rag = RAGWrapper()
        
        # Herramientas disponibles
        self.tools = {
            "rag": {
                "enabled": self.rag.enabled,
                "description": "Documentos agrícolas (etiquetas, análisis, manuales)",
                "keywords": ["etiqueta", "documento", "manual", "análisis", "producto", "dosis", "aplicación"]
            },
            "weather": {
                "file": "pipelines/weather_pipeline.py",
                "description": "Clima y pronósticos de Chile",
                "keywords": ["clima", "tiempo", "pronóstico", "lluvia", "temperatura", "mañana", "aplicar", "hoy", 
                           "chillan", "santiago", "valparaiso", "concepcion", "temuco", "valdivia", "osorno", 
                           "puerto montt", "iquique", "antofagasta", "la serena", "rancagua", "talca", "curico",
                           "linares", "chillán", "los angeles", "angol", "villarrica", "pucón", "castro",
                           "coyhaique", "punta arenas", "arica", "calama", "copiapó", "ovalle", "quillota"]
            },
            "prices": {
                "file": "pipelines/mercadolibre_pipeline.py",
                "description": "Precios de insumos agrícolas",
                "keywords": ["precio", "costo", "comprar", "valor"]
            },
            "expert": {
                "type": "openai",
                "model": "gpt-4o",
                "description": "Experto agrónomo",
                "system_prompt": """Eres un ingeniero agrónomo con 20+ años de experiencia.

REGLAS CRÍTICAS DE SEGURIDAD:
- NUNCA recomiendes un herbicida si no está EXPLÍCITAMENTE documentado que controla esa maleza específica
- NUNCA recomiendes un herbicida si no está EXPLÍCITAMENTE documentado que es apto para ese cultivo específico
- Si no hay información específica, di claramente: "No se encuentra información documentada sobre [producto] para [maleza] en [cultivo]"
- PROHIBIDO inventar o asumir compatibilidades no documentadas

REGLAS ULTRA-ESTRICTAS PARA MALEZAS:
- Si preguntan "¿qué herbicida para [maleza]?" y los documentos NO mencionan explícitamente esa maleza específica, responde: "No se encuentra información documentada sobre herbicidas específicos para [maleza] en la base de datos actual"
- NO recomiendes productos basándote en similitudes o suposiciones
- SOLO cita productos que EXPLÍCITAMENTE mencionen la maleza en cuestión
- Si un documento habla de "malezas de hoja ancha" pero no menciona "correhuela" específicamente, NO lo recomiendes para correhuela
- VERIFICA que el documento mencione la maleza por su nombre común O científico antes de recomendarlo

Proporciona recomendaciones prácticas considerando factores técnicos, económicos y ambientales.
Sé conciso pero completo. Prioriza la seguridad y eficacia."""
            }
        }
    
    def analyze_query(self, query: str, history: List[Dict[str, str]] = None) -> List[str]:
        """Análisis inteligente mejorado de qué herramientas necesita"""
        query_lower = query.lower()
        needed = []
        
        context_text = query_lower
        mentioned_products = set()
        
        # Detectar si la consulta actual es sobre malezas específicas
        weed_indicators = ["maleza", "malezas", "controlar", "control", "eliminar", "herbicida para", "que herbicida", "qué herbicida"]
        is_weed_query = any(indicator in query_lower for indicator in weed_indicators)
        
        # Solo usar historial si NO es una consulta nueva sobre malezas
        if history and not is_weed_query:
            for msg in history:
                content_lower = msg.get("content", "").lower()
                for product in ["shark", "fordor", "vulcano", "zorro", "tiburon", "asulox"]:
                    if product in content_lower:
                        mentioned_products.add(product)
                        print(f"[Análisis] Producto del historial: {product}")
                
                context_text += " " + content_lower
        elif is_weed_query:
            print(f"[Análisis] Consulta sobre malezas detectada - ignorando historial para búsqueda amplia")
        
        # PRIORIDAD 1: Detectar productos/documentos específicos
        product_indicators = [
            "shark", "fordor", "vulcano", "zorro", "tiburon", "asulox", "aliado", "sencor", 
            "glifoglex", "genius", "kerb", "loyant", "tordon", "raker", "atrazina", "producto",
            "etiqueta", "ficha", "manual", "hoja de seguridad",
            "dosis", "aplicacion", "aplicaciones", "temporada",
            "maximo", "minimo", "limite", "intervalo", "carencia",
            "aplicarlo", "usarlo", "ese producto"
        ]
        
        has_product_reference = any(ind in context_text for ind in product_indicators)
        
        if (has_product_reference or mentioned_products) and self.rag.enabled:
            needed.append("rag")
            print(f"[Análisis] RAG activado - Productos: {mentioned_products or 'detectado en query'}")
        
        # PRIORIDAD 2: Detectar otras herramientas por palabras clave
        for tool_name, config in self.tools.items():
            if tool_name == "rag":
                continue
            if not config.get("enabled", True):
                continue
            
            keywords = config.get("keywords", [])
            if any(kw in query_lower for kw in keywords):
                needed.append(tool_name)
                print(f"[Análisis] {tool_name} activado - keyword detectada")
        
        # PRIORIDAD 3: Si no detectó nada específico
        if not needed:
            if self.rag.enabled:
                needed.append("rag")
            needed.append("expert")
        
        # PRIORIDAD 4: Agregar experto para síntesis si hay múltiples fuentes
        if len(needed) > 1 and "expert" not in needed:
            needed.append("expert")
        
        return needed[:3]
    
    def execute_rag(self, query: str, request: Dict[str, Any]) -> Optional[str]:
        """Ejecuta consulta RAG con validación mejorada"""
        result = self.rag.query(query, request)
        
        if result and result.get("success"):
            response = result["response"]
            chunks = result["metadata"].get("chunks_found", 0)
            total = result["metadata"].get("total_chunks", 0)
            
            if chunks > 0:
                return response
            elif total > 0:
                return None
            else:
                return None
        
        return None
    
    def execute_pipeline(self, tool_name: str, query: str) -> Optional[str]:
        """Ejecuta pipeline externo con mejor manejo de errores"""
        tool = self.tools.get(tool_name)
        if not tool or "file" not in tool:
            print(f"[execute_pipeline] Tool {tool_name} no tiene archivo")
            return None
        
        try:
            pipeline_file = tool["file"]
            print(f"[execute_pipeline] Buscando: {pipeline_file}")
            
            # Buscar archivo en múltiples ubicaciones
            search_paths = [
                os.path.dirname(__file__),
                os.getcwd(),
                os.path.join(os.getcwd(), "pipelines"),
                os.path.dirname(os.path.abspath(__file__))
            ]
            
            found_path = None
            for path_base in search_paths:
                # Si pipeline_file ya tiene "pipelines/", no duplicar
                if "pipelines" in pipeline_file:
                    full_path = os.path.join(path_base, os.path.basename(pipeline_file))
                else:
                    full_path = os.path.join(path_base, pipeline_file)
                
                print(f"[execute_pipeline]   Probando: {full_path}")
                if os.path.exists(full_path):
                    found_path = full_path
                    print(f"[execute_pipeline] ✓ Encontrado en: {full_path}")
                    break
            
            if not found_path:
                print(f"[execute_pipeline] Archivo no encontrado: {pipeline_file}")
                print(f"[execute_pipeline]    Buscado en: {search_paths}")
                return None
            
            # Importar módulo
            print(f"[execute_pipeline] Importando módulo...")
            spec = importlib.util.spec_from_file_location(tool_name, found_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Verificar que tenga función pipeline
            if not hasattr(module, 'pipeline'):
                print(f"[execute_pipeline] Módulo sin función 'pipeline'")
                return None
            
            # Ejecutar pipeline
            print(f"[execute_pipeline] Ejecutando pipeline...")
            result = module.pipeline({
                "messages": [{"role": "user", "content": query}]
            })
            
            print(f"[execute_pipeline] Resultado tipo: {type(result)}")
            
            # Extraer respuesta
            if isinstance(result, dict):
                response = result.get("output") or result.get("response") or str(result)
            else:
                response = str(result)
            
            print(f"[execute_pipeline] ✓ Respuesta: {len(response)} caracteres")
            return response
            
        except Exception as e:
            print(f"[execute_pipeline] ERROR en {tool_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def call_expert(self, query: str, context: str = "") -> str:
        """Llama al experto agrónomo"""
        tool = self.tools["expert"]
        
        messages = [
            {"role": "system", "content": tool["system_prompt"]}
        ]
        
        if context:
            messages.append({"role": "system", "content": f"INFORMACIÓN DISPONIBLE:\n{context}"})
        
        messages.append({"role": "user", "content": query})
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": tool["model"],
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"[AgroIA] Error llamando experto: {e}")
        
        return "Error al consultar experto agrónomo"
    
    def synthesize_response(self, query: str, results: Dict[str, str]) -> str:
        """Sintetiza respuestas de múltiples fuentes"""
        
        # Filtrar válidas
        valid = {k: v for k, v in results.items() if v and len(v) > 50}
        
        if not valid:
            if self.rag.enabled:
                try:
                    total_docs = self.rag.rag_system.vector_store.count()
                    if total_docs == 0:
                        return (
                            "Base de documentos vacía\n\n"
                            "Aún no hay documentos técnicos cargados en el sistema RAG.\n\n"
                            "Para consultas técnicas específicas:\n"
                            "- Sube etiquetas de productos (PDF)\n"
                            "- Comparte análisis de suelo\n"
                            "- Adjunta manuales técnicos\n\n"
                            "Mientras tanto, puedo ayudarte con:\n"
                            "- Recomendaciones agronómicas generales\n"
                            "- Consultas sobre clima\n"
                            "- Precios de insumos\n\n"
                            "¿Reformulamos tu consulta?"
                        )
                except:
                    pass
            
            return "No se pudo obtener información suficiente. Intenta reformular tu consulta o sube documentos relevantes."
        
        # Si solo hay una fuente, devolverla directamente
        if len(valid) == 1:
            tool_name, response = list(valid.items())[0]
            tool_desc = self.tools[tool_name]["description"]
            
            # Si es RAG, mantener formato original
            if tool_name == "rag":
                return response
            
            return f"{response}\n\n---\n*Fuente: {tool_desc}*"
        
        # Múltiples fuentes - síntesis inteligente
        synthesis_prompt = f"""Sintetiza la siguiente información para responder la consulta del usuario.

CONSULTA: {query}

INFORMACIÓN RECOPILADA:
"""
        
        for tool_name, response in valid.items():
            tool_desc = self.tools[tool_name]["description"]
            synthesis_prompt += f"\n### {tool_desc}\n{response[:800]}\n"
        
        synthesis_prompt += """\n
INSTRUCCIONES PARA RECOMENDACIÓN AGRÍCOLA INTELIGENTE:

1. **ANÁLISIS TÉCNICO**: Revisa la información del producto (dosis, condiciones, restricciones)
2. **ANÁLISIS CLIMÁTICO**: Evalúa las condiciones meteorológicas (temperatura, lluvia, viento, humedad)
3. **COMPATIBILIDAD**: Determina si las condiciones climáticas son compatibles con los requisitos del producto
4. **RECOMENDACIÓN CLARA**: Da una conclusión definitiva:
   - ✅ "RECOMENDADO: Las condiciones son ideales para la aplicación"
   - ⚠️ "PRECAUCIÓN: Aplicar con cuidado debido a [razón específica]"
   - ❌ "NO RECOMENDADO: Las condiciones no son adecuadas debido a [razón específica]"

5. **FORMATO DE RESPUESTA ESTRUCTURADO**:

**PARA PREGUNTAS SIMPLES**:
**🌿 [PRODUCTO] - [TIPO]**
**🎯 Respuesta Directa**
**📋 Información Técnica**
**💧 Aplicación y Dosis**
**⚠️ Precauciones**

**PARA CONSULTAS DE APLICACIÓN**:
**🌾 Recomendación de Aplicación: [PRODUCTO]**

**📋 Información del Producto**
[Datos técnicos con citaciones]

**🌤️ Condiciones Climáticas**
[Datos meteorológicos específicos]

**🔍 Análisis de Compatibilidad**
[Evaluación técnica]

**🎯 Recomendación Final**
✅/⚠️/❌ [RECOMENDACIÓN CLARA]

**💡 Consejos Prácticos**
[Horarios, calibración, EPP]

**⚠️ Advertencias**
[Seguridad y precauciones]

6. **CONSIDERACIONES CLIMÁTICAS CRÍTICAS**:
   - Temperatura: Evitar aplicación si >25°C o <5°C
   - Lluvia: No aplicar si hay lluvia pronosticada en 6-24h
   - Viento: Evitar si viento >15 km/h
   - Humedad: Considerar para eficacia del producto

Genera una respuesta profesional, práctica y definitiva (máximo 500 palabras):"""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": synthesis_prompt}],
                    "temperature": 0.6,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                final = response.json()['choices'][0]['message']['content']
                
                # Agregar footer con fuentes
                sources = [self.tools[k]["description"] for k in valid.keys()]
                footer = f"\n\n---\n*Fuentes consultadas: {', '.join(sources)}*"
                
                return final + footer
        except Exception as e:
            print(f"[AgroIA] Error en síntesis: {e}")
        
        # Fallback: mejor respuesta individual
        priority = ["rag", "expert", "weather", "prices"]
        for tool in priority:
            if tool in valid:
                return f"{valid[tool]}\n\n---\n*Fuente principal: {self.tools[tool]['description']}*"
        
        return list(valid.values())[0]


def extract_message(request: Dict[str, Any]) -> tuple[str, List[Dict[str, str]]]:
    """Extrae mensaje del usuario Y el historial de conversación"""
    messages_history = []
    current_message = ""
    
    try:
        messages_list = None
        if "body" in request and "messages" in request["body"]:
            messages_list = request["body"]["messages"]
        elif "messages" in request:
            messages_list = request["messages"]
        
        if messages_list and isinstance(messages_list, list):
            for msg in messages_list[-6:-1]:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "").strip()
                    if content and role in ["user", "assistant"]:
                        messages_history.append({
                            "role": role,
                            "content": content[:300]
                        })
            
            if messages_list:
                current_message = messages_list[-1].get("content", "").strip()
        
        if not current_message:
            for key in ["prompt", "query", "input"]:
                if key in request:
                    current_message = str(request[key]).strip()
                    break
    except Exception as e:
        print(f"[Extract] Error: {e}")
    
    return current_message, messages_history


def pipeline(request: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline principal de Super AgroIA Master"""
    
    try:
        query, history = extract_message(request)
        
        if not query:
            return {"output": (
                "Super AgroIA Master v2.2\n\n"
                "Asistente agrícola inteligente con:\n"
                "- RAG sobre documentos técnicos\n"
                "- Clima en tiempo real\n"
                "- Precios de insumos\n"
                "- Experto agrónomo IA\n\n"
                "Ejemplos:\n"
                "- \"¿Cuántas veces aplicar Shark en trigo?\"\n"
                "- \"Clima para aplicar herbicida mañana\"\n"
                "- \"Precio de fertilizantes NPK\"\n\n"
                "¿En qué puedo ayudarte?"
            )}
        
        print(f"\n{'='*60}")
        print(f"[AgroIA Master] CONSULTA: {query[:80]}...")
        if history:
            print(f"[AgroIA Master] Historial: {len(history)} mensajes previos")
        print(f"{'='*60}\n")
        
        agroia = SuperAgroIAMaster()
        
        tools_needed = agroia.analyze_query(query, history)
        print(f"[AgroIA Master] Herramientas detectadas: {tools_needed}")
        
        enriched_query = query
        if history and "rag" in tools_needed:
            for msg in reversed(history):
                if msg["role"] == "user":
                    enriched_query = f"CONTEXTO PREVIO: {msg['content'][:150]}\n\nCONSULTA ACTUAL: {query}"
                    print(f"[AgroIA Master] Query enriquecida con contexto")
                    break
        
        results = {}
        
        for tool in tools_needed:
            print(f"\n[AgroIA Master] Ejecutando: {tool}")
            print(f"{'─'*60}")
            
            if tool == "rag":
                result = agroia.execute_rag(enriched_query, request)
                if result:
                    results[tool] = result
                    print(f"[AgroIA Master] ✓ RAG - {len(result)} caracteres")
                    print(f"    Preview: {result[:100]}...")
                else:
                    print(f"[AgroIA Master] RAG - Sin resultados")
                    
            elif tool == "expert":
                continue
            else:
                result = agroia.execute_pipeline(tool, query)
                if result:
                    results[tool] = result
                    print(f"[AgroIA Master] ✓ {tool} - {len(result)} caracteres")
                else:
                    print(f"[AgroIA Master] {tool} - Sin resultados")
        
        if "expert" in tools_needed:
            print(f"\n[AgroIA Master] Ejecutando: expert")
            print(f"{'─'*60}")
            
            if results:
                context_parts = []
                for k, v in results.items():
                    preview = v[:300] if len(v) > 300 else v
                    context_parts.append(f"{k}: {preview}")
                context = "\n\n".join(context_parts)
                
                results["expert"] = agroia.call_expert(query, context)
                print(f"[AgroIA Master] ✓ expert (con contexto) - {len(results['expert'])} caracteres")
            else:
                results["expert"] = agroia.call_expert(query)
                print(f"[AgroIA Master] ✓ expert (sin contexto) - {len(results['expert'])} caracteres")
        
        print(f"\n[AgroIA Master] Sintetizando respuesta final...")
        print(f"{'─'*60}")
        print(f"Resultados válidos: {list(results.keys())}")
        
        final_response = agroia.synthesize_response(query, results)
        
        print(f"\n[AgroIA Master] ✓ COMPLETADO")
        print(f"{'='*60}")
        print(f"Respuesta: {len(final_response)} caracteres")
        print(f"{'='*60}\n")
        
        return {"output": final_response}
        
    except Exception as e:
        print(f"\n[AgroIA Master] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"output": f"Error: {str(e)}\n\nIntenta reformular tu consulta."}


def main(request):
    return pipeline(request)