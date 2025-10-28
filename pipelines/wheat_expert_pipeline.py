"""
Pipeline: Experto Completo en Trigo
Integra: Conocimiento experto + RAG + OpenAI + Sistema de Clima

Autor: AgroIA Expert System
Versión: 1.0.0 - Especializado en Trigo
"""

import os
import sys
import json
import uuid
import time
import requests
import importlib.util
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# Importar LLM Router con fallback
try:
    from llm_router import llm_router
    LLM_ROUTER_AVAILABLE = True
except ImportError:
    LLM_ROUTER_AVAILABLE = False
    print("[Experto Trigo] ⚠️  LLM Router no disponible - usando OpenAI directo")

# Añadir path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))

# Metadata del pipeline
metadata = {
    "name": "wheat_expert_complete",
    "description": "Experto completo en trigo: Conocimiento + RAG + Clima + OpenAI con respuestas optimizadas",
    "type": "llm",
    "version": "1.0.0",
    "author": "AgroIA",
    "tags": ["trigo", "clima", "rag", "pesticidas", "completo", "openai-format"]
}


class WheatExpertCompletePipeline:
    """Pipeline experto completo en trigo con todas las integraciones"""
    
    def __init__(self):
        self.name = "🌾 Experto Completo en Trigo"
        
        # Configuración de API
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        # Inicializar componentes
        self.rag_system = self._init_rag_system()
        self.weather_system = self._init_weather_system()
        self.knowledge_base = self._load_wheat_knowledge()
        
        print(f"[Experto Trigo] 🌾 Pipeline Completo inicializado")
        print(f"[Experto Trigo] 📚 RAG: {'✅' if self.rag_system else '❌'}")
        print(f"[Experto Trigo] 🌤️  Clima: {'✅' if self.weather_system else '❌'}")
        print(f"[Experto Trigo] 🤖 Modelo: {self.model}")
    
    def _init_rag_system(self):
        """Inicializa el sistema RAG de trigo optimizado"""
        try:
            import sys
            from pathlib import Path
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from rag_trigo_optimizado import WheatRAGSystem
            rag = WheatRAGSystem()
            doc_count = rag.count()
            print(f"[Experto Trigo] RAG Trigo activo con {doc_count} documentos")
            return rag if doc_count > 0 else None
        except Exception as e:
            print(f"[Experto Trigo] RAG no disponible: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _init_weather_system(self):
        """Inicializa el sistema de clima con búsqueda mejorada"""
        try:
            import importlib.util
            
            weather_file = "weather_chat_enhanced.py"
            print(f"[Experto Trigo] Buscando: {weather_file}")
            
            # Buscar archivo en múltiples ubicaciones
            search_paths = [
                os.path.dirname(__file__),
                os.getcwd(),
                os.path.join(os.getcwd(), "pipelines"),
                os.path.dirname(os.path.abspath(__file__))
            ]
            
            found_path = None
            for path_base in search_paths:
                # Si weather_file ya tiene "pipelines/", no duplicar
                if "pipelines" in weather_file:
                    full_path = os.path.join(path_base, os.path.basename(weather_file))
                else:
                    full_path = os.path.join(path_base, weather_file)
                
                print(f"[Experto Trigo]   Probando: {full_path}")
                if os.path.exists(full_path):
                    found_path = full_path
                    print(f"[Experto Trigo] Pipeline clima encontrado: {full_path}")
                    break
            
            if not found_path:
                print(f"[Experto Trigo] Pipeline clima no encontrado: {weather_file}")
                print(f"[Experto Trigo]    Buscado en: {search_paths}")
                return None
            
            # Importar módulo
            print(f"[Experto Trigo] Importando módulo clima...")
            spec = importlib.util.spec_from_file_location("weather_chat_enhanced", found_path)
            weather_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(weather_module)
            
            # Verificar que tenga función pipeline
            if not hasattr(weather_module, 'pipeline'):
                print(f"[Experto Trigo] Módulo clima sin función 'pipeline'")
                return None
            
            print(f"[Experto Trigo] Sistema de clima integrado exitosamente")
            return weather_module.pipeline
                
        except Exception as e:
            print(f"[Experto Trigo] Error cargando clima: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_wheat_knowledge(self) -> Dict[str, Any]:
        """Carga la base de conocimientos sobre trigo"""
        return {
            "variedades_chile": {
                "pandora": {
                    "nombre": "Pandora-INIA",
                    "tipo": "Trigo Harinero",
                    "ciclo": "Intermedio (140-160 días)",
                    "rendimiento": "70-90 qq/ha",
                    "resistencia": "Resistente a roya amarilla y parda",
                    "zona": "Zonas Centro-Sur"
                },
                "maxwell": {
                    "nombre": "Maxwell",
                    "tipo": "Trigo Harinero",
                    "ciclo": "Precoz (130-140 días)",
                    "rendimiento": "65-85 qq/ha",
                    "tolerancia": "Tolerante a sequía",
                    "zona": "Zona Central y Sur"
                }
            },
            "etapas_criticas_clima": {
                "siembra": {
                    "temperatura_optima": "10-15°C",
                    "temperatura_minima": "3°C",
                    "dias": "Mayo-Julio",
                    "riesgo_lluvia": "Exceso de lluvia impide siembra",
                    "riesgo_sequia": "Falta humedad reduce germinación"
                },
                "macollaje": {
                    "temperatura_optima": "12-18°C",
                    "dias": "30-60 días post-siembra",
                    "riesgo_heladas": "Heladas severas < -5°C dañan macollos",
                    "riesgo_sequia": "Sequía reduce número de macollos"
                },
                "encañazon": {
                    "temperatura_optima": "15-20°C",
                    "dias": "60-90 días post-siembra",
                    "riesgo_heladas": "CRÍTICO - Heladas < -2°C dañan espiga",
                    "riesgo_lluvia": "Lluvia excesiva favorece enfermedades"
                },
                "floracion": {
                    "temperatura_optima": "18-25°C",
                    "dias": "90-110 días post-siembra",
                    "riesgo_heladas": "CRÍTICO - Heladas causan esterilidad",
                    "temperatura_critica": "-1°C",
                    "riesgo_lluvia": "Lluvia dificulta polinización"
                },
                "llenado_grano": {
                    "temperatura_optima": "20-25°C",
                    "dias": "110-140 días post-siembra",
                    "riesgo_sequia": "CRÍTICO - Reduce peso de mil granos",
                    "riesgo_calor": "Temperaturas > 32°C aceleran maduración",
                    "riesgo_lluvia": "Favorece fusarium y otras enfermedades"
                },
                "maduracion": {
                    "temperatura_optima": "22-28°C",
                    "dias": "140-160 días post-siembra",
                    "riesgo_lluvia": "CRÍTICO - Causa brotado y pérdida calidad",
                    "condicion_ideal": "Período seco para cosecha"
                }
            },
            "malezas_criticas": [
                "Avena guacha (Avena fatua)",
                "Ballica (Lolium rigidum)",
                "Alpiste (Phalaris minor)",
                "Avenilla (Avena sterilis)",
                "Bromo (Bromus rigidus)",
                "Rábano (Raphanus raphanistrum)"
            ],
            "enfermedades_climaticas": {
                "roya_amarilla": {
                    "condiciones": "Humedad alta + 10-15°C",
                    "periodo_critico": "Macollaje a encañazón",
                    "sintomas": "Pústulas amarillas en hojas"
                },
                "roya_parda": {
                    "condiciones": "Humedad alta + 15-25°C",
                    "periodo_critico": "Encañazón a floración",
                    "sintomas": "Pústulas café-rojizo"
                },
                "septoria": {
                    "condiciones": "Lluvia persistente + 15-25°C",
                    "periodo_critico": "Todo el ciclo",
                    "sintomas": "Manchas con picnidios negros"
                },
                "fusarium": {
                    "condiciones": "Humedad alta + 25-30°C",
                    "periodo_critico": "Floración a llenado",
                    "sintomas": "Espigas blanqueadas"
                }
            }
        }
    
    def _detect_query_intent(self, query: str) -> Dict[str, bool]:
        """Detecta qué sistemas necesita usar según la consulta"""
        query_lower = query.lower()
        
        intent = {
            "needs_weather": False,
            "needs_rag": False,
            "needs_expert": True,
            "weather_location": None,
            "weather_days": None,
            "query_type": "general"
        }
        
        # Detectar necesidad de clima
        weather_keywords = [
            "clima", "tiempo", "temperatura", "lluvia", "helada",
            "pronostico", "pronóstico", "meteorologico", "meteorológico",
            "va a llover", "cuándo sembrar", "cuando sembrar",
            "condiciones climáticas", "riesgo de heladas",
            "mañana", "hoy", "aplicar mañana", "puedo aplicar",
            "aplicar hoy", "aplicar en", "condiciones para aplicar",
            "chillan", "santiago", "valparaiso", "concepcion", "temuco", "valdivia", "osorno", 
            "puerto montt", "iquique", "antofagasta", "la serena", "rancagua", "talca", "curico",
            "linares", "chillán", "los angeles", "angol", "villarrica", "pucón", "castro",
            "coyhaique", "punta arenas", "arica", "calama", "copiapó", "ovalle", "quillota"
        ]
        
        if any(keyword in query_lower for keyword in weather_keywords):
            intent["needs_weather"] = True
            # Solo cambiar query_type si no es ya technical_product (RAG)
            if intent["query_type"] == "general":
                intent["query_type"] = "weather_related"
            
            # Detectar ubicación (ciudades principales de Chile)
            ciudades_chile = {
                "santiago": "Santiago",
                "valparaiso": "Valparaíso",
                "concepcion": "Concepción",
                "concepción": "Concepción",
                "la serena": "La Serena",
                "antofagasta": "Antofagasta",
                "temuco": "Temuco",
                "rancagua": "Rancagua",
                "talca": "Talca",
                "arica": "Arica",
                "chillan": "Chillán",
                "chillán": "Chillán",
                "curico": "Curicó",
                "valdivia": "Valdivia",
                "puerto montt": "Puerto Montt",
                "iquique": "Iquique",
                "coquimbo": "Coquimbo",
                "osorno": "Osorno",
                "quillota": "Quillota",
                "calama": "Calama",
                "linares": "Linares",
                "los angeles": "Los Ángeles",
                "copiapó": "Copiapó",
                "ovalle": "Ovalle",
                "angol": "Angol",
                "villarrica": "Villarrica",
                "pucón": "Pucón",
                "castro": "Castro",
                "coyhaique": "Coyhaique",
                "punta arenas": "Punta Arenas",
                "longavi": "Longaví",
                "longaví": "Longaví",
                "cobquecura": "Cobquecura",
                "quirihue": "Quirihue",
                "viña del mar": "Viña del Mar",
                "vina del mar": "Viña del Mar"
            }
            
            for ciudad_key, ciudad_name in ciudades_chile.items():
                if ciudad_key in query_lower:
                    intent["weather_location"] = ciudad_name
                    break
            
            # Detectar período temporal (hoy, mañana, próximos días)
            import re
            if "mañana" in query_lower or "tomorrow" in query_lower:
                intent["weather_days"] = 2
            elif "hoy" in query_lower or "today" in query_lower:
                intent["weather_days"] = 1
            elif "proximos" in query_lower or "próximos" in query_lower:
                dias_match = re.search(r'(\d+)\s*días', query_lower)
                if dias_match:
                    intent["weather_days"] = min(int(dias_match.group(1)), 7)
            # Si no se especifica, usar default de 7 días (se maneja en el pipeline)
        
        # Detectar necesidad de RAG
        rag_keywords = [
            "herbicida", "fungicida", "insecticida", "producto", "dosis", "aplicación",
            "etiqueta", "ficha técnica", "principio activo",
            "controlar maleza", "eliminar", "precauciones",
            "malezas", "enfermedades", "plagas",
            "glifosato", "2,4-d", "mcpa", "dicamba", "metsulfuron",
            # Nombres comerciales comunes
            "ajax", "portento", "ally", "hussar", "axial", "broadway", "atlantis",
            "traxos", "refinar", "affinity", "aramo", "topik", "puma", "cossack"
        ]
        
        if any(keyword in query_lower for keyword in rag_keywords):
            intent["needs_rag"] = True
            intent["query_type"] = "technical_product"
        
        return intent
    
    def _get_weather_data(self, location: str, days: int = 7) -> Optional[Dict[str, Any]]:
        """Obtiene datos del clima usando el pipeline de clima con manejo mejorado"""
        if not self.weather_system:
            return None
        
        try:
            # Construir consulta según el número de días
            if days == 1:
                query = f"Clima hoy en {location}"
            elif days == 2:
                query = f"Clima mañana en {location}"
            else:
                query = f"Pronóstico del clima para {location} próximos {days} días"
            
            print(f"[Experto Trigo] 🌤️ Ejecutando pipeline clima: '{query}'")
            
            # Ejecutar pipeline con formato estándar
            result = self.weather_system({
                "messages": [{"role": "user", "content": query}]
            })
            
            print(f"[Experto Trigo] Resultado clima tipo: {type(result)}")
            
            # Extraer respuesta del resultado
            if isinstance(result, dict):
                response = result.get("output") or result.get("response") or str(result)
            else:
                response = str(result)
            
            print(f"[Experto Trigo] Respuesta clima: {len(response)} caracteres")
            
            if response and len(response) > 50:
                print(f"[Experto Trigo] ✅ Clima obtenido exitosamente")
                return {
                    "raw_response": response,
                    "location": location,
                    "days": days,
                    "success": True
                }
            else:
                print(f"[Experto Trigo] ⚠️ Respuesta clima muy corta o vacía")
                return None
        
        except Exception as e:
            print(f"[Experto Trigo] Error obteniendo clima: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_rag_data(self, query: str, top_k: int = 15) -> Optional[str]:
        """Obtiene información de documentos RAG con búsqueda inteligente"""
        if not self.rag_system:
            return None
        
        try:
            query_lower = query.lower()
            is_general = any(indicator in query_lower for indicator in [
                "que herbicida", "qué herbicida", "herbicida para trigo",
                "herbicidas para", "control de malezas", "herbicida usar",
                "fungicida para", "insecticida para"
            ])
            
            if is_general:
                print(f"[Experto Trigo] 🔍 Consulta general - búsqueda amplia")
                
                search_queries = [
                    "trigo herbicida preemergente malezas",
                    "cereales herbicida postemergente control",
                    "trigo fungicida roya septoria",
                    "trigo insecticida pulgón trips",
                    "herbicida selectivo trigo aplicación"
                ]
                
                all_results = []
                for sq in search_queries:
                    results = self.rag_system.search(sq, top_k=20)
                    if results:
                        all_results.extend(results)
                
                seen_items = set()
                unique_results = []
                
                for result in all_results:
                    text = result.get('text', '')
                    source = result.get('metadata', {}).get('source_file', '')
                    unique_key = f"{source}_{text[:100]}"
                    
                    if unique_key not in seen_items:
                        seen_items.add(unique_key)
                        unique_results.append(result)
                        
                        if len(unique_results) >= 25:
                            break
                
                results = unique_results
                print(f"[Experto Trigo] ✅ Encontrados {len(results)} documentos únicos")
            
            else:
                enriched_query = f"trigo cereales {query}"
                results = self.rag_system.search(enriched_query, top_k=top_k)
            
            if not results:
                return None
            
            formatted = [f"# INFORMACIÓN TÉCNICA DE PESTICIDAS PARA TRIGO\n"]
            formatted.append(f"Se encontraron {len(results)} productos/documentos relevantes:\n")
            
            for i, result in enumerate(results, 1):
                text = result.get('text', '')
                metadata = result.get('metadata', {})
                source = metadata.get('source_file', 'Documento')
                similarity = result.get('similarity', 0)
                
                product_name = source.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
                
                formatted.append(f"\n## Producto {i}: {product_name}")
                formatted.append(f"Fuente: {source}")
                formatted.append(f"Relevancia: {similarity:.2%}\n")
                formatted.append(f"{text}\n")
                formatted.append("---\n")
            
            return "\n".join(formatted)
            
        except Exception as e:
            print(f"[Experto Trigo] Error en RAG: {e}")
            return None
    
    def _create_system_prompt(self, query: str, intent: Dict, weather_data: Optional[Dict] = None, rag_data: Optional[str] = None) -> str:
        """Crea el prompt del sistema integrando todas las fuentes"""
        
        prompt = """Eres un **Experto Agrónomo Especializado en Cultivo de Trigo** en Chile.

Tienes acceso a MÚLTIPLES fuentes de información para dar la mejor asesoría:

## 📚 FUENTES DE INFORMACIÓN DISPONIBLES

### 1️⃣ CONOCIMIENTO EXPERTO INTEGRADO
- Variedades de trigo chilenas (Pandora-INIA, Maxwell)
- Manejo agronómico completo del cultivo
- Etapas fenológicas y requerimientos
- Plagas, enfermedades y malezas específicas de trigo

### 2️⃣ DATOS METEOROLÓGICOS EN TIEMPO REAL
"""
        
        if weather_data:
            if weather_data.get("error") == "no_access":
                prompt += f"""
⚠️ **CLIMA NO DISPONIBLE PARA: {weather_data.get('location', 'Ubicación')}**

{weather_data.get('message', 'No hay acceso a datos climáticos')}

**INSTRUCCIONES PARA CONSULTAS DE CLIMA:**
- Explica que intentaste obtener datos climáticos pero no están disponibles
- Recomienda verificar el pronóstico local antes de aplicar
- Da recomendaciones generales sobre condiciones ideales
- Menciona factores críticos: lluvia (24h), viento (<10 km/h), temperatura (15-25°C)
"""
            else:
                prompt += f"""
✅ **CLIMA DISPONIBLE PARA: {weather_data.get('location', 'Ubicación')}**

{weather_data.get('raw_response', '')}

**INSTRUCCIONES CRÍTICAS PARA USO DE DATOS CLIMÁTICOS:**
- ANALIZA específicamente: lluvia, viento, temperatura para mañana
- Si hay lluvia en 24h → NO RECOMENDAR aplicación
- Si viento > 10 km/h → ADVERTIR sobre deriva
- Si temperatura < 5°C o > 30°C → RECOMENDAR esperar
- Si condiciones favorables → DAR LUZ VERDE con horarios específicos
- SIEMPRE menciona las condiciones del pronóstico
"""
        else:
            prompt += "\n❌ No hay datos climáticos disponibles\n"
        
        prompt += "\n### 3️⃣ DOCUMENTOS TÉCNICOS (Etiquetas, Fichas)\n"
        
        if rag_data:
            prompt += f"\n✅ **DOCUMENTOS ENCONTRADOS**:\n\n{rag_data}\n"
        else:
            prompt += "\n❌ No hay documentos técnicos relevantes\n"
        
        prompt += """

## 🎯 INSTRUCCIONES PARA RESPONDER

### FORMATO DE RESPUESTA:

**SIEMPRE usa Markdown bien estructurado:**
- Títulos con `##` para secciones principales
- Subtítulos con `###` para subsecciones
- Tablas Markdown para comparativas
- Listas con `-` o números
- **Negritas** para términos importantes
- Emojis para mejorar legibilidad (✅ ❌ ⚠️ 🌾 📊)

### ETAPAS CRÍTICAS Y CLIMA:

**SIEMBRA** (Mayo-Julio): 10-15°C óptimo | Riesgo: lluvia excesiva impide siembra
**MACOLLAJE** (30-60 días): 12-18°C | Heladas < -5°C dañan macollos
**ENCAÑAZÓN** (60-90 días): ⚠️ **CRÍTICO** | 15-20°C | Heladas < -2°C dañan espiga
**FLORACIÓN** (90-110 días): ⚠️ **MUY CRÍTICO** | 18-25°C | Heladas < -1°C = ESTERILIDAD
**LLENADO** (110-140 días): 20-25°C | Sequía = reduce peso granos | Calor > 32°C acelera maduración
**MADURACIÓN** (140-160 días): Período seco ideal | Lluvia = brotado y pérdida calidad

Si el pronóstico muestra riesgo para la etapa actual:
**ALERTAR AL AGRICULTOR y dar medidas preventivas inmediatas**

**FORMATO OBLIGATORIO - RESPUESTA COMPLETA Y ORDENADA:**

1. INCLUIR TODA LA INFORMACIÓN TÉCNICA DISPONIBLE
2. USAR ESPACIOS ENTRE PALABRAS - NUNCA JUNTAR
3. FORMATO: **N. NOMBRE** + Dosis + Momento + P.Activo + Observaciones
4. LÍNEA VACÍA ENTRE CADA PRODUCTO
5. INCLUIR SECCIÓN DE RECOMENDACIONES
6. INCLUIR CONDICIONES CLIMÁTICAS
7. INCLUIR PRECAUCIONES DE SEGURIDAD
8. MÁXIMO 35 LÍNEAS PARA RESPUESTA COMPLETA Y DETALLADA

Ahora responde la consulta de forma profesional, visual y práctica."""
        
        return prompt
    
    def pipe(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa la consulta y devuelve formato OpenAI"""
        
        try:
            # Extraer mensajes
            messages = body.get("messages", [])
            if not messages:
                return self._create_error_response("No se recibieron mensajes")
            
            # Obtener última consulta
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                return self._create_error_response("No se encontró mensaje del usuario")
            
            print(f"[Experto Trigo] 📝 Consulta: {user_message[:100]}...")
            
            # Detectar intención
            intent = self._detect_query_intent(user_message)
            print(f"[Experto Trigo] 🎯 Tipo: {intent['query_type']}")
            print(f"[Experto Trigo] 🌤️  Clima: {intent['needs_weather']}")
            print(f"[Experto Trigo] 📚 RAG: {intent['needs_rag']}")
            
            # Obtener datos de clima
            weather_data = None
            if intent["needs_weather"]:
                location = intent.get("weather_location", "Santiago")
                days = intent.get("weather_days", 7)
                
                print(f"[Experto Trigo] 🌍 Obteniendo clima: {location}, {days} días")
                
                if self.weather_system:
                    weather_data = self._get_weather_data(location, days)
                    if weather_data:
                        print(f"[Experto Trigo] ✅ Clima obtenido")
                    else:
                        print(f"[Experto Trigo] ⚠️  No se pudo obtener clima")
                else:
                    weather_data = {
                        "error": "no_access",
                        "location": location,
                        "message": f"No tengo acceso a datos climáticos para {location}"
                    }
            
            # Obtener datos RAG
            rag_data = None
            if intent["needs_rag"] and self.rag_system:
                print(f"[Experto Trigo] 📖 Buscando en documentos...")
                rag_data = self._get_rag_data(user_message, top_k=15)
                
                if rag_data:
                    print(f"[Experto Trigo] ✅ Documentos encontrados")
                else:
                    print(f"[Experto Trigo] ⚠️  No se encontraron documentos")
            
            # Crear system prompt
            system_prompt = self._create_system_prompt(
                user_message,
                intent,
                weather_data,
                rag_data
            )
            
            # Preparar mensajes para LLM
            llm_messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Añadir historial
            for msg in messages[-8:]:
                if msg.get("role") in ["user", "assistant"]:
                    llm_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Llamar a LLM con fallback automático
            if LLM_ROUTER_AVAILABLE:
                print(f"[Experto Trigo] 🤖 Usando LLM Router con fallback")
                llm_result = llm_router.generate(
                    messages=llm_messages,
                    temperature=0.1,
                    max_tokens=1800,
                    timeout=60
                )
                
                if llm_result["success"]:
                    answer = llm_result["content"]
                    model_used = llm_result["model_used"]
                    is_fallback = llm_result["is_fallback"]
                    
                    if is_fallback:
                        print(f"[Experto Trigo] 🔄 Fallback usado: {model_used}")
                    else:
                        print(f"[Experto Trigo] ✅ Modelo principal: {model_used}")
                else:
                    return self._create_error_response("Sistema de IA temporalmente no disponible")
            else:
                # Fallback a OpenAI directo si no hay router
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": llm_messages,
                        "temperature": 0.1,
                        "max_tokens": 1800
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    model_used = self.model
                    is_fallback = False
                else:
                    error_msg = f"Error en API OpenAI: {response.status_code}"
                    print(f"[Experto Trigo] {error_msg}")
                    return self._create_error_response(error_msg)
            
            # Footer informativo (común para ambos casos)
            footer = "\n\n---\n"
            sources = []
            
            if weather_data:
                sources.append("🌤️ Clima en tiempo real")
            if rag_data:
                sources.append("📚 Documentos técnicos")
            sources.append("🧠 Conocimiento experto")
            
            footer += f"**Fuentes**: {' + '.join(sources)}\n"
            footer += "🌾 *Experto Completo en Trigo - AgroIA v1.0.0*"
            
            return self._create_success_response(
                answer + footer,
                {
                    "used_weather": weather_data is not None,
                    "used_rag": rag_data is not None,
                    "query_type": intent["query_type"],
                    "model": model_used,
                    "is_fallback": is_fallback,
                    "version": "1.0.0"
                }
            )
        
        except Exception as e:
            error_msg = f"Error procesando consulta: {str(e)}"
            print(f"[Experto Trigo] {error_msg}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(error_msg)
    
    def _create_success_response(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Crea respuesta exitosa en formato OpenAI"""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "wheat_expert_complete",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "metadata": metadata or {}
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Crea respuesta de error en formato OpenAI"""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "wheat_expert_complete",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"❌ **Error**: {error_message}\n\nPor favor, intenta de nuevo o reformula tu pregunta."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


class Pipeline:
    def __init__(self):
        self.pipeline = WheatExpertCompletePipeline()
        self.name = "Experto Completo en Trigo"
        self.id = "wheat_expert_complete"
        
        # Metadata para OpenWebUI
        self.type = "llm"
        self.description = "Experto completo en trigo con respuestas optimizadas en formato OpenAI"
        self.version = "1.0.0"
    
    def pipe(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Devuelve directamente formato OpenAI"""
        result = self.pipeline.pipe(body)
        return result
    
    def __call__(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipe(body)


# Instancia global
pipeline = Pipeline()
