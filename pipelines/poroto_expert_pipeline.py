"""
Pipeline: Experto Completo en Porotos
Integra: Conocimiento experto + RAG + OpenAI + Sistema de Clima

Autor: AgroIA Expert System
Versión: 1.0.0 - Especializado en Porotos
"""

import os
import sys
import json
import uuid
import time
import requests
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Importar LLM Router con fallback
try:
    from llm_router import llm_router
    LLM_ROUTER_AVAILABLE = True
except ImportError:
    LLM_ROUTER_AVAILABLE = False
    print("[Experto Porotos] ⚠️  LLM Router no disponible - usando OpenAI directo")

# Añadir path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))

# Metadata del pipeline
metadata = {
    "name": "poroto_expert_complete",
    "description": "Experto completo en porotos: Conocimiento + RAG + Clima + OpenAI con respuestas optimizadas",
    "type": "llm",
    "version": "1.0.0",
    "author": "AgroIA",
    "tags": ["porotos", "leguminosas", "clima", "rag", "pesticidas", "openai-format"]
}


class PorotoExpertCompletePipeline:
    """Pipeline experto completo en porotos con todas las integraciones"""

    def __init__(self):
        self.name = "🫘 Experto Completo en Porotos"

        # Configuración de API
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Inicializar componentes
        self.rag_system = self._init_rag_system()
        self.weather_system = self._init_weather_system()
        self.knowledge_base = self._load_poroto_knowledge()

        print(f"[Experto Porotos] 🫘 Pipeline Completo inicializado")
        print(f"[Experto Porotos] 📚 RAG: {'✅' if self.rag_system else '❌'}")
        print(f"[Experto Porotos] 🌤️  Clima: {'✅' if self.weather_system else '❌'}")
        print(f"[Experto Porotos] 🤖 Modelo: {self.model}")

    def _init_rag_system(self):
        """Inicializa el sistema RAG de porotos"""
        try:
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            from rag_porotos_optimizado import PorotoRAGSystem
            rag = PorotoRAGSystem()
            doc_count = rag.count()
            print(f"[Experto Porotos] RAG Porotos activo con {doc_count} documentos")
            return rag if doc_count > 0 else None
        except Exception as e:
            print(f"[Experto Porotos] RAG no disponible: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _init_weather_system(self):
        """Inicializa el sistema de clima con búsqueda mejorada"""
        try:
            weather_file = "weather_chat_enhanced.py"
            print(f"[Experto Porotos] Buscando: {weather_file}")

            search_paths = [
                os.path.dirname(__file__),
                os.getcwd(),
                os.path.join(os.getcwd(), "pipelines"),
                os.path.dirname(os.path.abspath(__file__))
            ]

            found_path = None
            for path_base in search_paths:
                if "pipelines" in weather_file:
                    full_path = os.path.join(path_base, os.path.basename(weather_file))
                else:
                    full_path = os.path.join(path_base, weather_file)

                print(f"[Experto Porotos]   Probando: {full_path}")
                if os.path.exists(full_path):
                    found_path = full_path
                    print(f"[Experto Porotos] Pipeline clima encontrado: {full_path}")
                    break

            if not found_path:
                print(f"[Experto Porotos] Pipeline clima no encontrado: {weather_file}")
                print(f"[Experto Porotos]    Buscado en: {search_paths}")
                return None

            print(f"[Experto Porotos] Importando módulo clima...")
            spec = importlib.util.spec_from_file_location("weather_chat_enhanced", found_path)
            weather_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(weather_module)

            if not hasattr(weather_module, 'pipeline'):
                print(f"[Experto Porotos] Módulo clima sin función 'pipeline'")
                return None

            print(f"[Experto Porotos] Sistema de clima integrado exitosamente")
            return weather_module.pipeline

        except Exception as e:
            print(f"[Experto Porotos] Error cargando clima: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_poroto_knowledge(self) -> Dict[str, Any]:
        """Carga la base de conocimientos sobre porotos"""
        return {
            "variedades_chile": {
                "alubia": {
                    "nombre": "Alubia-INIA",
                    "tipo": "Poroto blanco",
                    "ciclo": "Intermedio (105-120 días)",
                    "rendimiento": "25-35 qq/ha",
                    "resistencia": "Tolera sequías cortas",
                    "zona": "Región del Maule y Ñuble"
                },
                "tijeral": {
                    "nombre": "Tijeral-INIA",
                    "tipo": "Poroto rojo",
                    "ciclo": "Precoz (95-105 días)",
                    "rendimiento": "22-30 qq/ha",
                    "resistencia": "Resistente a antracnosis",
                    "zona": "Región del Biobío"
                },
                "cargamanto": {
                    "nombre": "Cargamanto",
                    "tipo": "Poroto moteado",
                    "ciclo": "Tardío (120-130 días)",
                    "rendimiento": "20-28 qq/ha",
                    "resistencia": "Moderada tolerancia a sequía",
                    "zona": "Región del Maule"
                }
            },
            "etapas_criticas_clima": {
                "siembra": {
                    "temperatura_optima": "18-24°C",
                    "temperatura_minima": "12°C",
                    "dias": "Octubre-Noviembre (primaveras templadas)",
                    "riesgo_lluvia": "Lluvias intensas generan encostramiento",
                    "riesgo_sequia": "Semilla deshidratada reduce emergencia"
                },
                "emergencia": {
                    "temperatura_optima": "20-25°C",
                    "dias": "10-15 días post siembra",
                    "riesgo_frio": "Temperaturas < 15°C ralentizan crecimiento",
                    "riesgo_plagas": "Trips y pulgones se instalan en tejido joven"
                },
                "floracion": {
                    "temperatura_optima": "22-28°C",
                    "dias": "45-60 días post siembra",
                    "riesgo_calor": "> 32°C provoca aborto floral",
                    "riesgo_lluvia": "Lluvia continua favorece enfermedades foliares"
                },
                "llenado_vaina": {
                    "temperatura_optima": "20-26°C",
                    "dias": "70-90 días post siembra",
                    "riesgo_sequia": "CRÍTICO - Estrés hídrico reduce tamaño de granos",
                    "riesgo_heladas": "Heladas tempranas abortan vainas"
                },
                "maduracion": {
                    "temperatura_optima": "18-24°C",
                    "dias": "90-110 días post siembra",
                    "riesgo_lluvia": "CRÍTICO - Humedad alta genera brotado en vaina",
                    "condicion_ideal": "Período seco para uniformar secado"
                }
            },
            "malezas_criticas": [
                "Rama negra (Conyza bonariensis)",
                "Verdolaga (Portulaca oleracea)",
                "Chufa amarilla (Cyperus esculentus)",
                "Amaranthus spp.",
                "Ballica (Lolium multiflorum)"
            ],
            "plagas_climaticas": {
                "pulgon_verde": {
                    "condiciones": "Climas templados y secos",
                    "periodo_critico": "Floración a llenado de vaina",
                    "sintomas": "Enrrollamiento, melaza y virus"
                },
                "mosca_del_poroto": {
                    "condiciones": "Temperaturas > 25°C con humedad",
                    "periodo_critico": "Pos emergencia",
                    "sintomas": "Galerías en tallo, marchitez"
                },
                "trips": {
                    "condiciones": "Baja humedad relativa",
                    "periodo_critico": "Floración",
                    "sintomas": "Manchas plateadas, aborto floral"
                }
            },
            "enfermedades_climaticas": {
                "antracnosis": {
                    "condiciones": "Lluvia + 18-22°C",
                    "periodo_critico": "Floración a llenado",
                    "sintomas": "Lesiones hundidas en tallos y vainas"
                },
                "roya": {
                    "condiciones": "Humedad alta + 20-25°C",
                    "periodo_critico": "Vegetativo tardío",
                    "sintomas": "Pústulas naranja en hojas"
                },
                "mancha_bacteriana": {
                    "condiciones": "Lluvia + temperatura > 24°C",
                    "periodo_critico": "Todo el ciclo",
                    "sintomas": "Manchas acuosas evoluciona a necrosis"
                }
            }
        }

    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
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

        weather_keywords = [
            "clima", "tiempo", "temperatura", "lluvia", "helada",
            "pronostico", "pronóstico", "meteorologico", "meteorológico",
            "va a llover", "cuándo sembrar", "cuando sembrar",
            "condiciones climáticas", "riesgo de heladas",
            "mañana", "hoy", "aplicar mañana", "puedo aplicar",
            "aplicar hoy", "aplicar en", "condiciones para aplicar",
            "poroto", "porotos", "frijol", "frijoles", "judía", "judías",
            "talca", "linares", "longaví", "chillán", "los ángeles", "temuco",
            "curicó", "parral", "san javier"
        ]

        if any(keyword in query_lower for keyword in weather_keywords):
            intent["needs_weather"] = True
            if intent["query_type"] == "general":
                intent["query_type"] = "weather_related"

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
                "longavi": "Longaví",
                "longaví": "Longaví",
                "san javier": "San Javier",
                "linares": "Linares",
                "quillota": "Quillota",
                "osorno": "Osorno",
                "los angeles": "Los Ángeles",
                "castro": "Castro",
                "coyhaique": "Coyhaique",
                "punta arenas": "Punta Arenas"
            }

            for ciudad_key, ciudad_name in ciudades_chile.items():
                if ciudad_key in query_lower:
                    intent["weather_location"] = ciudad_name
                    break

            if not intent.get("weather_location"):
                import re
                patrones = [
                    r'en\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)',
                    r'para\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)',
                    r'de\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)'
                ]
                for pattern in patrones:
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip()
                        excluded = {'porotos', 'producto', 'aplicar'}
                        if candidate.lower() not in excluded:
                            intent["weather_location"] = candidate.title()
                            print(f"[Experto Porotos] Ciudad detectada: {candidate}")
                            break

            if not intent["weather_location"]:
                regiones = {
                    "maule": "Talca",
                    "ñuble": "Chillán",
                    "biobio": "Los Ángeles",
                    "araucania": "Temuco"
                }
                for region, ciudad in regiones.items():
                    if region in query_lower:
                        intent["weather_location"] = ciudad
                        break

            import re
            if "mañana" in query_lower or "tomorrow" in query_lower:
                intent["weather_days"] = 2
            elif "hoy" in query_lower or "today" in query_lower:
                intent["weather_days"] = 1
            elif "próximos" in query_lower or "proximos" in query_lower:
                dias_match = re.search(r'(\d+)\s*días', query_lower)
                if dias_match:
                    intent["weather_days"] = min(int(dias_match.group(1)), 7)

        rag_keywords = [
            "herbicida", "fungicida", "insecticida", "producto", "dosis", "aplicación",
            "etiqueta", "ficha técnica", "principio activo",
            "malezas", "plagas", "enfermedades", "controlar", "leguminosa",
            "porotos", "frijoles", "judías", "alubias", "antracnosis",
            "trips", "mosca del poroto", "pendimetalin", "trifluralin",
            "bentazon", "fomesafen", "imazetapir", "quizalofop", "clethodim"
        ]

        if any(keyword in query_lower for keyword in rag_keywords):
            intent["needs_rag"] = True
            intent["query_type"] = "technical_product"

        if any(word in query_lower for word in ["variedad", "variedades", "cultivar"]):
            intent["query_type"] = "variety_selection"
        elif any(word in query_lower for word in ["siembra", "sembrar", "germinación"]):
            intent["query_type"] = "planting"
        elif any(word in query_lower for word in ["fertilizar", "fertilización", "abonado"]):
            intent["query_type"] = "fertilization"

        return intent

    def _get_weather_data(self, location: str, days: int = 7) -> Optional[Dict[str, Any]]:
        """Obtiene datos del clima usando el pipeline de clima"""
        if not self.weather_system:
            return None

        try:
            if days == 1:
                query = f"Clima hoy en {location}"
            elif days == 2:
                query = f"Clima mañana en {location}"
            else:
                query = f"Pronóstico del clima para {location} próximos {days} días"

            print(f"[Experto Porotos] 🌤️ Ejecutando pipeline clima: '{query}'")

            result = self.weather_system({
                "messages": [{"role": "user", "content": query}]
            })

            print(f"[Experto Porotos] Resultado clima tipo: {type(result)}")

            if isinstance(result, dict):
                response = result.get("output") or result.get("response") or str(result)
            else:
                response = str(result)

            print(f"[Experto Porotos] Respuesta clima: {len(response)} caracteres")

            if response and len(response) > 50:
                print(f"[Experto Porotos] ✅ Clima obtenido exitosamente")
                return {
                    "raw_response": response,
                    "location": location,
                    "days": days,
                    "success": True
                }
            else:
                print(f"[Experto Porotos] ⚠️ Respuesta clima muy corta o vacía")
                return None

        except Exception as e:
            print(f"[Experto Porotos] Error obteniendo clima: {e}")
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
                "que herbicida", "qué herbicida", "herbicida para porotos",
                "herbicidas para", "control de malezas", "porotos malezas",
                "fungicida porotos", "insecticida porotos"
            ])

            if is_general:
                print(f"[Experto Porotos] 🔍 Consulta general - búsqueda amplia")
                search_queries = [
                    "porotos herbicida preemergente malezas",
                    "porotos postemergente control rama negra",
                    "porotos fungicida antracnosis manejo",
                    "porotos insecticida trips dosis",
                    "leguminosas herbicida pendimetalin dosis"
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
                print(f"[Experto Porotos] ✅ Encontrados {len(results)} documentos únicos")
            else:
                enriched_query = f"porotos cultivo {query}"
                results = self.rag_system.search(enriched_query, top_k=top_k)

            if not results:
                return None

            formatted = ["# INFORMACIÓN TÉCNICA DE PESTICIDAS PARA POROTOS\n"]
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
            print(f"[Experto Porotos] Error en RAG: {e}")
            return None

    def _create_system_prompt(self, query: str, intent: Dict[str, Any], weather_data: Optional[Dict] = None, rag_data: Optional[str] = None) -> str:
        """Crea el prompt del sistema integrando todas las fuentes"""

        prompt = """Eres un **Experto Agrónomo Especializado en Cultivo de Porotos** en Chile.

Tienes acceso a MÚLTIPLES fuentes de información para dar la mejor asesoría:

## 📚 FUENTES DE INFORMACIÓN DISPONIBLES

### 1️⃣ CONOCIMIENTO EXPERTO INTEGRADO
- Variedades de porotos chilenas (Alubia-INIA, Tijeral-INIA, Cargamanto)
- Manejo agronómico completo del cultivo
- Etapas fenológicas y requerimientos hídricos
- Plagas, enfermedades y malezas específicas del cultivo

### 2️⃣ DATOS METEOROLÓGICOS EN TIEMPO REAL
"""

        if weather_data:
            if weather_data.get("error") == "no_access":
                prompt += f"""
⚠️ **CLIMA NO DISPONIBLE PARA: {weather_data.get('location', 'Ubicación')}**

{weather_data.get('message', 'No hay acceso a datos climáticos')}

**INSTRUCCIONES PARA CONSULTAS DE CLIMA:**
- Explica que intentaste obtener datos climáticos pero no están disponibles
- Recomienda verificar el pronóstico local antes de aplicar o regar
- Da recomendaciones generales sobre condiciones ideales
- Menciona factores críticos: lluvia (24h), viento (<10 km/h), temperatura (18-28°C)
"""
            else:
                prompt += f"""
✅ **CLIMA DISPONIBLE PARA: {weather_data.get('location', 'Ubicación')}**

{weather_data.get('raw_response', '')}

**INSTRUCCIONES CRÍTICAS PARA USO DE DATOS CLIMÁTICOS:**
- ANALIZA lluvia, viento y temperatura para las próximas 24 h
- Si hay lluvia intensa → ADVERTIR sobre riesgo de enfermedades y acceso al campo
- Si viento > 10 km/h → ADVERTIR sobre deriva en aplicaciones
- Si temperatura < 15°C o > 32°C → RECOMENDAR ajustar siembras/aplicaciones
- SIEMPRE cita los datos relevantes del pronóstico
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
- Emojis para mejorar legibilidad (✅ ❌ ⚠️ 🫘 📊)

### ETAPAS CRÍTICAS Y CLIMA:

**SIEMBRA** (Primavera): 18-24°C | Riesgo: lluvia excesiva encostra y dificulta emergencia
**EMERGENCIA** (10-15 días): 20-25°C | Trip, pulgones y mosca del poroto amenazan brotes
**FLORACIÓN** (45-60 días): ⚠️ **CRÍTICO** | 22-28°C | Calor > 32°C provoca aborto floral
**LLENADO DE VAINA** (70-90 días): ⚠️ **CRÍTICO** | 20-26°C | Sequía reduce tamaño de granos
**MADURACIÓN** (90-110 días): Período seco ideal | Lluvia = riesgo de brotado y antracnosis

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
            messages = body.get("messages", [])
            if not messages:
                return self._create_error_response("No se recibieron mensajes")

            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

            if not user_message:
                return self._create_error_response("No se encontró mensaje del usuario")

            print(f"[Experto Porotos] 📝 Consulta: {user_message[:100]}...")

            intent = self._detect_query_intent(user_message)
            print(f"[Experto Porotos] 🎯 Tipo: {intent['query_type']}")
            print(f"[Experto Porotos] 🌤️  Clima: {intent['needs_weather']}")
            print(f"[Experto Porotos] 📚 RAG: {intent['needs_rag']}")

            weather_data = None
            if intent["needs_weather"]:
                location = intent.get("weather_location", "Talca")
                days = intent.get("weather_days", 7)
                print(f"[Experto Porotos] 🌍 Obteniendo clima: {location}, {days} días")

                if self.weather_system:
                    weather_data = self._get_weather_data(location, days)
                    if weather_data:
                        print(f"[Experto Porotos] ✅ Clima obtenido")
                    else:
                        print(f"[Experto Porotos] ⚠️  No se pudo obtener clima")
                else:
                    weather_data = {
                        "error": "no_access",
                        "location": location,
                        "message": f"No tengo acceso a datos climáticos para {location}"
                    }

            rag_data = None
            if intent["needs_rag"] and self.rag_system:
                print(f"[Experto Porotos] 📖 Buscando en documentos...")
                rag_data = self._get_rag_data(user_message, top_k=15)
                if rag_data:
                    print(f"[Experto Porotos] ✅ Documentos encontrados")
                else:
                    print(f"[Experto Porotos] ⚠️  No se encontraron documentos")

            system_prompt = self._create_system_prompt(
                user_message,
                intent,
                weather_data,
                rag_data
            )

            llm_messages = [
                {"role": "system", "content": system_prompt}
            ]

            for msg in messages[-8:]:
                if msg.get("role") in ["user", "assistant"]:
                    llm_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            if LLM_ROUTER_AVAILABLE:
                print(f"[Experto Porotos] 🤖 Usando LLM Router con fallback")
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
                        print(f"[Experto Porotos] 🔄 Fallback usado: {model_used}")
                    else:
                        print(f"[Experto Porotos] ✅ Modelo principal: {model_used}")
                else:
                    return self._create_error_response("Sistema de IA temporalmente no disponible")
            else:
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
                    print(f"[Experto Porotos] {error_msg}")
                    return self._create_error_response(error_msg)

            footer = "\n\n---\n"
            sources = []

            if weather_data:
                sources.append("🌤️ Clima en tiempo real")
            if rag_data:
                sources.append("📚 Documentos técnicos")
            sources.append("🧠 Conocimiento experto")

            footer += f"**Fuentes**: {' + '.join(sources)}\n"
            footer += "🫘 *Experto Completo en Porotos - AgroIA v1.0.0*"

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
            print(f"[Experto Porotos] {error_msg}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(error_msg)

    def _create_success_response(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Crea respuesta exitosa en formato OpenAI"""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "poroto_expert_complete",
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
            "model": "poroto_expert_complete",
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
        self.pipeline = PorotoExpertCompletePipeline()
        self.name = "Experto Completo en Porotos"
        self.id = "poroto_expert_complete"

        self.type = "llm"
        self.description = "Experto completo en porotos con respuestas optimizadas en formato OpenAI"
        self.version = "1.0.0"

    def pipe(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Devuelve directamente formato OpenAI"""
        result = self.pipeline.pipe(body)
        return result

    def __call__(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipe(body)


pipeline = Pipeline()
