"""
Pipeline: Experto Completo en Arroz
Integra: Conocimiento experto + RAG + OpenAI + Sistema de Clima

Autor: AgroIA Expert System
Versión: 1.0.0 - Especializado en Arroz
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
    print("[Experto Arroz] ⚠️  LLM Router no disponible - usando OpenAI directo")


# Añadir path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))


# Metadata del pipeline
metadata = {
    "name": "rice_expert_complete",
    "description": "Experto completo en arroz: Conocimiento + RAG + Clima + OpenAI con respuestas optimizadas",
    "type": "llm",
    "version": "1.0.0",
    "author": "AgroIA",
    "tags": ["arroz", "clima", "rag", "pesticidas", "completo", "openai-format"]
}


class RiceExpertCompletePipeline:
    """Pipeline experto completo en arroz con todas las integraciones"""

    def __init__(self):
        self.name = "🍚 Experto Completo en Arroz"

        # Configuración de API
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Inicializar componentes
        self.rag_system = self._init_rag_system()
        self.weather_system = self._init_weather_system()
        self.knowledge_base = self._load_rice_knowledge()

        print(f"[Experto Arroz] 🍚 Pipeline Completo inicializado")
        print(f"[Experto Arroz] 📚 RAG: {'✅' if self.rag_system else '❌'}")
        print(f"[Experto Arroz] 🌤️  Clima: {'✅' if self.weather_system else '❌'}")
        print(f"[Experto Arroz] 🤖 Modelo: {self.model}")

    def _init_rag_system(self):
        """Inicializa el sistema RAG de arroz optimizado"""
        try:
            import sys
            from pathlib import Path
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            from rag_arroz_optimizado import RiceRAGSystem
            rag = RiceRAGSystem()
            doc_count = rag.count()
            print(f"[Experto Arroz] RAG Arroz activo con {doc_count} documentos")
            return rag if doc_count > 0 else None
        except Exception as e:
            print(f"[Experto Arroz] RAG no disponible: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _init_weather_system(self):
        """Inicializa el sistema de clima con búsqueda mejorada"""
        try:
            import importlib.util

            weather_file = "weather_chat_enhanced.py"
            print(f"[Experto Arroz] Buscando: {weather_file}")

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

                print(f"[Experto Arroz]   Probando: {full_path}")
                if os.path.exists(full_path):
                    found_path = full_path
                    print(f"[Experto Arroz] Pipeline clima encontrado: {full_path}")
                    break

            if not found_path:
                print(f"[Experto Arroz] Pipeline clima no encontrado: {weather_file}")
                print(f"[Experto Arroz]    Buscado en: {search_paths}")
                return None

            print(f"[Experto Arroz] Importando módulo clima...")
            spec = importlib.util.spec_from_file_location("weather_chat_enhanced", found_path)
            weather_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(weather_module)

            if not hasattr(weather_module, 'pipeline'):
                print(f"[Experto Arroz] Módulo clima sin función 'pipeline'")
                return None

            print(f"[Experto Arroz] Sistema de clima integrado exitosamente")
            return weather_module.pipeline

        except Exception as e:
            print(f"[Experto Arroz] Error cargando clima: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_rice_knowledge(self) -> Dict[str, Any]:
        """Carga la base de conocimientos sobre arroz"""
        return {
            "variedades_chile": {
                "tucapel": {
                    "nombre": "Tucapel-INIA",
                    "tipo": "Arroz de grano largo",
                    "ciclo": "Intermedio (150-160 días)",
                    "rendimiento": "80-95 qq/ha",
                    "resistencia": "Tolerante a frío en macollaje",
                    "zona": "Región del Maule y Ñuble"
                },
                "quella": {
                    "nombre": "Quellón-INIA",
                    "tipo": "Arroz de grano largo",
                    "ciclo": "Tardío (165-175 días)",
                    "rendimiento": "85-100 qq/ha",
                    "resistencia": "Buena tolerancia a piricularia",
                    "zona": "Región del Biobío"
                },
                "zafiro": {
                    "nombre": "Zafiro-INIA",
                    "tipo": "Arroz prémium",
                    "ciclo": "Precoz (140-150 días)",
                    "rendimiento": "75-90 qq/ha",
                    "resistencia": "Resistente a heladas de primavera",
                    "zona": "Región del Maule"
                }
            },
            "etapas_criticas_clima": {
                "siembra": {
                    "temperatura_optima": "18-25°C",
                    "temperatura_minima": "12°C",
                    "dias": "Octubre-Noviembre",
                    "riesgo_lluvia": "Lluvia intensa provoca desbordes en tablones",
                    "riesgo_sequia": "Falta de agua impide inundación inicial"
                },
                "macollaje": {
                    "temperatura_optima": "20-28°C",
                    "dias": "30-60 días post-siembra",
                    "riesgo_frio": "Temperaturas < 15°C reducen macollos",
                    "riesgo_plagas": "Mosquito del arroz prolifera con agua estancada"
                },
                "primordio": {
                    "temperatura_optima": "22-30°C",
                    "dias": "70-90 días post-siembra",
                    "riesgo_frio": "CRÍTICO - Frío < 17°C causa esterilidad",
                    "riesgo_heladas": "Heladas leves dañan primordios",
                    "riesgo_viento": "Viento fuerte genera acame"
                },
                "espigamiento": {
                    "temperatura_optima": "24-32°C",
                    "dias": "90-110 días post-siembra",
                    "riesgo_frio": "CRÍTICO - Frío < 18°C reduce cuajado",
                    "riesgo_lluvia": "Lluvias torrenciales limpian polen",
                    "riesgo_hongos": "Alta humedad favorece piricularia"
                },
                "llenado_grano": {
                    "temperatura_optima": "23-30°C",
                    "dias": "110-140 días post-siembra",
                    "riesgo_sequia": "CRÍTICO - Estrés hídrico reduce peso del grano",
                    "riesgo_calor": "Temperaturas > 36°C causan granos yesosos",
                    "riesgo_viento": "Viento + lluvia favorecen acame"
                },
                "maduracion": {
                    "temperatura_optima": "22-28°C",
                    "dias": "140-165 días post-siembra",
                    "riesgo_lluvia": "CRÍTICO - Lluvia prolongada provoca brotado",
                    "condicion_ideal": "Secado gradual con viento moderado"
                }
            },
            "malezas_criticas": [
                "Hualcacho (Echinochloa spp.)",
                "Arroz rojo (Oryza sativa var. sylvatica)",
                "Ciperáceas (Cyperus difformis, Scirpus spp.)",
                "Dodder o cabellos de angel (Cuscuta spp.)",
                "Chépica (Paspalum distichum)"
            ],
            "plagas_climaticas": {
                "chinche_del_grano": {
                    "condiciones": "Temperaturas > 24°C con humedad alta",
                    "periodo_critico": "Espigamiento a llenado",
                    "sintomas": "Punteado en granos y decoloración"
                },
                "mosquito_del_arroz": {
                    "condiciones": "Aguas estancadas + 20-28°C",
                    "periodo_critico": "Macollaje",
                    "sintomas": "Plántulas cortadas y macollos huecos"
                }
            },
            "enfermedades_climaticas": {
                "piricularia": {
                    "condiciones": "Días calurosos + noches húmedas",
                    "periodo_critico": "Espigamiento",
                    "sintomas": "Lesiones en forma de ojo en hojas y panojas"
                },
                "tizón_de_la_vaina": {
                    "condiciones": "Alta humedad y temperaturas > 28°C",
                    "periodo_critico": "Macollaje tardío a llenado",
                    "sintomas": "Manchas pardas en vainas y tallos"
                },
                "mancha_parda": {
                    "condiciones": "Humedad relativa > 80%",
                    "periodo_critico": "Todo el ciclo",
                    "sintomas": "Puntos marrones en hojas y granos"
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
            "chillan", "santiago", "valparaiso", "concepcion", "temuco", "valdivia", "osorno",
            "puerto montt", "iquique", "antofagasta", "la serena", "rancagua", "talca", "curico",
            "linares", "chillán", "los angeles", "angol", "villarrica", "pucón", "castro",
            "coyhaique", "punta arenas", "arica", "calama", "copiapó", "ovalle", "quillota"
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
                "viña del mar": "Viña del Mar",
                "vina del mar": "Viña del Mar"
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
                        excluded = {'arroz', 'aplicar', 'producto'}
                        if candidate.lower() not in excluded:
                            intent["weather_location"] = candidate.title()
                            print(f"[Experto Arroz] Ciudad detectada: {candidate}")
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
            "controlar maleza", "eliminar", "precauciones",
            "malezas", "plagas", "piricularia", "chinche",
            "propanil", "bispiribac", "oxadiazon", "penoxsulam", "clomazone",
            "quatran", "facet", "command", "strada", "saturn"
        ]

        if any(keyword in query_lower for keyword in rag_keywords):
            intent["needs_rag"] = True
            intent["query_type"] = "technical_product"

        if any(word in query_lower for word in ["variedad", "variedades", "cultivar"]):
            intent["query_type"] = "variety_selection"
        elif any(word in query_lower for word in ["siembra", "plantar", "trasplante"]):
            intent["query_type"] = "planting"
        elif any(word in query_lower for word in ["fertilizar", "fertilización", "abonado"]):
            intent["query_type"] = "fertilization"

        return intent

    def _get_weather_data(self, location: str, days: int = 7) -> Optional[Dict[str, Any]]:
        """Obtiene datos del clima usando el pipeline de clima con manejo mejorado"""
        if not self.weather_system:
            return None

        try:
            if days == 1:
                query = f"Clima hoy en {location}"
            elif days == 2:
                query = f"Clima mañana en {location}"
            else:
                query = f"Pronóstico del clima para {location} próximos {days} días"

            print(f"[Experto Arroz] 🌤️ Ejecutando pipeline clima: '{query}'")

            result = self.weather_system({
                "messages": [{"role": "user", "content": query}]
            })

            print(f"[Experto Arroz] Resultado clima tipo: {type(result)}")

            if isinstance(result, dict):
                response = result.get("output") or result.get("response") or str(result)
            else:
                response = str(result)

            print(f"[Experto Arroz] Respuesta clima: {len(response)} caracteres")

            if response and len(response) > 50:
                print(f"[Experto Arroz] ✅ Clima obtenido exitosamente")
                return {
                    "raw_response": response,
                    "location": location,
                    "days": days,
                    "success": True
                }
            else:
                print(f"[Experto Arroz] ⚠️ Respuesta clima muy corta o vacía")
                return None

        except Exception as e:
            print(f"[Experto Arroz] Error obteniendo clima: {e}")
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
                "que herbicida", "qué herbicida", "herbicida para arroz",
                "herbicidas para", "control de malezas", "herbicida usar",
                "fungicida para", "insecticida para"
            ])

            if is_general:
                print(f"[Experto Arroz] 🔍 Consulta general - búsqueda amplia")

                search_queries = [
                    "arroz herbicida preemergente malezas",
                    "arroz postemergente control hualcacho",
                    "arroz fungicida piricularia manejo",
                    "arroz insecticida chinche del grano",
                    "arroz manejo malezas ciperaceas"
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
                print(f"[Experto Arroz] ✅ Encontrados {len(results)} documentos únicos")

            else:
                enriched_query = f"arroz cultivo {query}"
                results = self.rag_system.search(enriched_query, top_k=top_k)

            if not results:
                return None

            formatted = ["# INFORMACIÓN TÉCNICA DE PESTICIDAS PARA ARROZ\n"]
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
            print(f"[Experto Arroz] Error en RAG: {e}")
            return None

    def _create_system_prompt(self, query: str, intent: Dict[str, Any], weather_data: Optional[Dict] = None, rag_data: Optional[str] = None) -> str:
        """Crea el prompt del sistema integrando todas las fuentes"""

        prompt = """Eres un **Experto Agrónomo Especializado en Cultivo de Arroz** en Chile.


Tienes acceso a MÚLTIPLES fuentes de información para dar la mejor asesoría:


## 📚 FUENTES DE INFORMACIÓN DISPONIBLES


### 1️⃣ CONOCIMIENTO EXPERTO INTEGRADO
- Variedades de arroz chilenas (Tucapel-INIA, Quellón-INIA, Zafiro-INIA)
- Manejo agronómico completo del cultivo
- Etapas fenológicas y requerimientos hídricos
- Plagas, enfermedades y malezas específicas del arroz


### 2️⃣ DATOS METEOROLÓGICOS EN TIEMPO REAL
"""

        if weather_data:
            if weather_data.get("error") == "no_access":
                prompt += f"""
⚠️ **CLIMA NO DISPONIBLE PARA: {weather_data.get('location', 'Ubicación')}**


{weather_data.get('message', 'No hay acceso a datos climáticos')}


**INSTRUCCIONES PARA CONSULTAS DE CLIMA:**
- Explica que intentaste obtener datos climáticos pero no están disponibles
- Recomienda verificar el pronóstico local antes de inundar o aplicar
- Da recomendaciones generales sobre condiciones ideales
- Menciona factores críticos: lluvia (24h), viento (<10 km/h), temperatura (20-30°C)
"""
            else:
                prompt += f"""
✅ **CLIMA DISPONIBLE PARA: {weather_data.get('location', 'Ubicación')}**


{weather_data.get('raw_response', '')}


**INSTRUCCIONES CRÍTICAS PARA USO DE DATOS CLIMÁTICOS:**
- ANALIZA lluvia, viento y temperatura para las próximas 24 h
- Si hay lluvia intensa → ADVERTIR sobre riesgos de desbordes/acame
- Si viento > 10 km/h → ADVERTIR sobre deriva en aplicaciones aéreas
- Si temperatura < 17°C o > 35°C → RECOMENDAR postergar labores sensibles
- Si condiciones favorables → INDICA horarios óptimos y manejo del agua
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
- Emojis para mejorar legibilidad (✅ ❌ ⚠️ 🍚 📊)


### ETAPAS CRÍTICAS Y CLIMA:


**SIEMBRA** (Oct-Nov): 18-25°C | Riesgo: lluvias intensas o escasez de agua
**MACOLLAJE** (30-60 días): 20-28°C | Frío < 15°C reduce macollos
**PRIMORDIO** (70-90 días): ⚠️ **CRÍTICO** | 22-30°C | Frío < 17°C causa esterilidad
**ESPIGAMIENTO** (90-110 días): ⚠️ **MUY CRÍTICO** | 24-32°C | Lluvia + humedad = piricularia
**LLENADO** (110-140 días): 23-30°C | Sequía = granos livianos | Calor > 36°C = yesosidad
**MADURACIÓN** (140-165 días): Período seco ideal | Lluvia = brotado y calidad industrial baja


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

            print(f"[Experto Arroz] 📝 Consulta: {user_message[:100]}...")

            intent = self._detect_query_intent(user_message)
            print(f"[Experto Arroz] 🎯 Tipo: {intent['query_type']}")
            print(f"[Experto Arroz] 🌤️  Clima: {intent['needs_weather']}")
            print(f"[Experto Arroz] 📚 RAG: {intent['needs_rag']}")

            weather_data = None
            if intent["needs_weather"]:
                location = intent.get("weather_location", "Santiago")
                days = intent.get("weather_days", 7)

                print(f"[Experto Arroz] 🌍 Obteniendo clima: {location}, {days} días")

                if self.weather_system:
                    weather_data = self._get_weather_data(location, days)
                    if weather_data:
                        print(f"[Experto Arroz] ✅ Clima obtenido")
                    else:
                        print(f"[Experto Arroz] ⚠️  No se pudo obtener clima")
                else:
                    weather_data = {
                        "error": "no_access",
                        "location": location,
                        "message": f"No tengo acceso a datos climáticos para {location}"
                    }

            rag_data = None
            if intent["needs_rag"] and self.rag_system:
                print(f"[Experto Arroz] 📖 Buscando en documentos...")
                rag_data = self._get_rag_data(user_message, top_k=15)

                if rag_data:
                    print(f"[Experto Arroz] ✅ Documentos encontrados")
                else:
                    print(f"[Experto Arroz] ⚠️  No se encontraron documentos")

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
                print(f"[Experto Arroz] 🤖 Usando LLM Router con fallback")
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
                        print(f"[Experto Arroz] 🔄 Fallback usado: {model_used}")
                    else:
                        print(f"[Experto Arroz] ✅ Modelo principal: {model_used}")
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
                    print(f"[Experto Arroz] {error_msg}")
                    return self._create_error_response(error_msg)

            footer = "\n\n---\n"
            sources = []

            if weather_data:
                sources.append("🌤️ Clima en tiempo real")
            if rag_data:
                sources.append("📚 Documentos técnicos")
            sources.append("🧠 Conocimiento experto")

            footer += f"**Fuentes**: {' + '.join(sources)}\n"
            footer += "🍚 *Experto Completo en Arroz - AgroIA v1.0.0*"

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
            print(f"[Experto Arroz] {error_msg}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(error_msg)

    def _create_success_response(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Crea respuesta exitosa en formato OpenAI"""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "rice_expert_complete",
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
            "model": "rice_expert_complete",
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
        self.pipeline = RiceExpertCompletePipeline()
        self.name = "Experto Completo en Arroz"
        self.id = "rice_expert_complete"

        self.type = "llm"
        self.description = "Experto completo en arroz con respuestas optimizadas en formato OpenAI"
        self.version = "1.0.0"

    def pipe(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Devuelve directamente formato OpenAI"""
        result = self.pipeline.pipe(body)
        return result

    def __call__(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipe(body)


pipeline = Pipeline()
