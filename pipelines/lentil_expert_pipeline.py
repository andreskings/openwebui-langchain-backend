"""
Pipeline: Experto Completo en Lentejas
Integra: Conocimiento experto + RAG + OpenAI + Sistema de Clima

Autor: AgroIA Expert System
Versión: 3.2.0 - OPTIMIZADA (Formato OpenAI + Visualización Mejorada)
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

# Añadir path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))

# Metadata del pipeline
metadata = {
    "name": "lentil_expert_complete",
    "description": "Experto completo en lentejas: Conocimiento + RAG + Clima + OpenAI con respuestas optimizadas",
    "type": "llm",
    "version": "3.2.0",
    "author": "AgroIA",
    "tags": ["lentejas", "clima", "rag", "herbicidas", "completo", "openai-format"]
}


class LentilExpertCompletePipeline:
    """Pipeline experto completo en lentejas con todas las integraciones"""
    
    def __init__(self):
        self.name = "🌱 Experto Completo en Lentejas"
        
        # Configuración de API
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        # Inicializar componentes
        self.rag_system = self._init_rag_system()
        self.weather_system = self._init_weather_system()
        self.knowledge_base = self._load_lentil_knowledge()
        
        print(f"[Experto Lentejas] 🌱 Pipeline Completo inicializado")
        print(f"[Experto Lentejas] 📚 RAG: {'✅' if self.rag_system else '❌'}")
        print(f"[Experto Lentejas] 🌤️  Clima: {'✅' if self.weather_system else '❌'}")
        print(f"[Experto Lentejas] 🤖 Modelo: {self.model}")
    
    def _init_rag_system(self):
        """Inicializa el sistema RAG de lentejas optimizado"""
        try:
            import sys
            from pathlib import Path
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from rag_lentejas_optimized import LentilRAGSystem
            rag = LentilRAGSystem()
            doc_count = rag.count()
            print(f"[Experto Lentejas] RAG Lentejas activo con {doc_count} documentos")
            return rag if doc_count > 0 else None
        except Exception as e:
            print(f"[Experto Lentejas] RAG no disponible: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _init_weather_system(self):
        """Inicializa el sistema de clima con búsqueda mejorada"""
        try:
            import importlib.util
            
            weather_file = "weather_chat_enhanced.py"
            print(f"[Experto Lentejas] Buscando: {weather_file}")
            
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
                
                print(f"[Experto Lentejas]   Probando: {full_path}")
                if os.path.exists(full_path):
                    found_path = full_path
                    print(f"[Experto Lentejas] Pipeline clima encontrado: {full_path}")
                    break
            
            if not found_path:
                print(f"[Experto Lentejas] Pipeline clima no encontrado: {weather_file}")
                print(f"[Experto Lentejas]    Buscado en: {search_paths}")
                return None
            
            # Importar módulo
            print(f"[Experto Lentejas] Importando módulo clima...")
            spec = importlib.util.spec_from_file_location("weather_chat_enhanced", found_path)
            weather_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(weather_module)
            
            # Verificar que tenga función pipeline
            if not hasattr(weather_module, 'pipeline'):
                print(f"[Experto Lentejas] Módulo clima sin función 'pipeline'")
                return None
            
            print(f"[Experto Lentejas] Sistema de clima integrado exitosamente")
            return weather_module.pipeline
                
        except Exception as e:
            print(f"[Experto Lentejas] Error cargando clima: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_lentil_knowledge(self) -> Dict[str, Any]:
        """Carga la base de conocimientos sobre lentejas"""
        return {
            "variedades_chile": {
                "araucana": {
                    "nombre": "Araucana-INIA",
                    "tipo": "Lenteja Grande",
                    "ciclo": "Tardío (160-180 días)",
                    "rendimiento": "25-35 qq/ha",
                    "resistencia": "Resistente a roya y pulgón verde",
                    "zona": "Región del Maule y Ñuble"
                },
                "montana": {
                    "nombre": "Montaña-INIA",
                    "tipo": "Lenteja Grande",
                    "ciclo": "Intermedio (150-160 días)",
                    "rendimiento": "28-38 qq/ha",
                    "tolerancia": "Tolerante a sequía y heladas",
                    "zona": "Regiones del Biobío y La Araucanía"
                }
            },
            "etapas_criticas_clima": {
                "germinacion": {
                    "temperatura_optima": "15-20°C",
                    "temperatura_minima": "5°C",
                    "dias": "7-14 días post-siembra",
                    "riesgo_lluvia": "Lluvia excesiva causa pudrición de semilla",
                    "riesgo_sequia": "Sequía reduce stand de plantas"
                },
                "floracion": {
                    "temperatura_optima": "18-24°C",
                    "dias": "50-70 días post-siembra",
                    "riesgo_heladas": "CRÍTICO - Heladas causan aborto floral",
                    "temperatura_critica": "-2°C",
                    "riesgo_lluvia": "Lluvia dificulta polinización",
                    "riesgo_calor": "Temperaturas > 30°C causan aborto"
                },
                "llenado_grano": {
                    "temperatura_optima": "20-25°C",
                    "dias": "70-100 días post-siembra",
                    "riesgo_sequia": "CRÍTICO - Reduce tamaño y número de granos",
                    "riesgo_calor": "Acelera maduración, reduce calibre",
                    "riesgo_lluvia": "Favorece enfermedades (rabia, oidio)"
                },
                "maduracion": {
                    "temperatura_optima": "20-28°C",
                    "dias": "100-120 días post-siembra",
                    "riesgo_lluvia": "CRÍTICO - Causa manchado de grano y germinación en vaina",
                    "condicion_ideal": "Período seco para secado uniforme"
                }
            },
            "malezas_criticas": [
                "Avena guacha (Avena fatua)",
                "Ballica (Lolium multiflorum)",
                "Yuyo (Brassica campestris)",
                "Correhuela (Convolvulus arvensis)"
            ],
            "enfermedades_climaticas": {
                "roya": {
                    "condiciones": "Humedad alta + 15-22°C",
                    "periodo_critico": "Primavera húmeda",
                    "sintomas": "Pústulas café-rojizo"
                },
                "rabia": {
                    "condiciones": "Lluvia persistente + 15-20°C",
                    "periodo_critico": "Post-floración",
                    "sintomas": "Manchas con centro gris"
                },
                "oidio": {
                    "condiciones": "Días secos + noches frescas + 20-25°C",
                    "periodo_critico": "Primaveras secas",
                    "sintomas": "Polvo blanco en hojas"
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
                "cobquecura": "Cobquecura",
                "tome": "Tomé",
                "penco": "Penco",
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
                location_patterns = [
                    r'en\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)',
                    r'para\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)',
                    r'de\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)',
                    r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\s+mañana',
                    r'mañana\s+en\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)'
                ]
                
                for pattern in location_patterns:
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        potential_city = match.group(1).strip()
                        excluded_words = ['aplicar', 'herbicida', 'producto', 'lentejas', 'clima', 'tiempo']
                        if potential_city.lower() not in excluded_words:
                            intent["weather_location"] = potential_city.title()
                            print(f"[Experto Lentejas] Ciudad detectada: {potential_city}")
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
            
            if "proximos" in query_lower or "próximos" in query_lower:
                import re
                dias_match = re.search(r'(\d+)\s*días', query_lower)
                if dias_match:
                    intent["weather_days"] = int(dias_match.group(1))
        
        # Detectar necesidad de RAG
        rag_keywords = [
            "herbicida", "producto", "dosis", "aplicación",
            "etiqueta", "ficha técnica", "principio activo",
            "controlar maleza", "eliminar", "precauciones",
            "malezas", "hoja ancha", "hoja angosta",
            "afalon", "agil", "linuron", "centurion", "cletodim",
            "treflan", "spectro", "ripper", "glifosato"
        ]
        
        if any(keyword in query_lower for keyword in rag_keywords):
            intent["needs_rag"] = True
            intent["query_type"] = "technical_product"
        
        if any(word in query_lower for word in ["variedad", "variedades"]):
            intent["query_type"] = "variety_selection"
        elif any(word in query_lower for word in ["sembrar", "siembra"]):
            intent["query_type"] = "planting"
        elif any(word in query_lower for word in ["fertilizar", "fertilización"]):
            intent["query_type"] = "fertilization"
        
        return intent
    
    def _get_weather_data(self, location: str, days: int = 7) -> Optional[Dict[str, Any]]:
        """Obtiene datos del clima usando el pipeline de clima con manejo mejorado"""
        if not self.weather_system:
            return None
        
        try:
            query = f"Pronóstico del clima para {location} próximos {days} días"
            print(f"[Experto Lentejas] 🌤️ Ejecutando pipeline clima...")
            
            # Ejecutar pipeline con formato estándar
            result = self.weather_system({
                "messages": [{"role": "user", "content": query}]
            })
            
            print(f"[Experto Lentejas] Resultado clima tipo: {type(result)}")
            
            # Extraer respuesta del resultado
            if isinstance(result, dict):
                response = result.get("output") or result.get("response") or str(result)
            else:
                response = str(result)
            
            print(f"[Experto Lentejas] Respuesta clima: {len(response)} caracteres")
            
            if response and len(response) > 50:
                print(f"[Experto Lentejas] ✅ Clima obtenido exitosamente")
                return {
                    "raw_response": response,
                    "location": location,
                    "days": days,
                    "success": True
                }
            else:
                print(f"[Experto Lentejas] ⚠️ Respuesta clima muy corta o vacía")
                return None
        
        except Exception as e:
            print(f"[Experto Lentejas] Error obteniendo clima: {e}")
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
                "que herbicida", "qué herbicida", "herbicida para lentejas",
                "herbicidas para", "control de malezas", "herbicida usar",
                "malezas de hoja ancha", "hoja ancha"
            ])
            
            if is_general:
                print(f"[Experto Lentejas] 🔍 Consulta general - búsqueda amplia")
                
                search_queries = [
                    "lentejas herbicida preemergente hoja ancha",
                    "leguminosas herbicida malezas control",
                    "herbicida postemergente lentejas dosis",
                    "lentejas malezas hoja ancha control",
                    "herbicida selectivo lentejas aplicación"
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
                print(f"[Experto Lentejas] ✅ Encontrados {len(results)} documentos únicos")
            
            else:
                enriched_query = f"lentejas leguminosas {query}"
                results = self.rag_system.search(enriched_query, top_k=top_k)
            
            if not results:
                return None
            
            formatted = [f"# INFORMACIÓN TÉCNICA DE HERBICIDAS PARA LENTEJAS\n"]
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
            print(f"[Experto Lentejas] Error en RAG: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_system_prompt(self, query: str, intent: Dict, weather_data: Optional[Dict] = None, rag_data: Optional[str] = None) -> str:
        """Crea el prompt del sistema integrando todas las fuentes"""
        
        prompt = """Eres un **Experto Agrónomo Especializado en Cultivo de Lentejas** en Chile.

Tienes acceso a MÚLTIPLES fuentes de información para dar la mejor asesoría:

## 📚 FUENTES DE INFORMACIÓN DISPONIBLES

### 1️⃣ CONOCIMIENTO EXPERTO INTEGRADO
- Variedades de lentejas chilenas (Araucana-INIA, Montaña-INIA)
- Manejo agronómico completo del cultivo
- Etapas fenológicas y requerimientos
- Plagas, enfermedades y malezas

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
- Emojis para mejorar legibilidad (✅ ❌ ⚠️ 🌱 📊)

### PARA CONSULTAS DE HERBICIDAS CON MÚLTIPLES PRODUCTOS:

```markdown
## HERBICIDAS PARA LENTEJAS

**1. AFALON 50 SC**
Dosis: 1.0-1.5 L/ha
Momento: Inmediatamente post-siembra
P. Activo: Linurón
Observaciones: Mojamiento 200-250 L/ha

**2. TIBURON 500 SC**
Dosis: 1.0-1.5 L/ha
Momento: Inmediatamente post-siembra
P. Activo: Linurón
Observaciones: Efectivo contra hoja ancha

**3. PENDICLAN 33 EC**
Dosis: 3-4 L/ha
Momento: Pre-siembra o pre-emergencia
P. Activo: Pendimetalina
Observaciones: Evitar suelos con <2% M.O.

**4. SPECTRO 33 EC**
Dosis: 3-4 L/ha
Momento: Pre-siembra o pre-emergencia
P. Activo: Pendimetalina
Observaciones: Evitar suelos delgados o arenosos

**5. TREFLAN**
Dosis: 1.0-2.5 L/ha
Momento: Pre-siembra incorporado
P. Activo: Trifluralina
Observaciones: Dosis varía según tipo de suelo

---

**TOP 3 RECOMENDADOS:**
1. Afalon 50 SC - Eficacia comprobada y fácil aplicación
2. Tiburon 500 SC - Excelente control hoja ancha
3. Pendiclan 33 EC - Amplio espectro de control

**CONDICIONES CLIMA:** Viento <10 km/h, sin lluvia 24h, temp 15-25°C

**PRECAUCIONES:** EPP obligatorio, respetar carencias, no aplicar en suelos arenosos
```

### REGLAS CRÍTICAS:

1. **FORMATO COMPLETO:** **N. NOMBRE** + Dosis + Momento + P.Activo + Observaciones
2. **ESPACIOS OBLIGATORIOS:** Siempre espacio entre palabras
3. **LÍNEA VACÍA:** Entre cada producto para separación
4. **MÁXIMO 8 PRODUCTOS:** Incluir todos los productos relevantes encontrados
5. **INFORMACIÓN TÉCNICA:** Dosis exactas, principio activo, observaciones clave
6. **SEPARADOR:** Usar --- entre secciones principales
7. **RECOMENDACIONES:** Incluir TOP recomendado + condiciones + precauciones

### ETAPAS CRÍTICAS Y CLIMA:

**GERMINACIÓN** (0-14 días): 15-20°C óptimo | Riesgo: lluvia excesiva/sequía
**FLORACIÓN** (50-70 días): ⚠️ **MUY CRÍTICO** | 18-24°C | Heladas < -2°C = ABORTO TOTAL
**LLENADO** (70-100 días): 20-25°C | Sequía = reduce calibre | Lluvia = enfermedades
**MADURACIÓN** (100-120 días): Período seco ideal | Lluvia = manchado/germinación en vaina

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
        """✅ MEJORADO: Procesa la consulta y devuelve formato OpenAI"""
        
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
            
            print(f"[Experto Lentejas] 📝 Consulta: {user_message[:100]}...")
            
            # Detectar intención
            intent = self._detect_query_intent(user_message)
            print(f"[Experto Lentejas] 🎯 Tipo: {intent['query_type']}")
            print(f"[Experto Lentejas] 🌤️  Clima: {intent['needs_weather']}")
            print(f"[Experto Lentejas] 📚 RAG: {intent['needs_rag']}")
            
            # Obtener datos de clima
            weather_data = None
            if intent["needs_weather"]:
                location = intent.get("weather_location", "Talca")
                days = intent.get("weather_days", 7)
                
                print(f"[Experto Lentejas] 🌍 Obteniendo clima: {location}, {days} días")
                
                if self.weather_system:
                    weather_data = self._get_weather_data(location, days)
                    if weather_data:
                        print(f"[Experto Lentejas] ✅ Clima obtenido")
                    else:
                        print(f"[Experto Lentejas] ⚠️  No se pudo obtener clima")
                else:
                    weather_data = {
                        "error": "no_access",
                        "location": location,
                        "message": f"No tengo acceso a datos climáticos para {location}"
                    }
            
            # Obtener datos RAG
            rag_data = None
            if intent["needs_rag"] and self.rag_system:
                print(f"[Experto Lentejas] 📖 Buscando en documentos...")
                rag_data = self._get_rag_data(user_message, top_k=15)
                
                if rag_data:
                    print(f"[Experto Lentejas] ✅ Documentos encontrados")
                else:
                    print(f"[Experto Lentejas] ⚠️  No se encontraron documentos")
            
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
            
            # Llamar a OpenAI
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
                
                # Footer informativo
                footer = "\n\n---\n"
                sources = []
                
                if weather_data:
                    sources.append("🌤️ Clima en tiempo real")
                if rag_data:
                    sources.append("📚 Documentos técnicos")
                sources.append("🧠 Conocimiento experto")
                
                footer += f"**Fuentes**: {' + '.join(sources)}\n"
                footer += "🌱 *Experto Completo en Lentejas - AgroIA v3.2.0*"
                
                # ✅ NUEVO: Devolver en formato OpenAI
                return self._create_success_response(
                    answer + footer,
                    {
                        "used_weather": weather_data is not None,
                        "used_rag": rag_data is not None,
                        "query_type": intent["query_type"],
                        "model": self.model,
                        "version": "3.2.0"
                    }
                )
            else:
                error_msg = f"Error en API OpenAI: {response.status_code}"
                print(f"[Experto Lentejas] {error_msg}")
                return self._create_error_response(error_msg)
        
        except Exception as e:
            error_msg = f"Error procesando consulta: {str(e)}"
            print(f"[Experto Lentejas] {error_msg}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(error_msg)
    
    def _create_success_response(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """✅ NUEVO: Crea respuesta exitosa en formato OpenAI"""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "lentil_expert_complete",
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
        """✅ NUEVO: Crea respuesta de error en formato OpenAI"""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "lentil_expert_complete",
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


# ✅ MEJORADO: Clase Pipeline con formato OpenAI
class Pipeline:
    def __init__(self):
        self.pipeline = LentilExpertCompletePipeline()
        self.name = "Experto Completo en Lentejas"
        self.id = "lentil_expert_complete"
        
        # Metadata para OpenWebUI
        self.type = "llm"
        self.description = "Experto completo en lentejas con respuestas optimizadas en formato OpenAI"
        self.version = "3.2.0"
    
    def pipe(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """✅ OPTIMIZADO: Devuelve directamente formato OpenAI"""
        result = self.pipeline.pipe(body)
        
        # El pipeline ya devuelve formato OpenAI correcto
        # No necesitamos transformar nada
        return result
    
    def __call__(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipe(body)


# Instancia global
pipeline = Pipeline()