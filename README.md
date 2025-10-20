# 🌱 AgroIA - Experto en Lentejas

Sistema de inteligencia artificial **especializado en cultivo de lentejas** con pipeline optimizado para herbicidas, clima y variedades chilenas.

## ⭐ **Pipeline Principal: `lentil_expert_complete`**

¡El pipeline más avanzado y optimizado del sistema!

**✅ Características:**
- **27 herbicidas** completamente documentados
- **Integración climática** para 40+ ciudades chilenas
- **Variedades locales** (Araucana-INIA, Montaña-INIA)
- **Respuestas optimizadas** con formato claro y organizado
- **Búsqueda inteligente** en documentos técnicos

## 🚀 Inicio Rápido

### 1. Activar Entorno Virtual
```bash
# ⚠️ IMPORTANTE: Siempre activar el entorno virtual primero
.venv\Scripts\activate
```

### 2. Ejecutar Backend
```bash
# Después de activar el entorno virtual
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Acceder al Sistema
- **URL**: http://localhost:3002
- **Documentación API**: http://localhost:8000/docs 
"fastapi"

## 📋 Otros Pipelines Disponibles

### 🌱 **Lentejas (Alternativos)**
- **`lentil_expert_v2`** - Versión básica del experto en lentejas

### 🔍 **Sistemas de Apoyo**
- **`rag_lentejas_optimized`** - Búsqueda en documentos (usado por lentil_expert_complete)
- **`weather_chat_enhanced`** - Análisis climático (integrado en lentil_expert_complete)
- **`agroia_master`** - Coordinador general de múltiples fuentes

### 🛒 **Utilidades**
- **`google_search_pipeline`** - Búsqueda web
- **`mercadolibre_pipeline`** - Precios de productos
- **`langchain_v3`** - Pipeline genérico
- **`example_pipeline`** - Ejemplo para desarrollo

## ⚙️ Configuración Inicial

### 1. Instalar Dependencias
```bash
# Crear entorno virtual (solo primera vez)
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno
```bash
# Crear archivo .env con tus API keys
OPENAI_API_KEY=tu_api_key_aqui
OPENAI_MODEL=gpt-4o
TOMORROW_API_KEY=tu_weather_api_key  # Opcional
```

## 📝 Consultas de Ejemplo para `lentil_expert_complete`

### 🌱 **Herbicidas Preemergentes** (Especialidad principal)
```
"¿Qué herbicidas preemergentes para lentejas?"
"¿Dosis de Afalon 50 SC para lentejas?"
"¿Cuándo aplicar Tiburon 500 SC?"
"Herbicidas para malezas de hoja ancha en lentejas"
```

### 🌤️ **Consultas con Clima Integrado**
```
"¿Puedo aplicar herbicida mañana en Chillán?"
"Condiciones climáticas para aplicar en Temuco"
"Si quisiera aplicar Afalon en Talca me lo recomendarías por el clima"
```

### 🌿 **Control de Malezas Específicas**
```
"¿Cómo controlar correhuela en lentejas?"
"¿Herbicida para yuyo en lentejas?"
"Control de ballica en cultivo de lentejas"
```

### 🎯 **Recomendaciones Completas**
```
"Recomienda herbicida preemergente para mi cultivo de lentejas"
"Mejor herbicida para lentejas considerando el clima"
```


## ⚠️ Solución de Problemas

### Error: "No module found"
```bash
# Verificar que el entorno virtual esté activado
.venv\Scripts\activate
pip install -r requirements.txt
```

### Error: "OpenAI API key not found"
```bash
# Verificar archivo .env existe
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows
```

## 📊 Estado del Sistema - `lentil_expert_complete`

- **✅ OPTIMIZADO**: Pipeline principal completamente funcional
- **Base de Datos**: 356 documentos procesados especializados en lentejas
- **Herbicidas**: 27 productos completamente documentados
- **Clima**: 40+ ciudades chilenas integradas
- **Formato**: Respuestas organizadas y fáciles de leer
- **Última actualización**: Octubre 2025 - Versión optimizada

## 🛠️ Desarrollo

### Agregar Nuevos Herbicidas
```bash
# 1. Colocar PDF en documents/
# 2. Procesar documento
python ingest_lentejas_one_by_one.py
```

### Estructura de Archivos
```
openwebui-langchain-backend/
├── pipelines/          # Todos los pipelines especializados
├── documents/          # PDFs de herbicidas
├── chroma_lentejas_v2/ # Base de datos
├── main.py             # Servidor principal
└── .env                # Configuración (crear)
```
