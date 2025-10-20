"""
Sistema RAG Optimizado Específicamente para Lentejas
Extracción avanzada de herbicidas con tablas y metadata enriquecida
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
import requests
import fitz  # PyMuPDF
import pdfplumber
import re
from dataclasses import dataclass

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HerbicideInfo:
    """Estructura completa para información de herbicidas"""
    name: str
    active_ingredient: str = ""
    concentration: str = ""
    formulation: str = ""  # WP, EC, SC, SL, etc.
    dose_range: str = ""
    dose_unit: str = ""  # kg/ha, L/ha, g/ha
    application_timing: str = ""
    application_method: str = ""  # Aspersión, incorporación, etc.
    target_weeds: List[str] = None
    controlled_weeds: List[str] = None
    crops: List[str] = None
    crop_stage: str = ""  # Preemergente, postemergente, etc.
    water_volume: str = ""  # Volumen de agua
    precautions: List[str] = None
    restrictions: List[str] = None
    compatibility: List[str] = None
    carency_period: str = ""  # Período de carencia
    toxicity_class: str = ""
    source_file: str = ""
    
    def __post_init__(self):
        if self.target_weeds is None:
            self.target_weeds = []
        if self.controlled_weeds is None:
            self.controlled_weeds = []
        if self.crops is None:
            self.crops = []
        if self.precautions is None:
            self.precautions = []
        if self.restrictions is None:
            self.restrictions = []
        if self.compatibility is None:
            self.compatibility = []

class LentilRAGSystem:
    """Sistema RAG optimizado para herbicidas de lentejas"""
    
    def __init__(self, persist_directory: str = "./chroma_lentejas_v2"):
        self.persist_directory = persist_directory
        self.embedding_cache = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # Configurar ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Crear colección específica para lentejas (SIN OpenAI embeddings - usa embeddings locales 384D)
        self.collection = self.client.get_or_create_collection(
            name="herbicidas_lentejas",
            metadata={
                "description": "Herbicidas específicos para cultivo de lentejas",
                "embedding_model": "sentence-transformers",
                "embedding_dimension": 384
            }
        )
        
        logger.info(f"[RAG Lentejas] Inicializado con {self.collection.count()} documentos")
    
    def extract_herbicide_info(self, text: str, source_file: str) -> List[HerbicideInfo]:
        """Extrae información COMPLETA de herbicidas del texto"""
        herbicides = []
        text_lower = text.lower()
        
        # 1. EXTRAER NOMBRES DE PRODUCTOS (patrones mejorados y filtrados)
        name_patterns = [
            r'([A-Z][A-Z\s]*\d*\s*(?:WP|EC|SC|SL|WG)\b)',  # Productos con formulación
            r'\b([A-Z][A-Z]+)\s*®?\s*\d*\s*%?\s*(?:WP|EC|SC|SL|WG)\b',  # Marcas registradas
            r'Producto:\s*([^\n]+)',
            r'Herbicida:\s*([^\n]+)',
            r'Nombre comercial:\s*([^\n]+)',
            r'ETIQUETA\s+([A-Z][A-Z\s]+\d*)',
            # Patrones específicos para nombres conocidos
            r'\b(AFALON|LINUREX|AGIL|CENTURION|CLETODIM|RIPPER|SPECTRO|TREFLAN)\b.*?(?:WP|EC|SC|SL|WG)?'
        ]
        
        # 2. PATRONES PARA PRINCIPIOS ACTIVOS
        active_ingredient_patterns = [
            r'(?:principio activo|ingrediente activo|p\.a\.|i\.a\.):\s*([^\n]+)',
            r'(?:contiene|composición):\s*([^\n]+)',
            r'([a-z\-]+(?:\s+[a-z\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*%',
            r'([A-Z][a-z\-]+(?:\s+[a-z\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*g/L',
            r'([A-Z][a-z\-]+(?:\s+[a-z\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*g/kg'
        ]
        
        # 3. PATRONES PARA DOSIS (ultra-completos y específicos para tablas)
        dose_patterns = [
            # Patrones básicos
            r'(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(l/ha|g/ha|litros/ha|gramos/ha|ml/ha|kg/ha|cc/ha|lt/ha)',
            r'(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(litros?\s*por\s*hectarea|gramos?\s*por\s*hectarea)',
            r'dosis:?\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(l/ha|g/ha|ml/ha|kg/ha)',
            
            # Patrones para tablas (más específicos)
            r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*(?:kg|g|l|ml)\s*/\s*ha',
            r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*(kg|L|g|ml|cc)\s*/\s*ha',
            r'(\d+(?:[,\.]\d+)?)\s*a\s*(\d+(?:[,\.]\d+)?)\s*(kg|L|g|ml)\s*/\s*ha',
            
            # Patrones específicos para celdas de tabla
            r'(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)\s*/\s*há',
            r'(\d+(?:[,\.]\d+)?)\s*-\s*(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)',
            r'(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)\s*por\s*hectárea',
            
            # Patrones generales
            r'Dosis:\s*([^\n|]+)',
            r'aplicar\s+(\d+(?:[,\.]\d+)?)\s*(kg|L|g|ml)/ha',
            r'usar\s+(\d+(?:[,\.]\d+)?)\s*(kg|L|g|ml)/ha',
            r'volumen\s*de\s*aplicación:\s*(\d+(?:[,\.]\d+)?)\s*(l/ha|litros/ha)'
        ]
        
        # 4. PATRONES PARA MALEZAS CONTROLADAS
        weed_patterns = [
            r'(?:controla|control de|eficaz contra):\s*([^\n\.]+)',
            r'malezas?\s+controladas?:\s*([^\n\.]+)',
            r'(?:yuyo|ballica|avena guacha|correhuela|chepica|pata de gallina|verdolaga|quinoa|nabo)',
            r'(?:malezas? de hoja ancha|malezas? de hoja angosta|gramíneas)',
            r'(?:brassica|lolium|avena|convolvulus|chenopodium|portulaca)'
        ]
        
        # 5. PATRONES PARA MOMENTO DE APLICACIÓN
        timing_patterns = [
            r'(?:aplicar|aplicación)\s+([^\n\.]+?)(?:antes|después|durante)',
            r'(?:preemergente|pre-emergente|postemergente|post-emergente)',
            r'(?:antes de la emergencia|después de la emergencia)',
            r'(?:estado cotiledonar|2-4 hojas|roseta)',
            r'momento de aplicación:\s*([^\n]+)'
        ]
        
        # 6. PATRONES PARA PRECAUCIONES
        precaution_patterns = [
            r'(?:precaución|advertencia|cuidado):\s*([^\n]+)',
            r'(?:no aplicar|evitar|prohibido)\s+([^\n\.]+)',
            r'(?:riesgo|peligro|toxico?)\s+([^\n\.]+)',
            r'(?:período de carencia|intervalo de seguridad):\s*(\d+)\s*días?'
        ]
        
        # 7. PATRONES PARA CULTIVOS
        crop_patterns = [
            r'(?:cultivos?|en)\s+(lentejas?|leguminosas?|arveja|garbanzo|poroto)',
            r'registrado para:\s*([^\n]+)',
            r'uso en:\s*([^\n]+)'
        ]
        
        # EXTRAER NOMBRES DE PRODUCTOS
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name = match.strip() if isinstance(match, str) else match[0].strip()
                
                # FILTROS MEJORADOS para nombres válidos
                if self._is_valid_herbicide_name(name) and name not in [h.name for h in herbicides]:
                    herbicide = HerbicideInfo(name=name, source_file=source_file)
                    
                    # EXTRAER FORMULACIÓN
                    formulation_match = re.search(r'(WP|EC|SC|SL|WG)', name, re.IGNORECASE)
                    if formulation_match:
                        herbicide.formulation = formulation_match.group(1).upper()
                    
                    herbicides.append(herbicide)
        
        # Si no encontramos nombres, extraer del nombre del archivo
        if not herbicides:
            file_based_name = self._extract_name_from_filename(source_file)
            if file_based_name:
                herbicides.append(HerbicideInfo(name=file_based_name, source_file=source_file))
        
        # SIEMPRE añadir el nombre principal del archivo como herbicida principal
        main_herbicide = self._extract_name_from_filename(source_file)
        if main_herbicide and main_herbicide not in [h.name for h in herbicides]:
            # Insertar al principio como herbicida principal
            main_herb = HerbicideInfo(name=main_herbicide, source_file=source_file)
            herbicides.insert(0, main_herb)
        
        # ENRIQUECER CADA HERBICIDA CON INFORMACIÓN COMPLETA
        for herbicide in herbicides:
            
            # PRINCIPIO ACTIVO
            for pattern in active_ingredient_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not herbicide.active_ingredient:
                    if isinstance(matches[0], tuple):
                        herbicide.active_ingredient = f"{matches[0][0]} {matches[0][1]}%"
                    else:
                        herbicide.active_ingredient = matches[0].strip()
                    break
            
            # DOSIS - Mejorado para capturar dosis de tablas correctamente
            for pattern in dose_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not herbicide.dose_range:
                    if isinstance(matches[0], tuple):
                        if len(matches[0]) >= 3:
                            herbicide.dose_range = f"{matches[0][0]}-{matches[0][1]}"
                            herbicide.dose_unit = matches[0][2]
                        elif len(matches[0]) == 2:
                            # Verificar si es un rango (ej: 1,0 - 2,0)
                            if any(sep in text for sep in ['-', 'a']):
                                herbicide.dose_range = f"{matches[0][0]} - {matches[0][1]}"
                            else:
                                herbicide.dose_range = str(matches[0][0])
                                herbicide.dose_unit = str(matches[0][1])
                        else:
                            herbicide.dose_range = str(matches[0][0])
                    else:
                        herbicide.dose_range = str(matches[0])
                    break
            
            # BUSCAR DOSIS ESPECÍFICAS EN CONTEXTO DE TABLA
            if not herbicide.dose_range:
                # Buscar patrones específicos para dosis en tablas
                table_dose_patterns = [
                    r'(\d+,\d+)\s*-\s*(\d+,\d+)',  # 1,0 - 2,0
                    r'(\d+\.\d+)\s*-\s*(\d+\.\d+)',  # 1.0 - 2.0
                    r'(\d+,\d+)',  # 1,5
                    r'(\d+\.\d+)',  # 1.5
                ]
                
                for pattern in table_dose_patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                            herbicide.dose_range = f"{matches[0][0]} - {matches[0][1]}"
                        else:
                            herbicide.dose_range = str(matches[0])
                        break
            
            # DETECTAR UNIDADES DEL CONTEXTO
            if not herbicide.dose_unit:
                # Buscar unidades en el contexto
                unit_patterns = [
                    r'g/100\s*L\s*de\s*agua',
                    r'g/100L',
                    r'kg/ha',
                    r'g/ha',
                    r'L/ha',
                    r'ml/ha'
                ]
                
                for pattern in unit_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        herbicide.dose_unit = re.search(pattern, text, re.IGNORECASE).group(0)
                        break
            
            # MALEZAS CONTROLADAS
            for pattern in weed_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    weed = match.strip() if isinstance(match, str) else str(match)
                    if weed and weed not in herbicide.controlled_weeds:
                        herbicide.controlled_weeds.append(weed)
            
            # MOMENTO DE APLICACIÓN
            for pattern in timing_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not herbicide.application_timing:
                    herbicide.application_timing = matches[0].strip() if isinstance(matches[0], str) else str(matches[0])
                    break
            
            # PRECAUCIONES
            for pattern in precaution_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    precaution = match.strip() if isinstance(match, str) else str(match)
                    if precaution and precaution not in herbicide.precautions:
                        herbicide.precautions.append(precaution)
            
            # CULTIVOS
            for pattern in crop_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    crop = match.strip() if isinstance(match, str) else str(match)
                    if crop and crop not in herbicide.crops:
                        herbicide.crops.append(crop)
            
            # VOLUMEN DE AGUA
            water_match = re.search(r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*L?\s*(?:de\s*)?agua/ha', text, re.IGNORECASE)
            if water_match and not herbicide.water_volume:
                herbicide.water_volume = water_match.group(0)
            
            # DETERMINAR ETAPA DEL CULTIVO
            if 'preemergente' in text_lower or 'pre-emergente' in text_lower:
                herbicide.crop_stage = 'Preemergente'
            elif 'postemergente' in text_lower or 'post-emergente' in text_lower:
                herbicide.crop_stage = 'Postemergente'
            elif 'cotiledonar' in text_lower:
                herbicide.crop_stage = 'Postemergente temprano'
        
        # MEJORAR DOSIS ESPECÍFICAS POR CULTIVO
        herbicides = self._improve_dose_by_crop_context(herbicides, text)
        
        return herbicides
    
    def _improve_dose_by_crop_context(self, herbicides: List[HerbicideInfo], text: str) -> List[HerbicideInfo]:
        """Mejora las dosis buscando contexto específico por cultivo"""
        lines = text.split('\n')
        
        # Buscar líneas con cultivos específicos
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Si encuentra arveja o lenteja
            if ('arveja' in line_lower or 'lenteja' in line_lower) and len(line_lower) < 50:
                # Buscar dosis en las siguientes líneas
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    
                    # Buscar patrones de dosis específicos
                    dose_patterns = [
                        r'(\d+,\d+)\s*-\s*(\d+,\d+)',  # 1,0 - 2,0
                        r'(\d+\.\d+)\s*-\s*(\d+\.\d+)',  # 1.0 - 2.0
                        r'(\d+,\d+)',  # 1,5
                        r'(\d+\.\d+)',  # 1.5
                    ]
                    
                    for pattern in dose_patterns:
                        matches = re.findall(pattern, next_line)
                        if matches:
                            # Actualizar dosis para todos los herbicidas
                            for herbicide in herbicides:
                                if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                                    herbicide.dose_range = f"{matches[0][0]} - {matches[0][1]}"
                                else:
                                    herbicide.dose_range = str(matches[0])
                                
                                # Buscar unidades en el contexto cercano
                                context = ' '.join(lines[max(0, i-2):min(len(lines), i+5)])
                                if 'g/100' in context:
                                    herbicide.dose_unit = 'g/100 L de agua'
                                elif 'kg/ha' in context:
                                    herbicide.dose_unit = 'kg/ha'
                            
                            return herbicides
        
        return herbicides
    
    def _is_valid_herbicide_name(self, name: str) -> bool:
        """Valida si un nombre es realmente un herbicida válido"""
        if not name or len(name) < 3:
            return False
        
        name_lower = name.lower()
        
        # Filtrar palabras que NO son nombres de herbicidas (expandido)
        invalid_words = [
            'página', 'tabla', 'figura', 'anexo', 'realizar', 'cuando', 'altura',
            'malezas', 'con', 'tenga', 'aplicar', 'usar', 'dosis', 'cultivo',
            'control', 'de', 'la', 'el', 'en', 'para', 'por', 'se', 'es',
            'instrucciones', 'uso', 'etiqueta', 'producto', 'herbicida',
            'antes', 'usar', 'lote', 'son', 'verdaderas', 'exactas', 'informaci',
            'complementaria', 'fabricante', 'garantiza', 'calidad', 'porcentaje',
            'ingrediente', 'activo', 'momento', 'sustrae', 'control', 'directo',
            'marca', 'registrada', 'adama', 'agan', 'ltd', 'colombia', 'bogotá'
        ]
        
        # Si el nombre contiene solo palabras inválidas, rechazar
        words = name_lower.split()
        if all(word in invalid_words for word in words):
            return False
        
        # Si empieza con número o palabras comunes, rechazar
        if re.match(r'^\d+', name) or name_lower.startswith(('cm', 'mm', 'kg', 'g', 'l', 'ml')):
            return False
        
        # Si es muy corto y no tiene formulación, rechazar
        if len(name) < 5 and not re.search(r'(WP|EC|SC|SL|WG)', name, re.IGNORECASE):
            return False
        
        # Si contiene muchos números sin formulación, rechazar
        if len(re.findall(r'\d+', name)) > 2 and not re.search(r'(WP|EC|SC|SL|WG)', name, re.IGNORECASE):
            return False
        
        # Rechazar frases completas o fragmentos obvios
        if any(phrase in name_lower for phrase in [
            'antes de usar', 'lote n', 'son verdaderas', 'exactas', 'informaci',
            'del producto', 'fabricante', 'garantiza', 'porcentaje', 'ingrediente activo',
            'momento que', 'sustrae', 'control directo', 'marca registrada'
        ]):
            return False
        
        # Si tiene más de 4 palabras y no tiene formulación, probablemente es una frase
        if len(name.split()) > 4 and not re.search(r'(WP|EC|SC|SL|WG)', name, re.IGNORECASE):
            return False
        
        return True
    
    def _extract_name_from_filename(self, filename: str) -> str:
        """Extrae el nombre del herbicida del nombre del archivo"""
        # Remover extensión y limpiar
        base_name = filename.replace('.pdf', '')
        
        # Mapeo de nombres de archivos conocidos
        known_herbicides = {
            'afalon': 'AFALON 50 WP',
            'linurex': 'LINUREX 50 WP', 
            'agil': 'AGIL 100 EC',
            'centurion': 'CENTURION',
            'cletodim': 'CLETODIM 240 EC',
            'ripper': 'RIPPER',
            'spectro': 'SPECTRO 33 EC',
            'treflan': 'TREFLAN',
            'fortaleza': 'FORTALEZA 24.25 EC',
            'glifoglex': 'GLIFOGLEX 480 SL',
            'linuron': 'LINURON 500 SC',
            'rangoclan': 'RANGOCLAN 75 WG',
            'tiburon': 'TIBURON',
            'triflurex': 'TRIFLUREX 48 EC',
            'touchdown': 'TOUCHDOWN IQ 500 SL',
            'vesuvius': 'VESUVIUS',
            'aquiles': 'AQUILES 20 EC',
            'assure': 'ASSURE PRO',
            'flecha': 'FLECHA 9.6 EC',
            'pendiclan': 'PENDICLAN 33 EC',
            'rango': 'RANGO 480 SL'
        }
        
        # Buscar en nombres conocidos
        base_lower = base_name.lower()
        for key, full_name in known_herbicides.items():
            if key in base_lower:
                return full_name
        
        # Si no está en la lista, limpiar y formatear
        # Remover caracteres especiales y fechas
        clean_name = re.sub(r'[_\-\d]{4,}.*$', '', base_name)  # Remover fechas y códigos
        clean_name = re.sub(r'[_\-]+', ' ', clean_name)  # Reemplazar guiones con espacios
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()  # Normalizar espacios
        
        # Capitalizar apropiadamente
        if clean_name:
            return clean_name.upper()
        
        return None
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extrae tablas preservando estructura exacta por fila"""
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extraer tablas con configuración óptima
                    tables = page.extract_tables()
                    
                    for table_num, table in enumerate(tables, 1):
                        if table and len(table) > 1:  # Al menos header + 1 fila
                            # Procesar tabla manteniendo estructura exacta
                            structured_chunks = self._process_table_by_rows(table, page_num, table_num)
                            tables_data.extend(structured_chunks)
                            
        except Exception as e:
            logger.error(f"Error extrayendo tablas de {pdf_path}: {e}")
        
        return tables_data
    
    def _process_table_by_rows(self, table: List, page_num: int, table_num: int) -> List[Dict[str, Any]]:
        """Procesa tabla creando un chunk por cada fila de datos"""
        chunks = []
        
        if not table or len(table) < 2:
            return chunks
        
        # Extraer header
        headers = [str(cell).strip() if cell else "" for cell in table[0]]
        
        # Identificar columnas importantes
        cultivo_col = self._find_column_index(headers, ['cultivo', 'crop'])
        maleza_col = self._find_column_index(headers, ['maleza', 'malezas', 'weed'])
        dosis_col = self._find_column_index(headers, ['dosis', 'dose', 'cantidad'])
        obs_col = self._find_column_index(headers, ['observaciones', 'observación', 'instrucciones'])
        
        # Procesar cada fila de datos
        for row_idx, row in enumerate(table[1:], 1):
            if row and any(cell for cell in row if cell):
                # Crear chunk estructurado para esta fila específica
                chunk_text = self._create_structured_row_chunk(
                    headers, row, row_idx, page_num, table_num,
                    cultivo_col, maleza_col, dosis_col, obs_col
                )
                
                # Extraer metadata específica de esta fila
                row_metadata = self._extract_row_metadata(
                    headers, row, cultivo_col, maleza_col, dosis_col
                )
                
                chunks.append({
                    "page": page_num,
                    "table_num": table_num,
                    "row_num": row_idx,
                    "text": chunk_text,
                    "metadata": row_metadata,
                    "type": "table_row",
                    "raw_row": row,
                    "headers": headers
                })
        
        return chunks
    
    def _find_column_index(self, headers: List[str], keywords: List[str]) -> int:
        """Encuentra el índice de una columna basado en palabras clave"""
        for i, header in enumerate(headers):
            if header:
                header_lower = header.lower()
                for keyword in keywords:
                    if keyword in header_lower:
                        return i
        return -1
    
    def _create_structured_row_chunk(self, headers: List[str], row: List, row_idx: int, 
                                    page_num: int, table_num: int, cultivo_col: int, 
                                    maleza_col: int, dosis_col: int, obs_col: int) -> str:
        """Crea un chunk estructurado para una fila específica"""
        
        # Extraer datos de la fila
        cultivo = str(row[cultivo_col]).strip() if cultivo_col >= 0 and cultivo_col < len(row) and row[cultivo_col] else ""
        maleza = str(row[maleza_col]).strip() if maleza_col >= 0 and maleza_col < len(row) and row[maleza_col] else ""
        dosis = str(row[dosis_col]).strip() if dosis_col >= 0 and dosis_col < len(row) and row[dosis_col] else ""
        observaciones = str(row[obs_col]).strip() if obs_col >= 0 and obs_col < len(row) and row[obs_col] else ""
        
        # Crear texto estructurado
        chunk_text = f"=== REGISTRO DE APLICACIÓN - PÁGINA {page_num} ===\n\n"
        
        if cultivo:
            chunk_text += f"CULTIVO: {cultivo}\n"
        
        if maleza:
            chunk_text += f"MALEZAS CONTROLADAS: {maleza}\n"
        
        if dosis:
            chunk_text += f"DOSIS: {dosis}\n"
            # Detectar unidades del header y del contenido
            dosis_header = headers[dosis_col] if dosis_col >= 0 and dosis_col < len(headers) else ""
            
            # Buscar unidades en header
            if 'g/100' in dosis_header:
                chunk_text += f"UNIDAD: g/100 L de agua\n"
            elif 'kg/ha' in dosis_header or 'kg/há' in dosis_header:
                chunk_text += f"UNIDAD: kg/ha\n"
            elif 'l/ha' in dosis_header.lower() or 'l/há' in dosis_header.lower():
                chunk_text += f"UNIDAD: L/ha\n"
            elif 'g/ha' in dosis_header or 'g/há' in dosis_header:
                chunk_text += f"UNIDAD: g/ha\n"
            else:
                # Buscar unidades en el contenido de la fila
                row_text = ' '.join(str(cell) for cell in row if cell)
                if 'l/há' in row_text.lower() or 'l/ha' in row_text.lower():
                    chunk_text += f"UNIDAD: L/ha\n"
                elif 'kg/há' in row_text.lower() or 'kg/ha' in row_text.lower():
                    chunk_text += f"UNIDAD: kg/ha\n"
                elif 'g/há' in row_text.lower() or 'g/ha' in row_text.lower():
                    chunk_text += f"UNIDAD: g/ha\n"
        
        if observaciones:
            chunk_text += f"\nINSTRUCCIONES DE APLICACIÓN:\n{observaciones}\n"
        
        # Añadir información completa de la fila
        chunk_text += f"\nDATOS COMPLETOS DE LA FILA:\n"
        for i, (header, cell) in enumerate(zip(headers, row)):
            if header and cell:
                chunk_text += f"{header}: {str(cell).strip()}\n"
        
        chunk_text += f"\nUBICACIÓN: Página {page_num}, Tabla {table_num}, Fila {row_idx}\n"
        
        return chunk_text
    
    def _extract_row_metadata(self, headers: List[str], row: List, cultivo_col: int, 
                            maleza_col: int, dosis_col: int) -> Dict[str, Any]:
        """Extrae metadata específica de una fila"""
        metadata = {
            "type": "table_row",
            "has_dose_info": False,
            "has_crop_info": False,
            "has_weed_info": False,
            "dosis_especifica": "",
            "unidad_dosis": ""
        }
        
        # Extraer cultivo
        if cultivo_col >= 0 and cultivo_col < len(row) and row[cultivo_col]:
            cultivo_text = str(row[cultivo_col]).strip()
            if cultivo_text:
                metadata["has_crop_info"] = True
                # Separar múltiples cultivos
                cultivos = [c.strip() for c in cultivo_text.replace(',', '\n').split('\n') if c.strip()]
                metadata["cultivos"] = ", ".join(cultivos)  # Convertir lista a string
        
        # Extraer malezas
        if maleza_col >= 0 and maleza_col < len(row) and row[maleza_col]:
            maleza_text = str(row[maleza_col]).strip()
            if maleza_text:
                metadata["has_weed_info"] = True
                # Extraer malezas específicas
                malezas = self._extract_specific_weeds(maleza_text)
                metadata["malezas"] = ", ".join(malezas)  # Convertir lista a string
        
        # Extraer dosis
        if dosis_col >= 0 and dosis_col < len(row) and row[dosis_col]:
            dosis_text = str(row[dosis_col]).strip()
            if dosis_text:
                metadata["has_dose_info"] = True
                metadata["dosis_especifica"] = dosis_text
                
                # Extraer unidad del header y contenido
                if dosis_col < len(headers):
                    header = headers[dosis_col]
                    if 'g/100' in header:
                        metadata["unidad_dosis"] = "g/100 L de agua"
                    elif 'kg/ha' in header or 'kg/há' in header:
                        metadata["unidad_dosis"] = "kg/ha"
                    elif 'l/ha' in header.lower() or 'l/há' in header.lower():
                        metadata["unidad_dosis"] = "L/ha"
                    elif 'g/ha' in header or 'g/há' in header:
                        metadata["unidad_dosis"] = "g/ha"
                    else:
                        # Buscar en el contenido de la fila
                        row_text = ' '.join(str(cell) for cell in row if cell)
                        if 'l/há' in row_text.lower() or 'l/ha' in row_text.lower():
                            metadata["unidad_dosis"] = "L/ha"
                        elif 'kg/há' in row_text.lower() or 'kg/ha' in row_text.lower():
                            metadata["unidad_dosis"] = "kg/ha"
                        elif 'g/há' in row_text.lower() or 'g/ha' in row_text.lower():
                            metadata["unidad_dosis"] = "g/ha"
        
        return metadata
    
    def _extract_specific_weeds(self, maleza_text: str) -> List[str]:
        """Extrae malezas específicas del texto"""
        malezas = []
        
        # Lista de malezas conocidas
        known_weeds = [
            'yuyo', 'ballica', 'avena guacha', 'correhuela', 'chepica', 'pata de gallina',
            'verdolaga', 'quinoa', 'nabo', 'quilloy-quilloy', 'mostaza', 'bledo',
            'rábano', 'ambrosia', 'duraznillo', 'hualcacho', 'falaris', 'tomatillo',
            'sanguinaria', 'bolsita del pastor', 'manzanilla', 'ortiga', 'altamisa',
            'cúscuta amarilla', 'senecio', 'sclerantus', 'enredadera', 'pega-pega',
            'festuca', 'gramíneas', 'hoja ancha', 'hoja angosta'
        ]
        
        maleza_lower = maleza_text.lower()
        for weed in known_weeds:
            if weed in maleza_lower:
                malezas.append(weed)
        
        return malezas
    
    def _extract_descriptive_info(self, full_text: str, filename: str) -> str:
        """Extrae información descriptiva del herbicida (no tabular)"""
        
        # Dividir texto en secciones
        lines = full_text.split('\n')
        descriptive_sections = []
        
        # Secciones importantes a extraer
        important_sections = [
            'composición', 'principio activo', 'descripción', 'modo de acción',
            'precauciones', 'advertencias', 'restricciones', 'compatibilidad',
            'almacenamiento', 'fabricado por', 'importa', 'autorización'
        ]
        
        current_section = ""
        capturing = False
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            line_lower = line_clean.lower()
            
            # Detectar inicio de sección importante
            if any(section in line_lower for section in important_sections):
                capturing = True
                current_section = f"\n=== {line_clean.upper()} ===\n"
                descriptive_sections.append(current_section)
                continue
            
            # Detectar si es parte de tabla (saltar)
            if self._is_table_content(line_clean):
                capturing = False
                continue
            
            # Capturar contenido descriptivo
            if capturing and len(line_clean) > 10:
                descriptive_sections.append(line_clean + '\n')
        
        # Crear chunk descriptivo estructurado
        if descriptive_sections:
            herbicide_name = self._extract_name_from_filename(filename)
            
            descriptive_text = f"=== INFORMACIÓN DESCRIPTIVA - {herbicide_name} ===\n\n"
            descriptive_text += "".join(descriptive_sections)
            
            # Añadir información específica extraída
            composition = self._extract_composition(full_text)
            if composition:
                descriptive_text += f"\n=== COMPOSICIÓN DETALLADA ===\n{composition}\n"
            
            precautions = self._extract_precautions(full_text)
            if precautions:
                descriptive_text += f"\n=== PRECAUCIONES Y ADVERTENCIAS ===\n{precautions}\n"
            
            return descriptive_text
        
        return ""
    
    def _is_table_content(self, line: str) -> bool:
        """Detecta si una línea es contenido de tabla"""
        # Patrones que indican contenido tabular
        table_indicators = [
            '|', 'CULTIVO', 'DOSIS', 'MALEZAS', 'OBSERVACIONES',
            'kg/ha', 'l/ha', 'g/ha', 'L/há', 'kg/há', 'g/há'
        ]
        
        return any(indicator in line for indicator in table_indicators)
    
    def _extract_composition(self, text: str) -> str:
        """Extrae información de composición"""
        lines = text.split('\n')
        composition_lines = []
        capturing = False
        
        for line in lines:
            line_clean = line.strip()
            if 'composición' in line_clean.lower():
                capturing = True
                continue
            
            if capturing:
                if line_clean and not self._is_table_content(line_clean):
                    if any(end_marker in line_clean.lower() for end_marker in ['lote', 'fecha', 'autorización']):
                        break
                    composition_lines.append(line_clean)
                elif not line_clean:
                    continue
                else:
                    break
        
        return '\n'.join(composition_lines) if composition_lines else ""
    
    def _extract_precautions(self, text: str) -> str:
        """Extrae precauciones y advertencias"""
        lines = text.split('\n')
        precaution_lines = []
        capturing = False
        
        for line in lines:
            line_clean = line.strip()
            if any(keyword in line_clean.lower() for keyword in ['precauciones', 'advertencias', 'restricciones']):
                capturing = True
                continue
            
            if capturing:
                if line_clean and not self._is_table_content(line_clean):
                    if len(line_clean) > 200:  # Líneas muy largas probablemente son de tabla
                        break
                    precaution_lines.append(line_clean)
                elif not line_clean:
                    continue
                else:
                    break
        
        return '\n'.join(precaution_lines) if precaution_lines else ""
    
    def _create_comprehensive_chunks(self, full_text: str, filename: str) -> List[Dict[str, Any]]:
        """Crea chunks comprensivos con TODA la información del archivo"""
        chunks = []
        herbicide_name = self._extract_name_from_filename(filename)
        
        # 1. CHUNK DE INFORMACIÓN GENERAL DEL HERBICIDA
        general_info = self._extract_general_herbicide_info(full_text, herbicide_name)
        if general_info:
            chunks.append({
                "text": general_info,
                "metadata": {
                    "source_file": filename,
                    "type": "general_info",
                    "herbicide_principal": herbicide_name,
                    "has_composition": "composición" in full_text.lower(),
                    "has_description": "herbicida" in full_text.lower(),
                    "content_category": "product_description"
                }
            })
        
        # 2. CHUNK DE COMPOSICIÓN Y PRINCIPIO ACTIVO
        composition_info = self._extract_detailed_composition(full_text, herbicide_name)
        if composition_info:
            chunks.append({
                "text": composition_info,
                "metadata": {
                    "source_file": filename,
                    "type": "composition",
                    "herbicide_principal": herbicide_name,
                    "has_active_ingredient": True,
                    "content_category": "technical_specs"
                }
            })
        
        # 3. CHUNK DE PRECAUCIONES Y SEGURIDAD
        safety_info = self._extract_safety_info(full_text, herbicide_name)
        if safety_info:
            chunks.append({
                "text": safety_info,
                "metadata": {
                    "source_file": filename,
                    "type": "safety_info",
                    "herbicide_principal": herbicide_name,
                    "has_precautions": True,
                    "has_restrictions": True,
                    "content_category": "safety_guidelines"
                }
            })
        
        # 4. CHUNK DE INFORMACIÓN COMERCIAL Y LEGAL
        commercial_info = self._extract_commercial_info(full_text, herbicide_name)
        if commercial_info:
            chunks.append({
                "text": commercial_info,
                "metadata": {
                    "source_file": filename,
                    "type": "commercial_info",
                    "herbicide_principal": herbicide_name,
                    "has_manufacturer": True,
                    "has_authorization": True,
                    "content_category": "commercial_legal"
                }
            })
        
        # 5. CHUNK DE TEXTO COMPLETO (para búsquedas generales)
        full_document_chunk = self._create_full_document_chunk(full_text, herbicide_name, filename)
        chunks.append(full_document_chunk)
        
        return chunks
    
    def _extract_general_herbicide_info(self, text: str, herbicide_name: str) -> str:
        """Extrae información general del herbicida"""
        lines = text.split('\n')
        info_lines = []
        
        # Buscar descripción inicial del herbicida
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if herbicide_name.lower() in line_clean.lower() and 'herbicida' in line_clean.lower():
                # Capturar esta línea y las siguientes que sean descriptivas
                info_lines.append(line_clean)
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not self._is_table_content(next_line) and len(next_line) > 20:
                        if any(end_marker in next_line.lower() for end_marker in ['composición', 'cultivo', 'dosis']):
                            break
                        info_lines.append(next_line)
                break
        
        if info_lines:
            return f"=== INFORMACIÓN GENERAL - {herbicide_name} ===\n\n" + '\n'.join(info_lines)
        return ""
    
    def _extract_detailed_composition(self, text: str, herbicide_name: str) -> str:
        """Extrae composición detallada"""
        composition = self._extract_composition(text)
        if composition:
            return f"=== COMPOSICIÓN Y PRINCIPIO ACTIVO - {herbicide_name} ===\n\n{composition}"
        return ""
    
    def _extract_safety_info(self, text: str, herbicide_name: str) -> str:
        """Extrae información de seguridad completa"""
        lines = text.split('\n')
        safety_lines = []
        capturing = False
        
        safety_keywords = ['precauciones', 'advertencias', 'restricciones', 'no aplicar', 'evitar', 'peligro']
        
        for line in lines:
            line_clean = line.strip()
            if any(keyword in line_clean.lower() for keyword in safety_keywords):
                capturing = True
                safety_lines.append(line_clean)
                continue
            
            if capturing:
                if line_clean and not self._is_table_content(line_clean):
                    if any(end_marker in line_clean.lower() for end_marker in ['fabricado', 'importa', 'autorización']):
                        break
                    safety_lines.append(line_clean)
                elif not line_clean:
                    continue
                else:
                    break
        
        if safety_lines:
            return f"=== PRECAUCIONES Y SEGURIDAD - {herbicide_name} ===\n\n" + '\n'.join(safety_lines)
        return ""
    
    def _extract_commercial_info(self, text: str, herbicide_name: str) -> str:
        """Extrae información comercial y legal"""
        lines = text.split('\n')
        commercial_lines = []
        
        commercial_keywords = ['fabricado', 'importa', 'distribuye', 'autorización', 'servicio agrícola', 'marca registrada']
        
        for line in lines:
            line_clean = line.strip()
            if any(keyword in line_clean.lower() for keyword in commercial_keywords):
                commercial_lines.append(line_clean)
        
        if commercial_lines:
            return f"=== INFORMACIÓN COMERCIAL Y LEGAL - {herbicide_name} ===\n\n" + '\n'.join(commercial_lines)
        return ""
    
    def _create_full_document_chunk(self, text: str, herbicide_name: str, filename: str) -> Dict[str, Any]:
        """Crea chunk con documento completo para búsquedas generales"""
        return {
            "text": f"=== DOCUMENTO COMPLETO - {herbicide_name} ===\n\n{text}",
            "metadata": {
                "source_file": filename,
                "type": "full_document",
                "herbicide_principal": herbicide_name,
                "is_complete_document": True,
                "content_category": "complete_reference"
            }
        }
    
    def _deduplicate_tables(self, tables: List) -> List:
        """Elimina tablas duplicadas"""
        unique_tables = []
        seen_signatures = set()
        
        for table in tables:
            if not table:
                continue
                
            # Crear signature de la tabla
            signature = str(len(table)) + "_" + str(len(table[0]) if table[0] else 0)
            if table and table[0]:
                signature += "_" + "_".join(str(cell)[:10] if cell else "" for cell in table[0][:3])
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_tables.append(table)
        
        return unique_tables
    
    def _is_dose_table(self, table: List) -> bool:
        """Detecta si es una tabla de dosis/aplicación"""
        if not table:
            return False
            
        table_text = " ".join(" ".join(str(cell) if cell else "" for cell in row) for row in table).lower()
        
        dose_indicators = [
            'dosis', 'kg/ha', 'l/ha', 'ml/ha', 'aplicación', 'volumen',
            'concentración', 'cantidad', 'litros', 'gramos', 'hectarea'
        ]
        
        return sum(1 for indicator in dose_indicators if indicator in table_text) >= 2
    
    def _interpret_table_row(self, headers: List[str], row_data: List[str]) -> str:
        """Interpreta una fila de tabla para extraer información crítica"""
        if not headers or not row_data:
            return ""
        
        interpretation = []
        
        for header, data in zip(headers, row_data):
            if not header or not data:
                continue
                
            header_lower = header.lower()
            data_clean = data.strip()
            
            # Detectar dosis
            if any(word in header_lower for word in ['dosis', 'cantidad', 'aplicar']):
                if any(unit in data_clean.lower() for unit in ['kg/ha', 'l/ha', 'ml/ha', 'g/ha']):
                    interpretation.append(f"Dosis: {data_clean}")
            
            # Detectar cultivo
            if any(word in header_lower for word in ['cultivo', 'crop']):
                interpretation.append(f"Cultivo: {data_clean}")
            
            # Detectar maleza
            if any(word in header_lower for word in ['maleza', 'weed', 'control']):
                interpretation.append(f"Maleza: {data_clean}")
        
        return " | ".join(interpretation) if interpretation else ""
    
    def _extract_critical_table_info(self, table: List) -> str:
        """Extrae información crítica de toda la tabla con patrones mejorados"""
        if not table:
            return ""
        
        critical_info = []
        
        # Patrones mejorados para dosis en tablas
        dose_patterns = [
            r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*(kg/ha|l/ha|ml/ha|g/ha)',
            r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*(kg|g|l|ml)\s*/\s*há',
            r'(\d+(?:[,\.]\d+)?)\s*-\s*(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)',
            r'(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)\s*/\s*ha',
            r'(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)\s*por\s*hectárea'
        ]
        
        # Buscar dosis en cada celda de la tabla
        for row_idx, row in enumerate(table):
            if not row:
                continue
                
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue
                    
                cell_text = str(cell).strip()
                
                # Aplicar todos los patrones de dosis
                for pattern in dose_patterns:
                    doses = re.findall(pattern, cell_text, re.IGNORECASE)
                    for dose in doses:
                        if isinstance(dose, tuple):
                            if len(dose) >= 2:
                                dose_info = f"Dosis: {dose[0]} {dose[-1]} (Fila {row_idx+1}, Col {col_idx+1})"
                            else:
                                dose_info = f"Dosis: {dose[0]} (Fila {row_idx+1}, Col {col_idx+1})"
                        else:
                            dose_info = f"Dosis: {dose} (Fila {row_idx+1}, Col {col_idx+1})"
                        
                        if dose_info not in critical_info:
                            critical_info.append(dose_info)
        
        # Buscar cultivos mencionados en toda la tabla
        crop_pattern = r'\b(lentejas?|leguminosas?|arveja|garbanzo|poroto|ajo|apio|cebolla|espárrago|maravilla|lupino|papa|zanahoria)\b'
        table_text = " ".join(" ".join(str(cell) if cell else "" for cell in row) for row in table)
        
        crops = re.findall(crop_pattern, table_text, re.IGNORECASE)
        if crops:
            unique_crops = list(set([crop.lower() for crop in crops]))
            critical_info.append(f"Cultivos mencionados: {', '.join(unique_crops)}")
        
        # Buscar malezas mencionadas
        weed_pattern = r'\b(yuyo|ballica|avena guacha|correhuela|chepica|pata de gallina|verdolaga|quinoa|nabo|gramíneas|hoja ancha|hoja angosta)\b'
        weeds = re.findall(weed_pattern, table_text, re.IGNORECASE)
        if weeds:
            unique_weeds = list(set([weed.lower() for weed in weeds]))
            critical_info.append(f"Malezas mencionadas: {', '.join(unique_weeds)}")
        
        return "\n".join(critical_info) if critical_info else ""
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Procesa un PDF preservando estructura exacta por fila"""
        chunks = []
        filename = os.path.basename(pdf_path)
        
        logger.info(f"[RAG Lentejas] Procesando: {filename}")
        
        # 1. Extraer tablas estructuradas por fila
        table_chunks = self.extract_tables_from_pdf(pdf_path)
        
        # Procesar cada chunk de tabla (cada fila)
        for table_chunk in table_chunks:
            # Enriquecer metadata con información del archivo
            metadata = table_chunk.get("metadata", {})
            metadata["source_file"] = filename
            
            # Extraer herbicida principal del nombre del archivo
            main_herbicide = self._extract_name_from_filename(filename)
            if main_herbicide:
                metadata["herbicide_principal"] = main_herbicide
            
            chunks.append({
                "text": table_chunk["text"],
                "metadata": metadata
            })
        
        # 2. SIEMPRE extraer TODA la información adicional del archivo
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += f"\n=== PÁGINA {page_num + 1} ===\n{page_text}"
            
            doc.close()
            
            # Crear chunks adicionales con TODA la información
            additional_chunks = self._create_comprehensive_chunks(full_text, filename)
            chunks.extend(additional_chunks)
                
        except Exception as e:
            logger.error(f"Error procesando texto de {pdf_path}: {e}")
        
        logger.info(f"[RAG Lentejas] ✅ {filename}: {len(chunks)} chunks creados ({len(table_chunks)} de tablas)")
        return chunks
    
    def create_semantic_chunks(self, text: str, source_file: str, chunk_size: int = 1200) -> List[Dict[str, Any]]:
        """Crea chunks semánticos con metadata ULTRA-ENRIQUECIDA"""
        chunks = []
        
        # Dividir por páginas primero
        pages = text.split("=== PÁGINA")
        
        for page_num, page_text in enumerate(pages, 1):
            if not page_text.strip():
                continue
                
            # Extraer información COMPLETA de herbicidas de esta página
            herbicides = self.extract_herbicide_info(page_text, source_file)
            
            # Crear metadata enriquecida
            page_metadata = {
                "source_file": source_file,
                "page_number": page_num,
                "type": "text",
                "herbicides_found": len(herbicides),
                "herbicide_names": [h.name for h in herbicides],
                "active_ingredients": [h.active_ingredient for h in herbicides if h.active_ingredient],
                "formulations": [h.formulation for h in herbicides if h.formulation],
                "dose_ranges": [h.dose_range for h in herbicides if h.dose_range],
                "dose_units": [h.dose_unit for h in herbicides if h.dose_unit],
                "application_timings": [h.application_timing for h in herbicides if h.application_timing],
                "crop_stages": [h.crop_stage for h in herbicides if h.crop_stage],
                "controlled_weeds": [weed for h in herbicides for weed in h.controlled_weeds],
                "target_crops": [crop for h in herbicides for crop in h.crops],
                "water_volumes": [h.water_volume for h in herbicides if h.water_volume],
                "precautions": [prec for h in herbicides for prec in h.precautions],
                "has_dose_info": any(h.dose_range for h in herbicides),
                "has_weed_info": any(h.controlled_weeds for h in herbicides),
                "has_timing_info": any(h.application_timing for h in herbicides),
                "is_preemergent": any('preemergente' in h.crop_stage.lower() for h in herbicides if h.crop_stage),
                "is_postemergent": any('postemergente' in h.crop_stage.lower() for h in herbicides if h.crop_stage),
                "content_score": len(herbicides) * 2.0
            }
            
            chunks.append({
                "text": page_text.strip(),
                "metadata": page_metadata
            })
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Obtiene embedding usando OpenAI API"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json()["data"][0]["embedding"]
                self.embedding_cache[text] = embedding
                return embedding
            else:
                logger.error(f"Error API embedding: {response.status_code}")
                return [0.0] * 1536  # Embedding dummy
                
        except Exception as e:
            logger.error(f"Error obteniendo embedding: {e}")
            return [0.0] * 1536
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Añade documentos a la colección ChromaDB"""
        if not chunks:
            return
        
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"chunk_{i}_{chunk['metadata']['source_file']}" for i, chunk in enumerate(chunks)]
        
        # Obtener embeddings
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        # Añadir a ChromaDB
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        logger.info(f"[RAG Lentejas] ✅ {len(chunks)} chunks añadidos a la base")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Busca documentos relevantes usando embeddings de OpenAI"""
        try:
            # La colección ya tiene configurado OpenAI embeddings
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Formatear resultados
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    formatted_results.append({
                        "text": doc,
                        "metadata": metadata,
                        "similarity": 1 - distance,  # Convertir distancia a similitud
                        "rank": i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda de lentejas: {e}")
            return []
    
    def count(self) -> int:
        """Retorna el número de documentos en la colección"""
        return self.collection.count()

# Función de utilidad para procesar directorio
def process_documents_directory(rag_system: LentilRAGSystem, documents_dir: str = "./documents"):
    """Procesa todos los PDFs en el directorio de documentos"""
    documents_path = Path(documents_dir)
    
    if not documents_path.exists():
        logger.error(f"Directorio {documents_dir} no existe")
        return
    
    pdf_files = list(documents_path.glob("*.pdf"))
    logger.info(f"[RAG Lentejas] Encontrados {len(pdf_files)} archivos PDF")
    
    for pdf_file in pdf_files:
        try:
            chunks = rag_system.process_pdf(str(pdf_file))
            rag_system.add_documents(chunks)
        except Exception as e:
            logger.error(f"Error procesando {pdf_file}: {e}")
    
    total_docs = rag_system.count()
    logger.info(f"[RAG Lentejas] 🎉 Procesamiento completo: {total_docs} documentos en total")


class LentilRAGSystemEnhanced(LentilRAGSystem):
    """Versión mejorada del sistema RAG con capacidades adicionales de extracción"""
    
    def _extract_structured_text_info(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extrae información estructurada de texto cuando las tablas tradicionales fallan"""
        chunks = []
        
        # Buscar patrones de información estructurada en texto corrido
        lines = text.split('\n')
        
        # Patrones específicos para AGIL y documentos similares
        chunk_counter = 0
        
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 100:  # Líneas muy cortas probablemente no tienen info completa
                continue
            
            # Buscar patrón específico de AGIL: Cultivos, Malezas, Dosis, Observaciones
            agil_pattern = r'([A-Za-zÁ-ú\s,]+),\s*(Malezas\s+[^,]+),\s*(\d+[.,]\d+\s*-\s*\d+[.,]\d+)\s*(Realizar\s+[^.]+\.)'
            matches = re.findall(agil_pattern, line_clean, re.IGNORECASE)
            
            for match in matches:
                chunk_counter += 1
                
                # Extraer y limpiar componentes
                cultivos_raw = match[0].strip()
                malezas_raw = match[1].strip()
                dosis_raw = match[2].strip()
                observaciones_raw = match[3].strip()
                
                # Filtrar cultivos relevantes
                cultivos = self._extract_relevant_crops(cultivos_raw)
                
                # Limpiar malezas
                malezas = self._extract_malezas_from_text(malezas_raw)
                
                # Limpiar dosis
                dosis = dosis_raw.replace(',', '.')
                
                if cultivos and dosis:
                    # Crear chunk estructurado
                    chunk_text = f"=== REGISTRO DE APLICACIÓN - PÁGINA {page_num} ===\n\n"
                    chunk_text += f"CULTIVO: {cultivos}\n"
                    chunk_text += f"MALEZAS CONTROLADAS: {malezas}\n"
                    chunk_text += f"DOSIS: {dosis}\n"
                    chunk_text += f"UNIDAD: L/ha\n"
                    chunk_text += f"\nINSTRUCCIONES DE APLICACIÓN:\n{observaciones_raw}\n"
                    chunk_text += f"\nUBICACIÓN: Página {page_num}, Texto estructurado {chunk_counter}\n"
                    
                    # Crear metadata
                    metadata = {
                        "type": "table_row",
                        "source": "structured_text_parsing",
                        "has_dose_info": True,
                        "has_crop_info": True,
                        "has_weed_info": bool(malezas),
                        "cultivos": cultivos,
                        "malezas": malezas,
                        "dosis_especifica": dosis,
                        "unidad_dosis": "L/ha"
                    }
                    
                    chunks.append({
                        "page": page_num,
                        "chunk_num": chunk_counter,
                        "text": chunk_text,
                        "metadata": metadata,
                        "type": "structured_text",
                        "source": "text_parsing"
                    })
        
        return chunks
    
    def _extract_relevant_crops(self, cultivos_raw: str) -> str:
        """Extrae cultivos relevantes para el sistema de lentejas"""
        relevant_crops = []
        
        # Lista de cultivos relevantes
        crop_patterns = [
            r'\b(lupino|poroto|lenteja|arveja|papa|tomate|pimentón|ají|cebolla|ajo|repollo|coliflor|brócoli)\b'
        ]
        
        for pattern in crop_patterns:
            matches = re.findall(pattern, cultivos_raw, re.IGNORECASE)
            relevant_crops.extend([match.title() for match in matches])
        
        return ', '.join(list(set(relevant_crops))) if relevant_crops else ""
    
    def _extract_malezas_from_text(self, malezas_raw: str) -> str:
        """Extrae malezas específicas del texto"""
        # Remover prefijos
        malezas = re.sub(r'^(Malezas\s+(Anuales|Perennes):\s*)', '', malezas_raw, flags=re.IGNORECASE)
        
        # Lista de malezas conocidas
        known_weeds = [
            'avenilla', 'ballica', 'hualcacho', 'pega-pega', 'cola de zorro', 'cebadilla',
            'pata de gallina', 'maicillo', 'chépica', 'chepica', 'pasto cebolla'
        ]
        
        found_weeds = []
        for weed in known_weeds:
            if weed in malezas.lower():
                found_weeds.append(weed.title())
        
        return ', '.join(found_weeds) if found_weeds else malezas.strip()
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Versión mejorada que combina extracción de tablas tradicional + texto estructurado"""
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # 1. Intentar extracción de tablas tradicional (funciona para AFALON)
                    tables = page.extract_tables()
                    
                    if tables:
                        for table_num, table in enumerate(tables, 1):
                            if table and len(table) > 1:  # Al menos header + 1 fila
                                # Procesar tabla manteniendo estructura exacta
                                structured_chunks = self._process_table_by_rows(table, page_num, table_num)
                                tables_data.extend(structured_chunks)
                    
                    # 2. Si no hay tablas o hay pocas, intentar extracción de texto estructurado
                    if not tables or len(tables) < 2:
                        page_text = page.extract_text()
                        if page_text:
                            # Intentar extraer información estructurada del texto (para AGIL)
                            text_chunks = self._extract_structured_text_info(page_text, page_num)
                            tables_data.extend(text_chunks)
                            
        except Exception as e:
            logger.error(f"Error extrayendo tablas de {pdf_path}: {e}")
        
        return tables_data


if __name__ == "__main__":
    # Ejemplo de uso
    rag = LentilRAGSystem()
    
    # Procesar documentos
    process_documents_directory(rag)
    
    # Prueba de búsqueda
    results = rag.search("herbicida lentejas dosis", top_k=5)
    print(f"\nResultados de prueba: {len(results)}")
    for result in results:
        print(f"- {result['metadata']['source_file']} (similitud: {result['similarity']:.2%})")
