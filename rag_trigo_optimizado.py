"""
Sistema RAG Optimizado Específicamente para Trigo
Extracción avanzada de herbicidas, fungicidas e insecticidas con tablas y metadata enriquecida
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
class PesticideInfo:
    """Estructura completa para información de pesticidas para trigo"""
    name: str
    product_type: str = ""  # Herbicida, Fungicida, Insecticida
    active_ingredient: str = ""
    concentration: str = ""
    formulation: str = ""  # WP, EC, SC, SL, WG, etc.
    dose_range: str = ""
    dose_unit: str = ""  # kg/ha, L/ha, g/ha
    application_timing: str = ""
    application_method: str = ""  # Aspersión, incorporación, etc.
    target_pests: List[str] = None  # Malezas, enfermedades o plagas
    controlled_pests: List[str] = None
    crops: List[str] = None
    crop_stage: str = ""  # Preemergente, postemergente, macollaje, espigamiento, etc.
    water_volume: str = ""  # Volumen de agua
    precautions: List[str] = None
    restrictions: List[str] = None
    compatibility: List[str] = None
    carency_period: str = ""  # Período de carencia
    toxicity_class: str = ""
    source_file: str = ""
    
    def __post_init__(self):
        if self.target_pests is None:
            self.target_pests = []
        if self.controlled_pests is None:
            self.controlled_pests = []
        if self.crops is None:
            self.crops = []
        if self.precautions is None:
            self.precautions = []
        if self.restrictions is None:
            self.restrictions = []
        if self.compatibility is None:
            self.compatibility = []

class WheatRAGSystem:
    """Sistema RAG optimizado para pesticidas de trigo"""
    
    def __init__(self, persist_directory: str = "./chroma_trigo_v1"):
        self.persist_directory = persist_directory
        self.embedding_cache = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # Configurar ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Crear colección específica para trigo
        self.collection = self.client.get_or_create_collection(
            name="pesticidas_trigo",
            metadata={"description": "Pesticidas específicos para cultivo de trigo"}
        )
        
        logger.info(f"[RAG Trigo] Inicializado con {self.collection.count()} documentos")
    
    def extract_pesticide_info(self, text: str, source_file: str) -> List[PesticideInfo]:
        """Extrae información COMPLETA de pesticidas del texto"""
        pesticides = []
        text_lower = text.lower()
        
        # 1. EXTRAER NOMBRES DE PRODUCTOS (patrones mejorados para trigo)
        name_patterns = [
            r'([A-Z][A-Z\s]*\d*\s*(?:WP|EC|SC|SL|WG|SE|EW)\b)',
            r'\b([A-Z][A-Z]+)\s*®?\s*\d*\s*%?\s*(?:WP|EC|SC|SL|WG|SE|EW)\b',
            r'Producto:\s*([^\n]+)',
            r'Herbicida:\s*([^\n]+)',
            r'Fungicida:\s*([^\n]+)',
            r'Insecticida:\s*([^\n]+)',
            r'Nombre comercial:\s*([^\n]+)',
            r'ETIQUETA\s+([A-Z][A-Z\s]+\d*)',
        ]
        
        # 2. PATRONES PARA PRINCIPIOS ACTIVOS (específicos para trigo)
        active_ingredient_patterns = [
            r'(?:principio activo|ingrediente activo|p\.a\.|i\.a\.):\s*([^\n]+)',
            r'(?:contiene|composición):\s*([^\n]+)',
            r'([a-z\-]+(?:\s+[a-z\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*%',
            r'([A-Z][a-z\-]+(?:\s+[a-z\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*g/L',
            r'([A-Z][a-z\-]+(?:\s+[a-z\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*g/kg',
            # Patrones específicos para herbicidas de trigo
            r'(glifosato|2,4-D|MCPA|dicamba|metsulfuron|tribenuron|florasulam|pyroxsulam|pinoxaden|clodinafop|fenoxaprop|diclofop|tralkoxydim|clethodim|quizalofop|fluazifop|haloxyfop)\s+(\d+(?:[,\.]\d+)?)\s*(?:%|g/L|g/kg)',
            # Patrones para fungicidas de trigo
            r'(tebuconazole|propiconazole|epoxiconazole|metconazole|prothioconazole|azoxystrobin|pyraclostrobin|trifloxystrobin|boscalid|fluopyram|fluxapyroxad|carbendazim|tiofanato|mancozeb|chlorothalonil)\s+(\d+(?:[,\.]\d+)?)\s*(?:%|g/L|g/kg)',
            # Patrones para insecticidas de trigo
            r'(imidacloprid|thiamethoxam|clothianidin|acetamiprid|thiacloprid|lambda-cyhalothrin|gamma-cyhalothrin|deltamethrin|cypermethrin|bifenthrin|chlorpyrifos|dimethoate|malathion)\s+(\d+(?:[,\.]\d+)?)\s*(?:%|g/L|g/kg)'
        ]
        
        # 3. PATRONES PARA DOSIS (específicos para trigo)
        dose_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(l/ha|g/ha|litros/ha|gramos/ha|ml/ha|kg/ha|cc/ha|lt/ha)',
            r'(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(litros?\s*por\s*hectarea|gramos?\s*por\s*hectarea)',
            r'dosis:?\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(l/ha|g/ha|ml/ha|kg/ha)',
            r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*(?:kg|g|l|ml)\s*/\s*ha',
            r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*(kg|L|g|ml|cc)\s*/\s*ha',
            r'(\d+(?:[,\.]\d+)?)\s*a\s*(\d+(?:[,\.]\d+)?)\s*(kg|L|g|ml)\s*/\s*ha',
            r'(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)\s*/\s*há',
            r'(\d+(?:[,\.]\d+)?)\s*-\s*(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)',
            r'aplicar\s+(\d+(?:[,\.]\d+)?)\s*(kg|L|g|ml)/ha',
        ]
        
        # 4. PATRONES PARA PLAGAS/ENFERMEDADES/MALEZAS DE TRIGO
        pest_patterns = [
            r'(?:controla|control de|eficaz contra):\s*([^\n\.]+)',
            r'(?:malezas?|plagas?|enfermedades?)\s+controladas?:\s*([^\n\.]+)',
            # Malezas de trigo
            r'(?:avena|ballica|alpiste|avenilla|yuyo|bromo|rábano|mostaza|cardo|nabo)',
            # Enfermedades de trigo
            r'(?:roya|septoria|fusarium|oidio|mancha|helmintosporiosis|carbón)',
            # Plagas de trigo
            r'(?:pulgón|afido|trips|gusano|bicho bolita|chinche)',
        ]
        
        # 5. PATRONES PARA MOMENTO DE APLICACIÓN (específico trigo)
        timing_patterns = [
            r'(?:aplicar|aplicación)\s+([^\n\.]+?)(?:antes|después|durante)',
            r'(?:preemergente|pre-emergente|postemergente|post-emergente)',
            r'(?:macollaje|ahijamiento|encañazón|espigamiento|floración|llenado de grano)',
            r'(?:antes de la emergencia|después de la emergencia)',
            r'(?:estado|etapa)\s+([^\n\.]+)',
            r'momento de aplicación:\s*([^\n]+)'
        ]
        
        # 6. PATRONES PARA TIPO DE PRODUCTO
        product_type_patterns = [
            (r'herbicida', 'Herbicida'),
            (r'fungicida', 'Fungicida'),
            (r'insecticida', 'Insecticida'),
            (r'acaricida', 'Acaricida')
        ]
        
        # EXTRAER NOMBRES DE PRODUCTOS
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name = match.strip() if isinstance(match, str) else match[0].strip()
                
                if self._is_valid_pesticide_name(name) and name not in [p.name for p in pesticides]:
                    pesticide = PesticideInfo(name=name, source_file=source_file)
                    
                    # EXTRAER FORMULACIÓN
                    formulation_match = re.search(r'(WP|EC|SC|SL|WG|SE|EW)', name, re.IGNORECASE)
                    if formulation_match:
                        pesticide.formulation = formulation_match.group(1).upper()
                    
                    pesticides.append(pesticide)
        
        # Si no encontramos nombres, extraer del nombre del archivo
        if not pesticides:
            file_based_name = self._extract_name_from_filename(source_file)
            if file_based_name:
                pesticides.append(PesticideInfo(name=file_based_name, source_file=source_file))
        
        # SIEMPRE añadir el nombre principal del archivo
        main_pesticide = self._extract_name_from_filename(source_file)
        if main_pesticide and main_pesticide not in [p.name for p in pesticides]:
            main_pest = PesticideInfo(name=main_pesticide, source_file=source_file)
            pesticides.insert(0, main_pest)
        
        # ENRIQUECER CADA PESTICIDA CON INFORMACIÓN COMPLETA
        for pesticide in pesticides:
            
            # TIPO DE PRODUCTO
            for pattern, product_type in product_type_patterns:
                if re.search(pattern, text_lower):
                    pesticide.product_type = product_type
                    break
            
            # PRINCIPIO ACTIVO
            for pattern in active_ingredient_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not pesticide.active_ingredient:
                    if isinstance(matches[0], tuple):
                        pesticide.active_ingredient = f"{matches[0][0]} {matches[0][1]}%"
                    else:
                        pesticide.active_ingredient = matches[0].strip()
                    break
            
            # DOSIS
            for pattern in dose_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not pesticide.dose_range:
                    if isinstance(matches[0], tuple):
                        if len(matches[0]) >= 3:
                            pesticide.dose_range = f"{matches[0][0]}-{matches[0][1]}"
                            pesticide.dose_unit = matches[0][2]
                        elif len(matches[0]) == 2:
                            if any(sep in text for sep in ['-', 'a']):
                                pesticide.dose_range = f"{matches[0][0]} - {matches[0][1]}"
                            else:
                                pesticide.dose_range = str(matches[0][0])
                                pesticide.dose_unit = str(matches[0][1])
                        else:
                            pesticide.dose_range = str(matches[0][0])
                    else:
                        pesticide.dose_range = str(matches[0])
                    break
            
            # PLAGAS/ENFERMEDADES/MALEZAS CONTROLADAS
            for pattern in pest_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    pest = match.strip() if isinstance(match, str) else str(match)
                    if pest and pest not in pesticide.controlled_pests:
                        pesticide.controlled_pests.append(pest)
            
            # MOMENTO DE APLICACIÓN
            for pattern in timing_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not pesticide.application_timing:
                    pesticide.application_timing = matches[0].strip() if isinstance(matches[0], str) else str(matches[0])
                    break
            
            # CULTIVOS
            if 'trigo' in text_lower:
                pesticide.crops.append('trigo')
            
            # VOLUMEN DE AGUA
            water_match = re.search(r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*L?\s*(?:de\s*)?agua/ha', text, re.IGNORECASE)
            if water_match and not pesticide.water_volume:
                pesticide.water_volume = water_match.group(0)
            
            # DETERMINAR ETAPA DEL CULTIVO (específico trigo)
            if 'preemergente' in text_lower or 'pre-emergente' in text_lower:
                pesticide.crop_stage = 'Preemergente'
            elif 'postemergente' in text_lower or 'post-emergente' in text_lower:
                pesticide.crop_stage = 'Postemergente'
            elif 'macollaje' in text_lower or 'ahijamiento' in text_lower:
                pesticide.crop_stage = 'Macollaje'
            elif 'encañazón' in text_lower or 'encañado' in text_lower:
                pesticide.crop_stage = 'Encañazón'
            elif 'espigamiento' in text_lower or 'espigado' in text_lower:
                pesticide.crop_stage = 'Espigamiento'
            elif 'floración' in text_lower:
                pesticide.crop_stage = 'Floración'
        
        # MEJORAR DOSIS ESPECÍFICAS POR CULTIVO
        pesticides = self._improve_dose_by_crop_context(pesticides, text)
        
        return pesticides
    
    def _improve_dose_by_crop_context(self, pesticides: List[PesticideInfo], text: str) -> List[PesticideInfo]:
        """Mejora las dosis buscando contexto específico por cultivo"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            if 'trigo' in line_lower and len(line_lower) < 50:
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    
                    dose_patterns = [
                        r'(\d+,\d+)\s*-\s*(\d+,\d+)',
                        r'(\d+\.\d+)\s*-\s*(\d+\.\d+)',
                        r'(\d+,\d+)',
                        r'(\d+\.\d+)',
                    ]
                    
                    for pattern in dose_patterns:
                        matches = re.findall(pattern, next_line)
                        if matches:
                            for pesticide in pesticides:
                                if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                                    pesticide.dose_range = f"{matches[0][0]} - {matches[0][1]}"
                                else:
                                    pesticide.dose_range = str(matches[0])
                                
                                context = ' '.join(lines[max(0, i-2):min(len(lines), i+5)])
                                if 'g/100' in context:
                                    pesticide.dose_unit = 'g/100 L de agua'
                                elif 'kg/ha' in context:
                                    pesticide.dose_unit = 'kg/ha'
                                elif 'l/ha' in context.lower():
                                    pesticide.dose_unit = 'L/ha'
                            
                            return pesticides
        
        return pesticides
    
    def _is_valid_pesticide_name(self, name: str) -> bool:
        """Valida si un nombre es realmente un pesticida válido"""
        if not name or len(name) < 3:
            return False
        
        name_lower = name.lower()
        
        invalid_words = [
            'página', 'tabla', 'figura', 'anexo', 'realizar', 'cuando', 'altura',
            'malezas', 'con', 'tenga', 'aplicar', 'usar', 'dosis', 'cultivo',
            'control', 'de', 'la', 'el', 'en', 'para', 'por', 'se', 'es',
            'instrucciones', 'uso', 'etiqueta', 'producto', 'herbicida', 'fungicida',
            'antes', 'usar', 'lote', 'fabricante', 'trigo', 'cereal'
        ]
        
        words = name_lower.split()
        if all(word in invalid_words for word in words):
            return False
        
        if re.match(r'^\d+', name) or name_lower.startswith(('cm', 'mm', 'kg', 'g', 'l', 'ml')):
            return False
        
        if len(name) < 5 and not re.search(r'(WP|EC|SC|SL|WG|SE|EW)', name, re.IGNORECASE):
            return False
        
        if len(re.findall(r'\d+', name)) > 2 and not re.search(r'(WP|EC|SC|SL|WG|SE|EW)', name, re.IGNORECASE):
            return False
        
        if len(name.split()) > 4 and not re.search(r'(WP|EC|SC|SL|WG|SE|EW)', name, re.IGNORECASE):
            return False
        
        return True
    
    def _extract_name_from_filename(self, filename: str) -> str:
        """Extrae el nombre del pesticida del nombre del archivo"""
        base_name = filename.replace('.pdf', '')
        
        # Limpiar y formatear
        clean_name = re.sub(r'[_\-\d]{4,}.*$', '', base_name)
        clean_name = re.sub(r'[_\-]+', ' ', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        
        if clean_name:
            return clean_name.upper()
        
        return None
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extrae tablas preservando estructura exacta por fila"""
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        tables = page.extract_tables()
                        
                        for table_num, table in enumerate(tables, 1):
                            if table and len(table) > 1:
                                try:
                                    structured_chunks = self._process_table_by_rows(table, page_num, table_num)
                                    tables_data.extend(structured_chunks)
                                except Exception as table_error:
                                    logger.warning(f"Error procesando tabla {table_num} en página {page_num}: {table_error}")
                                    continue
                    except Exception as page_error:
                        logger.warning(f"Error procesando página {page_num}: {page_error}")
                        continue
                            
        except Exception as e:
            logger.error(f"Error extrayendo tablas de {pdf_path}: {e}")
        
        return tables_data
    
    def _process_table_by_rows(self, table: List, page_num: int, table_num: int) -> List[Dict[str, Any]]:
        """Procesa tabla creando un chunk por cada fila de datos"""
        chunks = []
        
        if not table or len(table) < 2:
            return chunks
        
        headers = [str(cell).strip() if cell else "" for cell in table[0]]
        
        cultivo_col = self._find_column_index(headers, ['cultivo', 'crop', 'trigo'])
        pest_col = self._find_column_index(headers, ['maleza', 'malezas', 'enfermedad', 'plaga', 'pest'])
        dosis_col = self._find_column_index(headers, ['dosis', 'dose', 'cantidad'])
        obs_col = self._find_column_index(headers, ['observaciones', 'observación', 'instrucciones'])
        
        for row_idx, row in enumerate(table[1:], 1):
            if row and any(cell for cell in row if cell):
                chunk_text = self._create_structured_row_chunk(
                    headers, row, row_idx, page_num, table_num,
                    cultivo_col, pest_col, dosis_col, obs_col
                )
                
                row_metadata = self._extract_row_metadata(
                    headers, row, cultivo_col, pest_col, dosis_col
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
                                    pest_col: int, dosis_col: int, obs_col: int) -> str:
        """Crea un chunk estructurado para una fila específica"""
        
        cultivo = str(row[cultivo_col]).strip() if cultivo_col >= 0 and cultivo_col < len(row) and row[cultivo_col] else ""
        pest = str(row[pest_col]).strip() if pest_col >= 0 and pest_col < len(row) and row[pest_col] else ""
        dosis = str(row[dosis_col]).strip() if dosis_col >= 0 and dosis_col < len(row) and row[dosis_col] else ""
        observaciones = str(row[obs_col]).strip() if obs_col >= 0 and obs_col < len(row) and row[obs_col] else ""
        
        chunk_text = f"=== REGISTRO DE APLICACIÓN - PÁGINA {page_num} ===\n\n"
        
        if cultivo:
            chunk_text += f"CULTIVO: {cultivo}\n"
        
        if pest:
            chunk_text += f"OBJETIVO DE CONTROL: {pest}\n"
        
        if dosis:
            chunk_text += f"DOSIS: {dosis}\n"
            dosis_header = headers[dosis_col] if dosis_col >= 0 and dosis_col < len(headers) else ""
            
            if 'g/100' in dosis_header:
                chunk_text += f"UNIDAD: g/100 L de agua\n"
            elif 'kg/ha' in dosis_header or 'kg/há' in dosis_header:
                chunk_text += f"UNIDAD: kg/ha\n"
            elif 'l/ha' in dosis_header.lower() or 'l/há' in dosis_header.lower():
                chunk_text += f"UNIDAD: L/ha\n"
            elif 'g/ha' in dosis_header or 'g/há' in dosis_header:
                chunk_text += f"UNIDAD: g/ha\n"
        
        if observaciones:
            chunk_text += f"\nINSTRUCCIONES DE APLICACIÓN:\n{observaciones}\n"
        
        chunk_text += f"\nDATOS COMPLETOS DE LA FILA:\n"
        for i, (header, cell) in enumerate(zip(headers, row)):
            if header and cell:
                chunk_text += f"{header}: {str(cell).strip()}\n"
        
        chunk_text += f"\nUBICACIÓN: Página {page_num}, Tabla {table_num}, Fila {row_idx}\n"
        
        return chunk_text
    
    def _extract_row_metadata(self, headers: List[str], row: List, cultivo_col: int, 
                            pest_col: int, dosis_col: int) -> Dict[str, Any]:
        """Extrae metadata específica de una fila"""
        metadata = {
            "type": "table_row",
            "has_dose_info": False,
            "has_crop_info": False,
            "has_pest_info": False,
            "dosis_especifica": "",
            "unidad_dosis": ""
        }
        
        if cultivo_col >= 0 and cultivo_col < len(row) and row[cultivo_col]:
            cultivo_text = str(row[cultivo_col]).strip()
            if cultivo_text:
                metadata["has_crop_info"] = True
                metadata["cultivos"] = cultivo_text
        
        if pest_col >= 0 and pest_col < len(row) and row[pest_col]:
            pest_text = str(row[pest_col]).strip()
            if pest_text:
                metadata["has_pest_info"] = True
                metadata["plagas_enfermedades_malezas"] = pest_text
        
        if dosis_col >= 0 and dosis_col < len(row) and row[dosis_col]:
            dosis_text = str(row[dosis_col]).strip()
            if dosis_text:
                metadata["has_dose_info"] = True
                metadata["dosis_especifica"] = dosis_text
        
        return metadata
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Procesa un PDF preservando estructura exacta por fila"""
        chunks = []
        filename = os.path.basename(pdf_path)
        
        logger.info(f"[RAG Trigo] Procesando: {filename}")
        
        table_chunks = self.extract_tables_from_pdf(pdf_path)
        
        for table_chunk in table_chunks:
            metadata = table_chunk.get("metadata", {})
            metadata["source_file"] = filename
            
            main_pesticide = self._extract_name_from_filename(filename)
            if main_pesticide:
                metadata["pesticide_principal"] = main_pesticide
            
            chunks.append({
                "text": table_chunk["text"],
                "metadata": metadata
            })
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += f"\n=== PÁGINA {page_num + 1} ===\n{page_text}"
            
            doc.close()
            
            additional_chunks = self._create_comprehensive_chunks(full_text, filename)
            chunks.extend(additional_chunks)
                
        except Exception as e:
            logger.error(f"Error procesando texto de {pdf_path}: {e}")
        
        logger.info(f"[RAG Trigo] ✅ {filename}: {len(chunks)} chunks creados ({len(table_chunks)} de tablas)")
        return chunks
    
    def _create_comprehensive_chunks(self, full_text: str, filename: str) -> List[Dict[str, Any]]:
        """Crea chunks comprensivos con TODA la información del archivo"""
        chunks = []
        pesticide_name = self._extract_name_from_filename(filename)
        
        general_info = self._extract_general_info(full_text, pesticide_name)
        if general_info:
            chunks.append({
                "text": general_info,
                "metadata": {
                    "source_file": filename,
                    "type": "general_info",
                    "pesticide_principal": pesticide_name,
                    "content_category": "product_description"
                }
            })
        
        full_document_chunk = {
            "text": f"=== DOCUMENTO COMPLETO - {pesticide_name} ===\n\n{full_text}",
            "metadata": {
                "source_file": filename,
                "type": "full_document",
                "pesticide_principal": pesticide_name,
                "is_complete_document": True,
                "content_category": "complete_reference"
            }
        }
        chunks.append(full_document_chunk)
        
        return chunks
    
    def _extract_general_info(self, text: str, pesticide_name: str) -> str:
        """Extrae información general del pesticida"""
        lines = text.split('\n')
        info_lines = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if pesticide_name and pesticide_name.lower() in line_clean.lower():
                info_lines.append(line_clean)
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and len(next_line) > 20:
                        info_lines.append(next_line)
                break
        
        if info_lines:
            return f"=== INFORMACIÓN GENERAL - {pesticide_name} ===\n\n" + '\n'.join(info_lines)
        return ""
    
    def create_semantic_chunks(self, text: str, source_file: str, chunk_size: int = 1200) -> List[Dict[str, Any]]:
        """Crea chunks semánticos con metadata ULTRA-ENRIQUECIDA"""
        chunks = []
        
        pages = text.split("=== PÁGINA")
        
        for page_num, page_text in enumerate(pages, 1):
            if not page_text.strip():
                continue
                
            pesticides = self.extract_pesticide_info(page_text, source_file)
            
            page_metadata = {
                "source_file": source_file,
                "page_number": page_num,
                "type": "text",
                "pesticides_found": len(pesticides),
                "pesticide_names": [p.name for p in pesticides],
                "product_types": [p.product_type for p in pesticides if p.product_type],
                "active_ingredients": [p.active_ingredient for p in pesticides if p.active_ingredient],
                "formulations": [p.formulation for p in pesticides if p.formulation],
                "dose_ranges": [p.dose_range for p in pesticides if p.dose_range],
                "dose_units": [p.dose_unit for p in pesticides if p.dose_unit],
                "application_timings": [p.application_timing for p in pesticides if p.application_timing],
                "crop_stages": [p.crop_stage for p in pesticides if p.crop_stage],
                "controlled_pests": [pest for p in pesticides for pest in p.controlled_pests],
                "target_crops": [crop for p in pesticides for crop in p.crops],
                "has_dose_info": any(p.dose_range for p in pesticides),
                "has_pest_info": any(p.controlled_pests for p in pesticides),
                "has_timing_info": any(p.application_timing for p in pesticides),
                "is_preemergent": any('preemergente' in p.crop_stage.lower() for p in pesticides if p.crop_stage),
                "is_postemergent": any('postemergente' in p.crop_stage.lower() for p in pesticides if p.crop_stage),
                "content_score": len(pesticides) * 2.0
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
                return [0.0] * 1536
                
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
        
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        logger.info(f"[RAG Trigo] ✅ {len(chunks)} chunks añadidos a la base")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Busca documentos relevantes"""
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
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
                    "similarity": 1 - distance,
                    "rank": i + 1
                })
        
        return formatted_results
    
    def count(self) -> int:
        """Retorna el número de documentos en la colección"""
        return self.collection.count()


class WheatRAGSystemEnhanced(WheatRAGSystem):
    """Versión mejorada del sistema RAG con capacidades adicionales de extracción"""
    
    def _extract_structured_text_info(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extrae información estructurada de texto cuando las tablas tradicionales fallan"""
        chunks = []
        lines = text.split('\n')
        chunk_counter = 0
        
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 100:
                continue
            
            # Patrón para información estructurada en texto
            pattern = r'([A-Za-zÁ-ú\s,]+),\s*([^,]+),\s*(\d+[.,]\d+\s*-\s*\d+[.,]\d+)\s*([^.]+\.)'
            matches = re.findall(pattern, line_clean, re.IGNORECASE)
            
            for match in matches:
                chunk_counter += 1
                
                cultivos_raw = match[0].strip()
                target_raw = match[1].strip()
                dosis_raw = match[2].strip()
                observaciones_raw = match[3].strip()
                
                cultivos = self._extract_relevant_crops(cultivos_raw)
                target = target_raw
                dosis = dosis_raw.replace(',', '.')
                
                if cultivos and dosis:
                    chunk_text = f"=== REGISTRO DE APLICACIÓN - PÁGINA {page_num} ===\n\n"
                    chunk_text += f"CULTIVO: {cultivos}\n"
                    chunk_text += f"OBJETIVO DE CONTROL: {target}\n"
                    chunk_text += f"DOSIS: {dosis}\n"
                    chunk_text += f"UNIDAD: L/ha\n"
                    chunk_text += f"\nINSTRUCCIONES DE APLICACIÓN:\n{observaciones_raw}\n"
                    chunk_text += f"\nUBICACIÓN: Página {page_num}, Texto estructurado {chunk_counter}\n"
                    
                    metadata = {
                        "type": "table_row",
                        "source": "structured_text_parsing",
                        "has_dose_info": True,
                        "has_crop_info": True,
                        "has_pest_info": bool(target),
                        "cultivos": cultivos,
                        "objetivo_control": target,
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
        """Extrae cultivos relevantes para el sistema de trigo"""
        relevant_crops = []
        
        crop_patterns = [
            r'\b(trigo|cebada|avena|centeno|triticale|maíz|sorgo|arroz)\b'
        ]
        
        for pattern in crop_patterns:
            matches = re.findall(pattern, cultivos_raw, re.IGNORECASE)
            relevant_crops.extend([match.title() for match in matches])
        
        return ', '.join(list(set(relevant_crops))) if relevant_crops else ""
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Versión mejorada que combina extracción de tablas tradicional + texto estructurado"""
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    if tables:
                        for table_num, table in enumerate(tables, 1):
                            if table and len(table) > 1:
                                structured_chunks = self._process_table_by_rows(table, page_num, table_num)
                                tables_data.extend(structured_chunks)
                    
                    if not tables or len(tables) < 2:
                        page_text = page.extract_text()
                        if page_text:
                            text_chunks = self._extract_structured_text_info(page_text, page_num)
                            tables_data.extend(text_chunks)
                            
        except Exception as e:
            logger.error(f"Error extrayendo tablas de {pdf_path}: {e}")
        
        return tables_data


def process_documents_directory(rag_system: WheatRAGSystem, documents_dir: str = "./documents_trigo"):
    """Procesa todos los PDFs en el directorio de documentos de trigo"""
    documents_path = Path(documents_dir)
    
    if not documents_path.exists():
        logger.error(f"Directorio {documents_dir} no existe")
        return
    
    pdf_files = list(documents_path.glob("*.pdf"))
    logger.info(f"[RAG Trigo] Encontrados {len(pdf_files)} archivos PDF")
    
    processed_count = 0
    error_count = 0
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"[RAG Trigo] Procesando: {pdf_file.name}")
            chunks = rag_system.process_pdf(str(pdf_file))
            if chunks:
                rag_system.add_documents(chunks)
                processed_count += 1
                logger.info(f"[RAG Trigo] ✅ {pdf_file.name} procesado exitosamente")
            else:
                logger.warning(f"[RAG Trigo] ⚠️ {pdf_file.name} no generó chunks")
        except Exception as e:
            error_count += 1
            logger.error(f"[RAG Trigo] ❌ Error procesando {pdf_file.name}: {e}")
            continue
    
    total_docs = rag_system.count()
    logger.info(f"[RAG Trigo] 🎉 Procesamiento completo:")
    logger.info(f"  - Archivos procesados: {processed_count}/{len(pdf_files)}")
    logger.info(f"  - Errores: {error_count}")
    logger.info(f"  - Total documentos en base: {total_docs}")


if __name__ == "__main__":
    # Ejemplo de uso
    rag = WheatRAGSystem()
    
    # Procesar documentos
    process_documents_directory(rag)
    
    # Prueba de búsqueda
    results = rag.search("herbicida trigo dosis", top_k=5)
    print(f"\nResultados de prueba: {len(results)}")
    for result in results:
        print(f"- {result['metadata']['source_file']} (similitud: {result['similarity']:.2%})")