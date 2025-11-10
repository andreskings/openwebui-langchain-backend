"""
Sistema RAG Optimizado Específicamente para Porotos
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
    """Estructura completa para información de pesticidas para porotos"""
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
    crop_stage: str = ""  # Preemergente, postemergente, estadios vegetativos/reproductivos
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


class PorotoRAGSystem:
    """Sistema RAG optimizado para pesticidas de porotos"""

    def __init__(self, persist_directory: str = "./chroma_porotos_v1"):
        self.persist_directory = persist_directory
        self.embedding_cache: Dict[str, List[float]] = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.embedding_model = "text-embedding-3-small"

        # Configurar ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Crear colección específica para porotos
        self.collection = self.client.get_or_create_collection(
            name="pesticidas_porotos",
            metadata={"description": "Pesticidas específicos para cultivo de porotos"}
        )

        logger.info(f"[RAG Porotos] Inicializado con {self.collection.count()} documentos")

    # ---------------------------------------------------------------------
    # Extracción estructurada
    # ---------------------------------------------------------------------
    def extract_pesticide_info(self, text: str, source_file: str) -> List[PesticideInfo]:
        """Extrae información COMPLETA de pesticidas del texto"""
        pesticides: List[PesticideInfo] = []
        text_lower = text.lower()

        # 1. EXTRAER NOMBRES DE PRODUCTOS
        name_patterns = [
            r'([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]*\d*\s*(?:WP|EC|SC|SL|WG|SE|EW)\b)',
            r'\b([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ]+)\s*®?\s*\d*\s*%?\s*(?:WP|EC|SC|SL|WG|SE|EW)\b',
            r'Producto:?\s*([^\n]+)',
            r'Herbicida:?\s*([^\n]+)',
            r'Fungicida:?\s*([^\n]+)',
            r'Insecticida:?\s*([^\n]+)',
            r'Nombre comercial:?\s*([^\n]+)',
            r'ETIQUETA\s+([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+\d*)',
        ]

        # 2. PATRONES PARA PRINCIPIOS ACTIVOS (frecuentes en leguminosas y porotos)
        active_ingredient_patterns = [
            r'(?:principio activo|ingrediente activo|p\.a\.|i\.a\.):\s*([^\n]+)',
            r'(?:contiene|composición):\s*([^\n]+)',
            r'([a-záéíóú\-]+(?:\s+[a-záéíóú\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*%',
            r'([A-ZÁÉÍÓÚÑ][a-záéíóú\-]+(?:\s+[a-záéíóú\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*g/L',
            r'([A-ZÁÉÍÓÚÑ][a-záéíóú\-]+(?:\s+[a-záéíóú\-]+)*?)\s+(\d+(?:[,\.]\d+)?)\s*g/kg',
            r'(bentazon|imazethapyr|imazamox|imazapic|fomesafen|flumioxazin|metribuzin|linuron|quizalofop|clethodim|sethoxydim|haloxyfop|pendimetalin|trifluralin|glyphosate|glifosato)\s+(\d+(?:[,\.]\d+)?)\s*(?:%|g/L|g/kg)',
            r'(azoxystrobin|boscalid|chlorotalonil|cymoxanil|metalaxyl|tebuconazole|propiconazole|cobre|oxicloruro de cobre|sulfoxaflor|lambda-cyhalothrin|bifentrin|spinetoram|spinosad|thiamethoxam)\s+(\d+(?:[,\.]\d+)?)\s*(?:%|g/L|g/kg)'
        ]

        # 3. PATRONES PARA DOSIS
        dose_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(l/ha|g/ha|litros/ha|gramos/ha|ml/ha|kg/ha|cc/ha|lt/ha)',
            r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*(?:kg|g|l|ml)\s*/\s*ha',
            r'(\d+(?:[,\.]\d+)?)\s*a\s*(\d+(?:[,\.]\d+)?)\s*(kg|L|g|ml)\s*/\s*ha',
            r'(\d+(?:[,\.]\d+)?)\s*(kg|g|l|ml)\s*/\s*h[áa]',
            r'Dosis:?\s*([^\n|]+)',
            r'aplicar\s+(\d+(?:[,\.]\d+)?)\s*(kg|L|g|ml)/ha'
        ]

        # 4. PATRONES PARA PLAGAS/ENFERMEDADES/MALEZAS DE POROTOS
        pest_patterns = [
            r'(?:controla|control de|eficaz contra):\s*([^\n\.]+)',
            r'(?:malezas?|plagas?|enfermedades?)\s+controladas?:\s*([^\n\.]+)',
            r'(?:rama negra|chufa amarilla|correhuela|gramíneas|amaranthus|verdolaga|ballica|malva|cenizo|quenopodio|cola de zorro)',
            r'(?:antracnosis|roya|oidio|mancha angular|mancha bacteriana|fusarium|sclerotinia|tiz[óo]n)',
            r'(?:pulg[óo]n|mosca blanca|trips|chanchito blanco|babosa|gusano cogollero|polilla del poroto|picudo)'
        ]

        # 5. PATRONES PARA MOMENTO DE APLICACIÓN
        timing_patterns = [
            r'(?:aplicar|aplicación)\s+([^\n\.]+?)(?:antes|después|durante)',
            r'(?:preemergente|pre-emergente|postemergente|post-emergente|pre siembra|pre-siembra)',
            r'(?:emergencia|V\d|floración|llenado de vaina|R\d)',
            r'(?:antes de la emergencia|después de la emergencia|post trasplante)',
            r'momento de aplicación:?\s*([^\n]+)'
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

                    formulation_match = re.search(r'(WP|EC|SC|SL|WG|SE|EW)', name, re.IGNORECASE)
                    if formulation_match:
                        pesticide.formulation = formulation_match.group(1).upper()

                    pesticides.append(pesticide)

        if not pesticides:
            file_based_name = self._extract_name_from_filename(source_file)
            if file_based_name:
                pesticides.append(PesticideInfo(name=file_based_name, source_file=source_file))

        main_pesticide = self._extract_name_from_filename(source_file)
        if main_pesticide and main_pesticide not in [p.name for p in pesticides]:
            pesticides.insert(0, PesticideInfo(name=main_pesticide, source_file=source_file))

        # ENRIQUECER CADA PESTICIDA CON INFORMACIÓN COMPLETA
        for pesticide in pesticides:
            for pattern, product_type in product_type_patterns:
                if re.search(pattern, text_lower):
                    pesticide.product_type = product_type
                    break

            for pattern in active_ingredient_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not pesticide.active_ingredient:
                    sample = matches[0]
                    if isinstance(sample, tuple):
                        if len(sample) >= 2:
                            pesticide.active_ingredient = f"{sample[0]} {sample[1]}%"
                        else:
                            pesticide.active_ingredient = " ".join(sample).strip()
                    else:
                        pesticide.active_ingredient = sample.strip()
                    break

            for pattern in dose_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not pesticide.dose_range:
                    sample = matches[0]
                    if isinstance(sample, tuple):
                        numeric_values = [val for val in sample if re.match(r"\d", str(val))]
                        units = [val for val in sample if isinstance(val, str) and any(u in val.lower() for u in ["/ha", "ha", "l", "g", "kg", "ml", "cc"])]
                        if len(numeric_values) >= 2:
                            pesticide.dose_range = f"{numeric_values[0]} - {numeric_values[1]}"
                        elif numeric_values:
                            pesticide.dose_range = str(numeric_values[0])
                        if units:
                            pesticide.dose_unit = units[0]
                    else:
                        pesticide.dose_range = str(sample)
                    break

            for pattern in pest_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    pest = match.strip() if isinstance(match, str) else str(match)
                    if pest and pest not in pesticide.controlled_pests:
                        pesticide.controlled_pests.append(pest)

            for pattern in timing_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and not pesticide.application_timing:
                    pesticide.application_timing = matches[0].strip() if isinstance(matches[0], str) else str(matches[0])
                    break

            if any(token in text_lower for token in ["poroto", "porotos", "frijol", "frijoles", "judía", "judías", "alubia", "alubias"]):
                pesticide.crops.append("porotos")

            water_match = re.search(r'(\d+(?:[,\.]\d+)?)\s*(?:-\s*\d+(?:[,\.]\d+)?)?\s*L?\s*(?:de\s*)?agua/ha', text, re.IGNORECASE)
            if water_match and not pesticide.water_volume:
                pesticide.water_volume = water_match.group(0)

            if 'preemergente' in text_lower or 'pre-emergente' in text_lower:
                pesticide.crop_stage = 'Preemergente'
            elif 'postemergente' in text_lower or 'post-emergente' in text_lower:
                pesticide.crop_stage = 'Postemergente'
            elif any(stage in text_lower for stage in ['v3', 'v4', 'vegetativo', 'desarrollo vegetativo']):
                pesticide.crop_stage = 'Vegetativo'
            elif any(stage in text_lower for stage in ['floración', 'r1', 'r2']):
                pesticide.crop_stage = 'Floración'
            elif any(stage in text_lower for stage in ['llenado de vaina', 'r5', 'r6']):
                pesticide.crop_stage = 'Reproductivo'

        pesticides = self._improve_dose_by_crop_context(pesticides, text)
        return pesticides

    def _improve_dose_by_crop_context(self, pesticides: List[PesticideInfo], text: str) -> List[PesticideInfo]:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(term in line_lower for term in ['poroto', 'porotos', 'frijol', 'judía']) and len(line_lower) < 80:
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    dose_patterns = [
                        r'(\d+,\d+)\s*-\s*(\d+,\d+)',
                        r'(\d+\.\d+)\s*-\s*(\d+\.\d+)',
                        r'(\d+,\d+)',
                        r'(\d+\.\d+)'
                    ]
                    for pattern in dose_patterns:
                        matches = re.findall(pattern, next_line)
                        if matches:
                            sample = matches[0]
                            for pesticide in pesticides:
                                if isinstance(sample, tuple) and len(sample) == 2:
                                    pesticide.dose_range = f"{sample[0]} - {sample[1]}"
                                else:
                                    pesticide.dose_range = str(sample)

                                context = ' '.join(lines[max(0, i - 2):min(len(lines), i + 6)]).lower()
                                if 'g/100' in context:
                                    pesticide.dose_unit = 'g/100 L de agua'
                                elif 'kg/ha' in context:
                                    pesticide.dose_unit = 'kg/ha'
                                elif any(tag in context for tag in ['l/ha', 'lt/ha']):
                                    pesticide.dose_unit = 'L/ha'
                            return pesticides
        return pesticides

    def _is_valid_pesticide_name(self, name: str) -> bool:
        if not name or len(name) < 3:
            return False
        name_lower = name.lower()
        invalid_words = [
            'página', 'tabla', 'figura', 'anexo', 'realizar', 'cuando', 'altura',
            'malezas', 'con', 'tenga', 'aplicar', 'usar', 'dosis', 'cultivo',
            'control', 'de', 'la', 'el', 'en', 'para', 'por', 'se', 'es',
            'instrucciones', 'uso', 'etiqueta', 'producto', 'herbicida', 'fungicida',
            'antes', 'usar', 'lote', 'fabricante', 'poroto', 'porotos', 'frijol'
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
        if len(name.split()) > 5 and not re.search(r'(WP|EC|SC|SL|WG|SE|EW)', name, re.IGNORECASE):
            return False
        return True

    def _extract_name_from_filename(self, filename: str) -> Optional[str]:
        base_name = filename.replace('.pdf', '')
        clean_name = re.sub(r'[_\-\d]{4,}.*$', '', base_name)
        clean_name = re.sub(r'[_\-]+', ' ', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        return clean_name.upper() if clean_name else None

    # ---------------------------------------------------------------------
    # Extracción de tablas y chunking semántico
    # ---------------------------------------------------------------------
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        tables_data: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        tables = page.extract_tables()
                        for table_num, table in enumerate(tables, 1):
                            if table and len(table) > 1:
                                structured_chunks = self._process_table_by_rows(table, page_num, table_num)
                                tables_data.extend(structured_chunks)
                    except Exception as page_error:
                        logger.warning(f"Error procesando página {page_num}: {page_error}")
        except Exception as e:
            logger.error(f"Error extrayendo tablas de {pdf_path}: {e}")
        return tables_data

    def _process_table_by_rows(self, table: List, page_num: int, table_num: int) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not table or len(table) < 2:
            return chunks
        headers = [str(cell).strip() if cell else "" for cell in table[0]]
        cultivo_col = self._find_column_index(headers, ['cultivo', 'crop', 'poroto', 'porotos', 'frijol'])
        pest_col = self._find_column_index(headers, ['maleza', 'malezas', 'plaga', 'plagas', 'enfermedad', 'pest'])
        dosis_col = self._find_column_index(headers, ['dosis', 'dose', 'cantidad'])
        obs_col = self._find_column_index(headers, ['observaciones', 'observación', 'instrucciones'])

        for row_idx, row in enumerate(table[1:], 1):
            if row and any(cell for cell in row if cell):
                chunk_text = self._create_structured_row_chunk(
                    headers, row, row_idx, page_num, table_num,
                    cultivo_col, pest_col, dosis_col, obs_col
                )
                metadata = self._extract_row_metadata(headers, row, cultivo_col, pest_col, dosis_col)
                chunks.append({
                    "page": page_num,
                    "table_num": table_num,
                    "row_num": row_idx,
                    "text": chunk_text,
                    "metadata": metadata,
                    "type": "table_row",
                    "raw_row": row,
                    "headers": headers
                })
        return chunks

    def _find_column_index(self, headers: List[str], keywords: List[str]) -> int:
        for i, header in enumerate(headers):
            if header:
                header_lower = header.lower()
                for keyword in keywords:
                    if keyword in header_lower:
                        return i
        return -1

    def _create_structured_row_chunk(
        self,
        headers: List[str],
        row: List,
        row_idx: int,
        page_num: int,
        table_num: int,
        cultivo_col: int,
        pest_col: int,
        dosis_col: int,
        obs_col: int
    ) -> str:
        cultivo = str(row[cultivo_col]).strip() if 0 <= cultivo_col < len(row) and row[cultivo_col] else ""
        pest = str(row[pest_col]).strip() if 0 <= pest_col < len(row) and row[pest_col] else ""
        dosis = str(row[dosis_col]).strip() if 0 <= dosis_col < len(row) and row[dosis_col] else ""
        observaciones = str(row[obs_col]).strip() if 0 <= obs_col < len(row) and row[obs_col] else ""

        chunk_text = f"=== REGISTRO DE APLICACIÓN - PÁGINA {page_num} ===\n\n"
        if cultivo:
            chunk_text += f"CULTIVO: {cultivo}\n"
        if pest:
            chunk_text += f"OBJETIVO DE CONTROL: {pest}\n"
        if dosis:
            chunk_text += f"DOSIS: {dosis}\n"
            dosis_header = headers[dosis_col] if 0 <= dosis_col < len(headers) else ""
            unidad = self._infer_unit_from_header(dosis_header)
            if unidad:
                chunk_text += f"UNIDAD: {unidad}\n"
        if observaciones:
            chunk_text += f"\nINSTRUCCIONES DE APLICACIÓN:\n{observaciones}\n"

        chunk_text += "\nDATOS COMPLETOS DE LA FILA:\n"
        for header, cell in zip(headers, row):
            if header and cell:
                chunk_text += f"{header}: {str(cell).strip()}\n"
        chunk_text += f"\nUBICACIÓN: Página {page_num}, Tabla {table_num}, Fila {row_idx}\n"
        return chunk_text

    def _infer_unit_from_header(self, header: str) -> str:
        header_lower = header.lower()
        if 'g/100' in header_lower:
            return 'g/100 L de agua'
        if 'kg/ha' in header_lower or 'kg/há' in header_lower:
            return 'kg/ha'
        if 'l/ha' in header_lower or 'l/há' in header_lower:
            return 'L/ha'
        if 'g/ha' in header_lower or 'g/há' in header_lower:
            return 'g/ha'
        return ''

    def _extract_row_metadata(self, headers: List[str], row: List, cultivo_col: int, pest_col: int, dosis_col: int) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "type": "table_row",
            "has_dose_info": False,
            "has_crop_info": False,
            "has_pest_info": False,
            "dosis_especifica": "",
            "unidad_dosis": ""
        }
        if 0 <= cultivo_col < len(row) and row[cultivo_col]:
            metadata["has_crop_info"] = True
            metadata["cultivos"] = str(row[cultivo_col]).strip()
        if 0 <= pest_col < len(row) and row[pest_col]:
            metadata["has_pest_info"] = True
            metadata["plagas_enfermedades_malezas"] = str(row[pest_col]).strip()
        if 0 <= dosis_col < len(row) and row[dosis_col]:
            metadata["has_dose_info"] = True
            metadata["dosis_especifica"] = str(row[dosis_col]).strip()
        return metadata

    # ---------------------------------------------------------------------
    # Procesamiento completo del PDF
    # ---------------------------------------------------------------------
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        filename = os.path.basename(pdf_path)
        logger.info(f"[RAG Porotos] Procesando: {filename}")

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

        logger.info(f"[RAG Porotos] ✅ {filename}: {len(chunks)} chunks creados ({len(table_chunks)} de tablas)")
        return chunks

    def _create_comprehensive_chunks(self, full_text: str, filename: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
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

    def _extract_general_info(self, text: str, pesticide_name: Optional[str]) -> str:
        if not pesticide_name:
            return ""
        lines = text.split('\n')
        info_lines: List[str] = []
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if pesticide_name.lower() in line_clean.lower():
                info_lines.append(line_clean)
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and len(next_line) > 20:
                        info_lines.append(next_line)
                break
        if info_lines:
            return f"=== INFORMACIÓN GENERAL - {pesticide_name} ===\n\n" + '\n'.join(info_lines)
        return ""

    def create_semantic_chunks(self, text: str, source_file: str, chunk_size: int = 1200) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
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
                "content_score": len(pesticides) * 2.0
            }
            chunks.append({
                "text": page_text.strip(),
                "metadata": page_metadata
            })
        return chunks

    # ---------------------------------------------------------------------
    # Gestión de embeddings y base vectorial
    # ---------------------------------------------------------------------
    def get_embedding(self, text: str) -> List[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        if not self.api_key:
            logger.error("OPENAI_API_KEY no configurada")
            return [0.0] * 1536
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.embedding_model,
                    "input": text
                },
                timeout=30
            )
            if response.status_code == 200:
                embedding = response.json()["data"][0]["embedding"]
                self.embedding_cache[text] = embedding
                return embedding
            logger.error(f"Error API embedding: {response.status_code} -> {response.text}")
        except Exception as e:
            logger.error(f"Error obteniendo embedding: {e}")
        return [0.0] * 1536

    def add_documents(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"chunk_{i}_{chunk['metadata'].get('source_file', 'porotos')}" for i, chunk in enumerate(chunks)]
        embeddings = [self.get_embedding(text) for text in texts]
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        logger.info(f"[RAG Porotos] ✅ {len(chunks)} chunks añadidos a la base")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        formatted_results: List[Dict[str, Any]] = []
        if results.get("documents") and results["documents"][0]:
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
        return self.collection.count()

    # ------------------------------------------------------------------
    # Persistencia del cache de embeddings (opcional)
    # ------------------------------------------------------------------
    def save_cache(self, cache_path: Optional[Path] = None):
        cache_path = cache_path or Path(self.persist_directory) / "embedding_cache.pkl"
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.error(f"Error guardando cache de embeddings: {e}")

    def load_cache(self, cache_path: Optional[Path] = None):
        cache_path = cache_path or Path(self.persist_directory) / "embedding_cache.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self.embedding_cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"No se pudo cargar cache de embeddings: {e}")


# -------------------------------------------------------------------------
# Procesamiento de directorios
# -------------------------------------------------------------------------

def process_documents_directory(rag_system: PorotoRAGSystem, documents_dir: str = "./documents_porotos"):
    """Procesa todos los PDFs en el directorio de documentos de porotos"""
    documents_path = Path(documents_dir)
    if not documents_path.exists():
        logger.error(f"Directorio {documents_dir} no existe")
        return

    pdf_files = list(documents_path.glob("*.pdf"))
    logger.info(f"[RAG Porotos] Encontrados {len(pdf_files)} archivos PDF")
    processed_count = 0
    error_count = 0

    for pdf_file in pdf_files:
        try:
            logger.info(f"[RAG Porotos] Procesando: {pdf_file.name}")
            chunks = rag_system.process_pdf(str(pdf_file))
            if chunks:
                rag_system.add_documents(chunks)
                processed_count += 1
                logger.info(f"[RAG Porotos] ✅ {pdf_file.name} procesado exitosamente")
            else:
                logger.warning(f"[RAG Porotos] ⚠️ {pdf_file.name} no generó chunks")
        except Exception as e:
            error_count += 1
            logger.error(f"[RAG Porotos] ❌ Error procesando {pdf_file.name}: {e}")

    total_docs = rag_system.count()
    logger.info("[RAG Porotos] 🎉 Procesamiento completo:")
    logger.info(f"  - Archivos procesados: {processed_count}/{len(pdf_files)}")
    logger.info(f"  - Errores: {error_count}")
    logger.info(f"  - Total documentos en base: {total_docs}")


if __name__ == "__main__":
    rag = PorotoRAGSystem()
    process_documents_directory(rag)
    results = rag.search("herbicida porotos dosis", top_k=5)
    print(f"\nResultados de prueba: {len(results)}")
    for result in results:
        metadata = result.get("metadata", {})
        print(f"- {metadata.get('source_file', 'desconocido')} (similitud: {result['similarity']:.2%})")
