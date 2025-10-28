#!/usr/bin/env python3
"""
Evaluador Automático de Pesticidas para Trigo
Procesa múltiples archivos PDF, evalúa si son relevantes para trigo,
y permite al usuario decidir cuáles guardar en la base de datos.
"""

import os
import re
import fitz
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rag_trigo_optimizado import WheatRAGSystem

@dataclass
class EvaluationResult:
    """Resultado de la evaluación de un pesticida para trigo"""
    filename: str
    product_name: str
    chunks_count: int
    table_rows: int
    has_wheat: bool
    wheat_mentions: List[str]
    related_crops: List[str]
    active_ingredient: Optional[str]
    dose_info: List[str]
    product_type: str  # Herbicida, Fungicida, Insecticida
    priority: str  # 'HIGH', 'MEDIUM', 'LOW', 'SKIP'
    recommendation: str
    reasons: List[str]

class WheatPesticideEvaluator:
    """Evaluador automático de pesticidas para trigo"""
    
    def __init__(self):
        self.rag = WheatRAGSystem()
        self.processed_products = self._get_processed_products()
        
        # Cultivos cereales relacionados
        self.related_crops = [
            'cebada', 'avena', 'centeno', 'triticale', 'maíz', 'sorgo', 'arroz',
            'cereal', 'cereales', 'gramínea', 'gramíneas'
        ]
        
        # Principios activos conocidos para trigo
        self.known_ingredients = [
            # Herbicidas
            'glifosato', '2,4-d', 'mcpa', 'dicamba', 'metsulfuron', 'tribenuron',
            'florasulam', 'pyroxsulam', 'pinoxaden', 'clodinafop', 'fenoxaprop',
            'diclofop', 'tralkoxydim', 'clethodim', 'quizalofop', 'fluazifop',
            'haloxyfop', 'iodosulfuron', 'mesosulfuron', 'sulfosulfuron',
            # Fungicidas
            'tebuconazole', 'propiconazole', 'epoxiconazole', 'metconazole',
            'prothioconazole', 'azoxystrobin', 'pyraclostrobin', 'trifloxystrobin',
            'boscalid', 'fluopyram', 'fluxapyroxad', 'carbendazim', 'tiofanato',
            'mancozeb', 'chlorothalonil', 'benomyl', 'triadimenol',
            # Insecticidas
            'imidacloprid', 'thiamethoxam', 'clothianidin', 'acetamiprid',
            'thiacloprid', 'lambda-cyhalothrin', 'gamma-cyhalothrin', 'deltamethrin',
            'cypermethrin', 'bifenthrin', 'chlorpyrifos', 'dimethoate', 'malathion'
        ]
        
        # Patrones de dosis - simplificados y sin errores
        self.dose_patterns = [
            # Patrones básicos con unidades
            r'\d+[.,]?\d*\s*-\s*\d+[.,]?\d*\s*L/ha',
            r'\d+[.,]?\d*\s*L/ha',
            r'\d+[.,]?\d*\s*-\s*\d+[.,]?\d*\s*kg/ha',
            r'\d+[.,]?\d*\s*kg/ha',
            r'\d+[.,]?\d*\s*-\s*\d+[.,]?\d*\s*g/ha',
            r'\d+[.,]?\d*\s*g/ha',
            r'\d+[.,]?\d*\s*-\s*\d+[.,]?\d*\s*ml/ha',
            r'\d+[.,]?\d*\s*ml/ha',
            r'\d+[.,]?\d*\s*-\s*\d+[.,]?\d*\s*gr/ha',
            r'\d+[.,]?\d*\s*gr/ha',
            r'\d+[.,]?\d*\s*g/100\s*L',
            
            # Patrones con DOSIS
            r'DOSIS:\s*\d+[.,]?\d*',
            r'dosis:\s*\d+[.,]?\d*',
            r'Dosis:\s*\d+[.,]?\d*',
            
            # Patrones simples de números
            r'\d+\s*-\s*\d+',
            r'\d+[.,]\d+\s*-\s*\d+[.,]\d+'
        ]

    def _get_processed_products(self) -> List[str]:
        """Obtiene lista de productos ya procesados"""
        # Lista inicial - se puede expandir según productos ya en la base
        return [
            'glifosato', 'roundup', 'touchdown', 'mcpa', 'dicamba', 'banvel',
            'metsulfuron', 'ally', 'tribenuron', 'granstar', 'atlantis'
        ]

    def _is_already_processed(self, filename: str) -> bool:
        """Verifica si un archivo ya fue procesado"""
        filename_lower = filename.lower()
        for product in self.processed_products:
            if product in filename_lower:
                return True
        return False

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extrae todo el texto de un PDF"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ''
            for page in doc:
                full_text += page.get_text()
            doc.close()
            return full_text
        except Exception as e:
            print(f"Error extrayendo texto de {pdf_path}: {e}")
            return ""

    def _find_wheat_mentions(self, text: str) -> List[str]:
        """Busca menciones específicas de trigo con múltiples variaciones"""
        mentions = []
        lines = text.split('\n')
        
        # Patrones de búsqueda para trigo
        wheat_patterns = [
            r'trigo',
            r'triticum',
            r'wheat',
            r'cereales?\s+de\s+invierno',
            r'trigo\s+de\s+invierno',
            r'trigo\s+harinero',
            r'trigo\s+candeal',
            r'trigo\s+duro'
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            for pattern in wheat_patterns:
                if re.search(pattern, line_lower):
                    mentions.append(f"Línea {i+1}: {line.strip()}")
                    break  # Evitar duplicados de la misma línea
        
        return mentions

    def _find_related_crops(self, text: str) -> List[str]:
        """Busca cultivos cereales relacionados"""
        found_crops = []
        text_lower = text.lower()
        
        for crop in self.related_crops:
            if crop in text_lower:
                found_crops.append(crop)
        
        return list(set(found_crops))

    def _find_active_ingredient(self, text: str) -> Optional[str]:
        """Identifica el principio activo"""
        text_lower = text.lower()
        
        for ingredient in self.known_ingredients:
            if ingredient in text_lower:
                return ingredient
        
        return None

    def _find_product_type(self, text: str) -> str:
        """Identifica el tipo de producto"""
        text_lower = text.lower()
        
        if 'herbicida' in text_lower:
            return 'Herbicida'
        elif 'fungicida' in text_lower:
            return 'Fungicida'
        elif 'insecticida' in text_lower:
            return 'Insecticida'
        elif 'acaricida' in text_lower:
            return 'Acaricida'
        
        return 'Desconocido'

    def _find_dose_info(self, text: str) -> List[str]:
        """Busca información de dosis específicamente para trigo"""
        dose_info = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Buscar líneas que mencionen trigo Y tengan dosis
            if 'trigo' in line_lower:
                for pattern in self.dose_patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        dose_info.append(f"TRIGO - Línea {i+1}: {line.strip()}")
            
            # También buscar dosis en líneas cercanas a menciones de trigo
            else:
                for pattern in self.dose_patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        # Verificar si hay mención de trigo en líneas cercanas (±3 líneas)
                        context_start = max(0, i-3)
                        context_end = min(len(lines), i+4)
                        context_text = ' '.join(lines[context_start:context_end]).lower()
                        
                        if 'trigo' in context_text:
                            dose_info.append(f"CONTEXTO TRIGO - Línea {i+1}: {line.strip()}")
                        break
        
        return dose_info

    def _extract_wheat_doses_from_chunks(self, chunks: List[Dict]) -> List[str]:
        """Extrae dosis específicas de trigo de los chunks del RAG interpretando tablas estructuradas"""
        wheat_doses = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            chunk_text_lower = chunk_text.lower()
            metadata = chunk.get('metadata', {})
            
            # PRIORIDAD 1: Chunks de tabla que mencionan trigo
            if metadata.get('type') == 'table_row' and 'trigo' in chunk_text_lower:
                # Buscar información estructurada específica
                lines = chunk_text.split('\n')
                cultivo_info = ""
                dosis_info = ""
                unidad_info = ""
                
                for line in lines:
                    line_clean = line.strip()
                    line_lower = line.lower()
                    
                    if 'cultivo:' in line_lower and 'trigo' in line_lower:
                        cultivo_info = line_clean
                    elif 'dosis:' in line_lower:
                        dosis_info = line_clean
                    elif 'unidad:' in line_lower:
                        unidad_info = line_clean
                
                # Si encontramos información completa de tabla
                if cultivo_info and (dosis_info or unidad_info):
                    wheat_doses.append(f"RAG-TABLA-COMPLETA-{i+1} - {cultivo_info} | {dosis_info} {unidad_info}")
                elif 'trigo' in chunk_text_lower:
                    # Buscar cualquier patrón de dosis en el chunk
                    for pattern in self.dose_patterns:
                        if re.search(pattern, chunk_text_lower):
                            wheat_doses.append(f"RAG-TABLA-TRIGO-{i+1} - {chunk_text.strip()[:150]}...")
                            break
            
            # PRIORIDAD 2: Chunks que mencionan trigo directamente
            elif 'trigo' in chunk_text_lower:
                lines = chunk_text.split('\n')
                
                for line_num, line in enumerate(lines):
                    line_lower = line.lower()
                    line_clean = line.strip()
                    
                    # Líneas que mencionan trigo Y tienen dosis
                    if 'trigo' in line_lower:
                        for pattern in self.dose_patterns:
                            if re.search(pattern, line_lower):
                                wheat_doses.append(f"RAG-TRIGO-DOSIS-{i+1} - {line_clean}")
                                break
                    
                    # Líneas con DOSIS en contexto de trigo
                    elif 'dosis:' in line_lower:
                        wheat_doses.append(f"RAG-DOSIS-CONTEXTO-{i+1} - {line_clean}")
                    
                    # Líneas con UNIDAD en contexto de trigo
                    elif 'unidad:' in line_lower and any(unit in line_lower for unit in ['g/ha', 'l/ha', 'kg/ha', 'ml/ha']):
                        wheat_doses.append(f"RAG-UNIDAD-CONTEXTO-{i+1} - {line_clean}")
            
            # PRIORIDAD 3: Chunks de tabla sin mención directa pero con estructura
            elif metadata.get('type') == 'table_row':
                # Verificar si tiene estructura típica de aplicación
                if any(word in chunk_text_lower for word in ['cultivo', 'aplicación', 'dosis']) and \
                   any(re.search(pattern, chunk_text_lower) for pattern in self.dose_patterns):
                    wheat_doses.append(f"RAG-TABLA-GENERAL-{i+1} - {chunk_text.strip()[:100]}...")
        
        return wheat_doses

    def _calculate_priority(self, result: EvaluationResult) -> Tuple[str, str, List[str]]:
        """Calcula prioridad y recomendación"""
        reasons = []
        
        # Verificar si ya fue procesado
        if self._is_already_processed(result.filename):
            return 'SKIP', 'Ya procesado anteriormente', ['Producto ya en la base de datos']
        
        # Verificar si menciona trigo explícitamente
        if result.has_wheat:
            reasons.append('✅ Menciona TRIGO explícitamente')
            
            # Contar dosis específicas para trigo
            trigo_doses = [dose for dose in result.dose_info if 'TRIGO' in dose]
            context_doses = [dose for dose in result.dose_info if 'CONTEXTO TRIGO' in dose]
            
            if result.table_rows > 0:
                reasons.append(f'✅ {result.table_rows} registros de aplicación')
                
                if trigo_doses:
                    reasons.append(f'✅ {len(trigo_doses)} dosis ESPECÍFICAS para trigo')
                    if result.active_ingredient:
                        reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                        return 'HIGH', 'PROCESAR - Dosis específicas para trigo + información completa', reasons
                    else:
                        return 'HIGH', 'PROCESAR - Dosis específicas para trigo (principio activo a verificar)', reasons
                elif context_doses:
                    reasons.append(f'✅ {len(context_doses)} dosis en contexto de trigo')
                    if result.active_ingredient:
                        reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                        return 'MEDIUM', 'PROCESAR - Dosis en contexto de trigo + principio activo', reasons
                    else:
                        return 'MEDIUM', 'PROCESAR - Dosis en contexto de trigo (verificar principio activo)', reasons
                else:
                    return 'MEDIUM', 'PROCESAR - Menciona trigo pero sin dosis específicas', reasons
            else:
                reasons.append('⚠️ Sin tablas de aplicación')
                if trigo_doses:
                    reasons.append(f'✅ {len(trigo_doses)} dosis específicas para trigo')
                    return 'MEDIUM', 'PROCESAR - Dosis para trigo pero sin tablas', reasons
                else:
                    return 'LOW', 'EVALUAR - Menciona trigo pero información muy limitada', reasons
        
        # Verificar cultivos cereales relacionados
        elif result.related_crops:
            reasons.append(f'⚠️ Menciona cereales relacionados: {", ".join(result.related_crops)}')
            if result.active_ingredient:
                reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                if result.product_type in ['Herbicida', 'Fungicida', 'Insecticida']:
                    reasons.append(f'✅ Tipo de producto: {result.product_type}')
                    return 'MEDIUM', 'PROCESAR OPCIONAL - Cereales relacionados con principio activo conocido', reasons
                else:
                    return 'LOW', 'EVALUAR MANUALMENTE - Cereales relacionados pero tipo desconocido', reasons
            else:
                return 'LOW', 'EVALUAR MANUALMENTE - Cereales relacionados pero principio activo desconocido', reasons
        
        # Sin información relevante
        else:
            reasons.append('❌ No menciona trigo ni cereales relacionados')
            if result.chunks_count < 5:
                reasons.append('❌ Información muy limitada')
                return 'SKIP', 'NO PROCESAR - Sin información relevante', reasons
            else:
                return 'LOW', 'EVALUAR MANUALMENTE - Sin menciones directas pero documento completo', reasons

    def evaluate_file(self, pdf_path: str) -> EvaluationResult:
        """Evalúa un archivo PDF individual"""
        filename = os.path.basename(pdf_path)
        print(f"🔍 Evaluando: {filename}")
        
        # Extraer chunks usando el sistema RAG
        try:
            chunks = self.rag.process_pdf(pdf_path)
            chunks_count = len(chunks)
            
            # Contar tipos de chunks
            table_rows = sum(1 for chunk in chunks 
                           if chunk.get('metadata', {}).get('type') == 'table_row')
            
            # 🎯 NUEVO: Buscar dosis de trigo en los chunks del RAG
            rag_wheat_doses = self._extract_wheat_doses_from_chunks(chunks)
            
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
            chunks_count = 0
            table_rows = 0
            rag_wheat_doses = []
        
        # Extraer texto completo para análisis
        full_text = self._extract_pdf_text(pdf_path)
        
        # Buscar información específica
        wheat_mentions = self._find_wheat_mentions(full_text)
        related_crops = self._find_related_crops(full_text)
        active_ingredient = self._find_active_ingredient(full_text)
        dose_info = self._find_dose_info(full_text)
        product_type = self._find_product_type(full_text)
        
        # 🎯 Combinar dosis del texto y del RAG
        dose_info.extend(rag_wheat_doses)
        
        # Crear resultado
        result = EvaluationResult(
            filename=filename,
            product_name=self._extract_product_name(filename),
            chunks_count=chunks_count,
            table_rows=table_rows,
            has_wheat=len(wheat_mentions) > 0,
            wheat_mentions=wheat_mentions,
            related_crops=related_crops,
            active_ingredient=active_ingredient,
            dose_info=dose_info,
            product_type=product_type,
            priority='',
            recommendation='',
            reasons=[]
        )
        
        # Calcular prioridad
        priority, recommendation, reasons = self._calculate_priority(result)
        result.priority = priority
        result.recommendation = recommendation
        result.reasons = reasons
        
        return result

    def _extract_product_name(self, filename: str) -> str:
        """Extrae nombre del producto del archivo"""
        # Remover extensión y caracteres especiales
        name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        
        # Extraer primera parte (generalmente el nombre del producto)
        parts = name.split()
        if len(parts) > 0:
            return parts[0].upper()
        
        return name.upper()

    def evaluate_directory(self, directory_path: str = './documents_trigo') -> List[EvaluationResult]:
        """Evalúa todos los PDFs en un directorio"""
        directory = Path(directory_path)
        pdf_files = list(directory.glob('*.pdf'))
        
        print(f"🔍 EVALUADOR AUTOMÁTICO DE PESTICIDAS PARA TRIGO")
        print(f"=" * 60)
        print(f"📁 Directorio: {directory_path}")
        print(f"📄 Archivos PDF encontrados: {len(pdf_files)}")
        print(f"🗄️ Estado actual de la base: {self.rag.count()} documentos")
        print()
        
        results = []
        
        for pdf_file in pdf_files:
            try:
                result = self.evaluate_file(str(pdf_file))
                results.append(result)
                
                # Mostrar resultado inmediato
                self._print_evaluation_summary(result)
                print()
                
            except Exception as e:
                print(f"❌ Error evaluando {pdf_file.name}: {e}")
                print()
        
        return results

    def _print_evaluation_summary(self, result: EvaluationResult):
        """Imprime resumen de evaluación"""
        priority_colors = {
            'HIGH': '🟢',
            'MEDIUM': '🟡', 
            'LOW': '🟠',
            'SKIP': '🔴'
        }
        
        color = priority_colors.get(result.priority, '⚪')
        
        print(f"{color} {result.priority}: {result.product_name}")
        print(f"   📊 {result.chunks_count} chunks, {result.table_rows} tablas")
        print(f"   🧪 Tipo: {result.product_type}")
        
        if result.has_wheat:
            print(f"   🌾 ✅ Menciona TRIGO ({len(result.wheat_mentions)} veces)")
        elif result.related_crops:
            print(f"   🌾 ⚠️ Cereales relacionados: {', '.join(result.related_crops)}")
        else:
            print(f"   ❌ Sin menciones de trigo")
        
        if result.active_ingredient:
            print(f"   🧪 Principio activo: {result.active_ingredient}")
        
        if result.dose_info:
            # Separar dosis por tipo y origen
            tabla_completa = [dose for dose in result.dose_info if 'RAG-TABLA-COMPLETA' in dose]
            tabla_trigo = [dose for dose in result.dose_info if 'RAG-TABLA-TRIGO' in dose]
            trigo_dosis = [dose for dose in result.dose_info if 'RAG-TRIGO-DOSIS' in dose]
            dosis_contexto = [dose for dose in result.dose_info if 'RAG-DOSIS-CONTEXTO' in dose]
            unidad_contexto = [dose for dose in result.dose_info if 'RAG-UNIDAD-CONTEXTO' in dose]
            tabla_general = [dose for dose in result.dose_info if 'RAG-TABLA-GENERAL' in dose]
            text_doses = [dose for dose in result.dose_info if 'TRIGO -' in dose or 'CONTEXTO TRIGO' in dose]
            
            if tabla_completa:
                print(f"   🎯 TABLA COMPLETA (TRIGO): {len(tabla_completa)} encontradas")
                for dose in tabla_completa[:2]:
                    clean_dose = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      ✅ {clean_dose}")
            
            if tabla_trigo:
                print(f"   📊 TABLA CON TRIGO: {len(tabla_trigo)} encontradas")
                for dose in tabla_trigo[:1]:
                    clean_dose = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      🌾 {clean_dose}")
            
            if trigo_dosis:
                print(f"   🎯 TRIGO + DOSIS: {len(trigo_dosis)} encontradas")
                for dose in trigo_dosis[:1]:
                    clean_dose = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      ✅ {clean_dose}")
            
            if dosis_contexto:
                print(f"   💊 DOSIS EN CONTEXTO: {len(dosis_contexto)} encontradas")
                for dose in dosis_contexto[:1]:
                    clean_dose = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      📍 {clean_dose}")
            
            if unidad_contexto:
                print(f"   📏 UNIDAD EN CONTEXTO: {len(unidad_contexto)} encontradas")
                for dose in unidad_contexto[:1]:
                    clean_dose = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      📏 {clean_dose}")
            
            if text_doses:
                print(f"   📄 TEXTO: {len(text_doses)} encontradas")
            
            if not any([tabla_completa, tabla_trigo, trigo_dosis, dosis_contexto, unidad_contexto, text_doses]):
                print(f"   ⚠️ Dosis generales: {len(result.dose_info)} (sin especificar trigo)")
        
        print(f"   💡 {result.recommendation}")

    def interactive_processing(self, results: List[EvaluationResult]):
        """Procesamiento interactivo con decisión del usuario"""
        print(f"\n🎯 RESUMEN DE EVALUACIÓN")
        print(f"=" * 50)
        
        # Agrupar por prioridad
        by_priority = {}
        for result in results:
            if result.priority not in by_priority:
                by_priority[result.priority] = []
            by_priority[result.priority].append(result)
        
        # Mostrar resumen
        for priority in ['HIGH', 'MEDIUM', 'LOW', 'SKIP']:
            if priority in by_priority:
                count = len(by_priority[priority])
                print(f"{priority}: {count} archivos")
        
        print(f"\n📋 CANDIDATOS PARA PROCESAMIENTO:")
        print(f"=" * 40)
        
        candidates = [r for r in results if r.priority in ['HIGH', 'MEDIUM']]
        
        if not candidates:
            print("❌ No se encontraron candidatos relevantes para trigo")
            return
        
        for i, result in enumerate(candidates, 1):
            print(f"\n{i}. {result.product_name} ({result.filename})")
            print(f"   Prioridad: {result.priority}")
            print(f"   Tipo: {result.product_type}")
            print(f"   📊 {result.chunks_count} chunks, {result.table_rows} tablas")
            
            if result.wheat_mentions:
                print(f"   🌾 Menciones de trigo:")
                for mention in result.wheat_mentions[:2]:  # Mostrar máximo 2
                    print(f"      • {mention}")
            
            if result.related_crops:
                print(f"   🌾 Cereales relacionados: {', '.join(result.related_crops)}")
            
            if result.active_ingredient:
                print(f"   🧪 Principio activo: {result.active_ingredient}")
            
            print(f"   💡 {result.recommendation}")
            
            # Preguntar al usuario
            while True:
                response = input(f"\n¿Procesar {result.product_name}? (s/n/d=detalles): ").lower().strip()
                
                if response == 'd':
                    self._show_detailed_info(result)
                    continue
                elif response in ['s', 'si', 'y', 'yes']:
                    self._process_and_save(result)
                    break
                elif response in ['n', 'no']:
                    print(f"⏭️ Omitiendo {result.product_name}")
                    break
                else:
                    print("Por favor responde 's' para sí, 'n' para no, o 'd' para ver detalles")

    def _show_detailed_info(self, result: EvaluationResult):
        """Muestra información detallada de un resultado"""
        print(f"\n📋 INFORMACIÓN DETALLADA: {result.product_name}")
        print(f"=" * 50)
        print(f"Archivo: {result.filename}")
        print(f"Tipo de producto: {result.product_type}")
        print(f"Chunks extraídos: {result.chunks_count}")
        print(f"Registros de aplicación: {result.table_rows}")
        print(f"Prioridad: {result.priority}")
        print()
        
        if result.wheat_mentions:
            print("🌾 MENCIONES DE TRIGO:")
            for mention in result.wheat_mentions:
                print(f"   • {mention}")
            print()
        
        if result.related_crops:
            print(f"🌾 CEREALES RELACIONADOS: {', '.join(result.related_crops)}")
            print()
        
        if result.active_ingredient:
            print(f"🧪 PRINCIPIO ACTIVO: {result.active_ingredient}")
            print()
        
        if result.dose_info:
            print("💊 INFORMACIÓN DE DOSIS:")
            for dose in result.dose_info[:3]:  # Mostrar máximo 3
                print(f"   • {dose}")
            print()
        
        print("📝 RAZONES DE LA EVALUACIÓN:")
        for reason in result.reasons:
            print(f"   • {reason}")
        print()

    def _process_and_save(self, result: EvaluationResult):
        """Procesa y guarda un pesticida en la base de datos"""
        print(f"\n💾 PROCESANDO Y GUARDANDO: {result.product_name}")
        print(f"=" * 50)
        
        try:
            # Obtener path completo
            pdf_path = f"./documents_trigo/{result.filename}"
            
            # Verificar estado inicial
            initial_count = self.rag.count()
            print(f"Estado inicial: {initial_count} documentos")
            
            # Procesar archivo
            chunks = self.rag.process_pdf(pdf_path)
            print(f"Chunks extraídos: {len(chunks)}")
            
            # Guardar en la base de datos
            print("Guardando en la base de datos...")
            self.rag.add_documents(chunks)
            
            # Verificar estado final
            final_count = self.rag.count()
            added_docs = final_count - initial_count
            
            print(f"✅ COMPLETADO EXITOSAMENTE!")
            print(f"   Documentos añadidos: {added_docs}")
            print(f"   Estado final: {final_count} documentos")
            print(f"   {result.product_name} agregado al sistema")
            
        except Exception as e:
            print(f"❌ Error procesando {result.product_name}: {e}")

def main():
    """Función principal"""
    evaluator = WheatPesticideEvaluator()
    
    # Evaluar todos los archivos
    results = evaluator.evaluate_directory()
    
    # Procesamiento interactivo
    evaluator.interactive_processing(results)
    
    print(f"\n🎉 EVALUACIÓN COMPLETADA")
    print(f"Estado final de la base: {evaluator.rag.count()} documentos")

if __name__ == "__main__":
    main()
