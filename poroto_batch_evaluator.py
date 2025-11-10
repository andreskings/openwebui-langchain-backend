#!/usr/bin/env python3
"""
Evaluador Automático de Pesticidas para Porotos
Procesa múltiples archivos PDF, evalúa si son relevantes para porotos,
y permite al usuario decidir cuáles guardar en la base de datos.
"""

import os
import re
import fitz
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rag_porotos_optimizado import PorotoRAGSystem


@dataclass
class EvaluationResult:
    """Resultado de la evaluación de un pesticida para porotos"""
    filename: str
    product_name: str
    chunks_count: int
    table_rows: int
    has_beans: bool
    bean_mentions: List[str]
    related_crops: List[str]
    active_ingredient: Optional[str]
    dose_info: List[str]
    product_type: str  # Herbicida, Fungicida, Insecticida
    priority: str  # 'HIGH', 'MEDIUM', 'LOW', 'SKIP'
    recommendation: str
    reasons: List[str]


class PorotoPesticideEvaluator:
    """Evaluador automático de pesticidas para porotos"""

    def __init__(self):
        self.rag = PorotoRAGSystem()
        self.processed_products = self._get_processed_products()

        # Cultivos leguminosos relacionados
        self.related_crops = [
            'frijol', 'frijoles', 'judía', 'judías', 'alubia', 'alubias',
            'garbanzo', 'lenteja', 'arveja', 'soja', 'soya', 'haba', 'lupino',
            'leguminosa', 'leguminosas'
        ]

        # Principios activos conocidos para porotos
        self.known_ingredients = [
            # Herbicidas
            'bentazon', 'imazethapyr', 'imazamox', 'imazapic', 'fomesafen',
            'flumioxazin', 'metribuzin', 'linuron', 'pendimetalin', 'trifluralin',
            'clethodim', 'quizalofop', 'sethoxydim', 'haloxyfop', 'glyphosate', 'glifosato',
            'oxadiazon', 'propanil', 'fenoxaprop', 'clomazone', 'carbetamide',
            # Fungicidas
            'azoxystrobin', 'boscalid', 'chlorotalonil', 'cymoxanil', 'metalaxil',
            'tebuconazole', 'propiconazole', 'cobre', 'oxicloruro de cobre',
            'mancozeb', 'benomyl', 'carbendazim', 'tiofanato', 'sclerotinia',
            # Insecticidas
            'lambda-cyhalothrin', 'bifentrin', 'spinetoram', 'spinosad',
            'thiamethoxam', 'acetamiprid', 'imidacloprid', 'chlorpyrifos',
            'dimetoato', 'malation', 'beta-ciflutrin', 'cypermethrin'
        ]

        # Patrones de dosis reutilizados
        self.dose_patterns = [
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
            r'DOSIS:\s*\d+[.,]?\d*',
            r'dosis:\s*\d+[.,]?\d*',
            r'Dosis:\s*\d+[.,]?\d*',
            r'\d+\s*-\s*\d+',
            r'\d+[.,]\d+\s*-\s*\d+[.,]\d+'
        ]

    def _get_processed_products(self) -> List[str]:
        """Obtiene lista de productos ya procesados"""
        return [
            'bentazon', 'imazetapir', 'fomesafen', 'linuron', 'metribuzin',
            'pendimetalin', 'trifluralin', 'glifosato', 'clethodim', 'quizalofop'
        ]

    def _is_already_processed(self, filename: str) -> bool:
        filename_lower = filename.lower()
        return any(product in filename_lower for product in self.processed_products)

    def _extract_pdf_text(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            full_text = ''.join(page.get_text() for page in doc)
            doc.close()
            return full_text
        except Exception as e:
            print(f"Error extrayendo texto de {pdf_path}: {e}")
            return ""

    def _find_bean_mentions(self, text: str) -> List[str]:
        mentions = []
        lines = text.split('\n')
        bean_patterns = [
            r'poroto', r'porotos', r'frijol', r'frijoles', r'judía', r'judías',
            r'alubia', r'alubias', r'phaseolus vulgaris', r'bean'
        ]
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(re.search(pattern, line_lower) for pattern in bean_patterns):
                mentions.append(f"Línea {i+1}: {line.strip()}")
        return mentions

    def _find_related_crops(self, text: str) -> List[str]:
        text_lower = text.lower()
        return list({crop for crop in self.related_crops if crop in text_lower})

    def _find_active_ingredient(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for ingredient in self.known_ingredients:
            if ingredient in text_lower:
                return ingredient
        return None

    def _find_product_type(self, text: str) -> str:
        text_lower = text.lower()
        if 'herbicida' in text_lower:
            return 'Herbicida'
        if 'fungicida' in text_lower:
            return 'Fungicida'
        if 'insecticida' in text_lower:
            return 'Insecticida'
        if 'acaricida' in text_lower:
            return 'Acaricida'
        return 'Desconocido'

    def _find_dose_info(self, text: str) -> List[str]:
        dose_info: List[str] = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(term in line_lower for term in ['poroto', 'frijol', 'judía', 'alubia']):
                for pattern in self.dose_patterns:
                    if re.findall(pattern, line):
                        dose_info.append(f"POROTOS - Línea {i+1}: {line.strip()}")
            else:
                for pattern in self.dose_patterns:
                    if re.findall(pattern, line):
                        context_start = max(0, i - 3)
                        context_end = min(len(lines), i + 4)
                        context_text = ' '.join(lines[context_start:context_end]).lower()
                        if any(term in context_text for term in ['poroto', 'frijol', 'judía', 'alubia']):
                            dose_info.append(f"CONTEXTO POROTOS - Línea {i+1}: {line.strip()}")
                        break
        return dose_info

    def _extract_bean_doses_from_chunks(self, chunks: List[Dict]) -> List[str]:
        bean_doses: List[str] = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            chunk_text_lower = chunk_text.lower()
            metadata = chunk.get('metadata', {})

            if metadata.get('type') == 'table_row' and any(term in chunk_text_lower for term in ['poroto', 'frijol', 'judía', 'alubia']):
                lines = chunk_text.split('\n')
                cultivo_info = ''
                dosis_info = ''
                unidad_info = ''
                for line in lines:
                    norm = line.lower()
                    if 'cultivo:' in norm and any(term in norm for term in ['poroto', 'frijol', 'judía', 'alubia']):
                        cultivo_info = line.strip()
                    elif 'dosis:' in norm:
                        dosis_info = line.strip()
                    elif 'unidad:' in norm:
                        unidad_info = line.strip()
                if cultivo_info and (dosis_info or unidad_info):
                    bean_doses.append(f"RAG-TABLA-COMPLETA-{i+1} - {cultivo_info} | {dosis_info} {unidad_info}")
                elif any(re.search(pattern, chunk_text_lower) for pattern in self.dose_patterns):
                    bean_doses.append(f"RAG-TABLA-POROTOS-{i+1} - {chunk_text.strip()[:150]}...")

            elif any(term in chunk_text_lower for term in ['poroto', 'frijol', 'judía', 'alubia']):
                lines = chunk_text.split('\n')
                for line in lines:
                    norm = line.lower()
                    clean = line.strip()
                    if any(term in norm for term in ['poroto', 'frijol', 'judía', 'alubia']):
                        if any(re.search(pattern, norm) for pattern in self.dose_patterns):
                            bean_doses.append(f"RAG-POROTOS-DOSIS-{i+1} - {clean}")
                    elif 'dosis:' in norm:
                        bean_doses.append(f"RAG-DOSIS-CONTEXTO-{i+1} - {clean}")
                    elif 'unidad:' in norm and any(unit in norm for unit in ['g/ha', 'l/ha', 'kg/ha', 'ml/ha']):
                        bean_doses.append(f"RAG-UNIDAD-CONTEXTO-{i+1} - {clean}")

            elif metadata.get('type') == 'table_row':
                if any(word in chunk_text_lower for word in ['cultivo', 'aplicación', 'dosis']) and \
                   any(re.search(pattern, chunk_text_lower) for pattern in self.dose_patterns):
                    bean_doses.append(f"RAG-TABLA-GENERAL-{i+1} - {chunk_text.strip()[:100]}...")

        return bean_doses

    def _calculate_priority(self, result: EvaluationResult) -> Tuple[str, str, List[str]]:
        reasons: List[str] = []
        if self._is_already_processed(result.filename):
            return 'SKIP', 'Ya procesado anteriormente', ['Producto ya en la base de datos']

        if result.has_beans:
            reasons.append('✅ Menciona POROTOS explícitamente')
            bean_doses = [dose for dose in result.dose_info if 'POROTOS -' in dose or 'RAG-POROTOS-DOSIS' in dose]
            context_doses = [dose for dose in result.dose_info if 'CONTEXTO POROTOS' in dose or 'RAG-DOSIS-CONTEXTO' in dose]
            if result.table_rows > 0:
                reasons.append(f'✅ {result.table_rows} registros de aplicación')
                if bean_doses:
                    reasons.append(f'✅ {len(bean_doses)} dosis específicas para porotos')
                    if result.active_ingredient:
                        reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                        return 'HIGH', 'PROCESAR - Dosis para porotos + información completa', reasons
                    return 'HIGH', 'PROCESAR - Dosis para porotos (verificar principio activo)', reasons
                if context_doses:
                    reasons.append(f'✅ {len(context_doses)} dosis en contexto de porotos')
                    if result.active_ingredient:
                        reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                        return 'MEDIUM', 'PROCESAR - Dosis en contexto + principio activo', reasons
                    return 'MEDIUM', 'PROCESAR - Dosis en contexto (verificar principio activo)', reasons
                return 'MEDIUM', 'PROCESAR - Menciona porotos sin dosis específicas', reasons
            reasons.append('⚠️ Sin tablas de aplicación')
            if bean_doses:
                reasons.append(f'✅ {len(bean_doses)} dosis específicas para porotos')
                return 'MEDIUM', 'PROCESAR - Dosis para porotos pero sin tablas', reasons
            return 'LOW', 'EVALUAR - Información limitada sobre porotos', reasons

        if result.related_crops:
            reasons.append(f'⚠️ Menciona leguminosas relacionadas: {", ".join(result.related_crops)}')
            if result.active_ingredient:
                reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                if result.product_type in ['Herbicida', 'Fungicida', 'Insecticida']:
                    reasons.append(f'✅ Tipo de producto: {result.product_type}')
                    return 'MEDIUM', 'PROCESAR OPCIONAL - Leguminosas relacionadas + principio activo', reasons
                return 'LOW', 'EVALUAR MANUALMENTE - Leguminosas relacionadas sin tipo claro', reasons
            return 'LOW', 'EVALUAR MANUALMENTE - Leguminosas relacionadas sin principio activo', reasons

        reasons.append('❌ No menciona porotos ni leguminosas relevantes')
        if result.chunks_count < 5:
            reasons.append('❌ Información muy limitada')
            return 'SKIP', 'NO PROCESAR - Sin información relevante', reasons
        return 'LOW', 'EVALUAR MANUALMENTE - Documento general sin referencias claras', reasons

    def evaluate_file(self, pdf_path: str) -> EvaluationResult:
        filename = os.path.basename(pdf_path)
        print(f"🔍 Evaluando: {filename}")
        try:
            chunks = self.rag.process_pdf(pdf_path)
            chunks_count = len(chunks)
            table_rows = sum(1 for chunk in chunks if chunk.get('metadata', {}).get('type') == 'table_row')
            rag_poroto_doses = self._extract_bean_doses_from_chunks(chunks)
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
            chunks_count = 0
            table_rows = 0
            rag_poroto_doses = []

        full_text = self._extract_pdf_text(pdf_path)
        bean_mentions = self._find_bean_mentions(full_text)
        related_crops = self._find_related_crops(full_text)
        active_ingredient = self._find_active_ingredient(full_text)
        dose_info = self._find_dose_info(full_text)
        product_type = self._find_product_type(full_text)
        dose_info.extend(rag_poroto_doses)

        result = EvaluationResult(
            filename=filename,
            product_name=self._extract_product_name(filename),
            chunks_count=chunks_count,
            table_rows=table_rows,
            has_beans=len(bean_mentions) > 0,
            bean_mentions=bean_mentions,
            related_crops=related_crops,
            active_ingredient=active_ingredient,
            dose_info=dose_info,
            product_type=product_type,
            priority='',
            recommendation='',
            reasons=[]
        )

        priority, recommendation, reasons = self._calculate_priority(result)
        result.priority = priority
        result.recommendation = recommendation
        result.reasons = reasons
        return result

    def _extract_product_name(self, filename: str) -> str:
        name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        parts = name.split()
        return parts[0].upper() if parts else name.upper()

    def evaluate_directory(self, directory_path: str = './documents_porotos') -> List[EvaluationResult]:
        directory = Path(directory_path)
        pdf_files = list(directory.glob('*.pdf'))

        print(f"🔍 EVALUADOR AUTOMÁTICO DE PESTICIDAS PARA POROTOS")
        print("=" * 60)
        print(f"📁 Directorio: {directory_path}")
        print(f"📄 Archivos PDF encontrados: {len(pdf_files)}")
        print(f"🗄️ Estado actual de la base: {self.rag.count()} documentos")
        print()

        results: List[EvaluationResult] = []
        for pdf_file in pdf_files:
            try:
                result = self.evaluate_file(str(pdf_file))
                results.append(result)
                self._print_evaluation_summary(result)
                print()
            except Exception as e:
                print(f"❌ Error evaluando {pdf_file.name}: {e}")
                print()
        return results

    def _print_evaluation_summary(self, result: EvaluationResult):
        priority_colors = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🟠', 'SKIP': '🔴'}
        color = priority_colors.get(result.priority, '⚪')
        print(f"{color} {result.priority}: {result.product_name}")
        print(f"   📊 {result.chunks_count} chunks, {result.table_rows} tablas")
        print(f"   🧪 Tipo: {result.product_type}")

        if result.has_beans:
            print(f"   🫘 ✅ Menciona POROTOS ({len(result.bean_mentions)} veces)")
        elif result.related_crops:
            print(f"   🫘 ⚠️ Leguminosas relacionadas: {', '.join(result.related_crops)}")
        else:
            print(f"   ❌ Sin menciones de porotos")

        if result.active_ingredient:
            print(f"   🧪 Principio activo: {result.active_ingredient}")

        if result.dose_info:
            tabla_completa = [d for d in result.dose_info if 'RAG-TABLA-COMPLETA' in d]
            tabla_porotos = [d for d in result.dose_info if 'RAG-TABLA-POROTOS' in d]
            poroto_dosis = [d for d in result.dose_info if 'RAG-POROTOS-DOSIS' in d]
            dosis_contexto = [d for d in result.dose_info if 'RAG-DOSIS-CONTEXTO' in d]
            unidad_contexto = [d for d in result.dose_info if 'RAG-UNIDAD-CONTEXTO' in d]
            tabla_general = [d for d in result.dose_info if 'RAG-TABLA-GENERAL' in d]
            texto_dosis = [d for d in result.dose_info if 'POROTOS -' in d or 'CONTEXTO POROTOS' in d]

            if tabla_completa:
                print(f"   🎯 TABLA COMPLETA (POROTOS): {len(tabla_completa)} encontradas")
                for dose in tabla_completa[:2]:
                    clean = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      ✅ {clean}")
            if tabla_porotos:
                print(f"   📊 TABLA CON POROTOS: {len(tabla_porotos)} encontradas")
                for dose in tabla_porotos[:1]:
                    clean = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      🫘 {clean}")
            if poroto_dosis:
                print(f"   🎯 POROTOS + DOSIS: {len(poroto_dosis)} encontradas")
                for dose in poroto_dosis[:1]:
                    clean = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      ✅ {clean}")
            if dosis_contexto:
                print(f"   💊 DOSIS EN CONTEXTO: {len(dosis_contexto)} encontradas")
                for dose in dosis_contexto[:1]:
                    clean = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      📍 {clean}")
            if unidad_contexto:
                print(f"   📏 UNIDAD EN CONTEXTO: {len(unidad_contexto)} encontradas")
                for dose in unidad_contexto[:1]:
                    clean = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      📏 {clean}")
            if texto_dosis:
                print(f"   📄 TEXTO: {len(texto_dosis)} encontradas")
            if not any([tabla_completa, tabla_porotos, poroto_dosis, dosis_contexto, unidad_contexto, texto_dosis, tabla_general]):
                print(f"   ⚠️ Dosis generales: {len(result.dose_info)} (sin especificar porotos)")

        print(f"   💡 {result.recommendation}")

    def interactive_processing(self, results: List[EvaluationResult]):
        print(f"\n🎯 RESUMEN DE EVALUACIÓN")
        print("=" * 50)
        by_priority: Dict[str, List[EvaluationResult]] = {}
        for result in results:
            by_priority.setdefault(result.priority, []).append(result)
        for priority in ['HIGH', 'MEDIUM', 'LOW', 'SKIP']:
            if priority in by_priority:
                print(f"{priority}: {len(by_priority[priority])} archivos")

        print(f"\n📋 CANDIDATOS PARA PROCESAMIENTO:")
        print("=" * 40)
        candidates = [r for r in results if r.priority in ['HIGH', 'MEDIUM']]
        if not candidates:
            print("❌ No se encontraron candidatos relevantes para porotos")
            return

        for i, result in enumerate(candidates, 1):
            print(f"\n{i}. {result.product_name} ({result.filename})")
            print(f"   Prioridad: {result.priority}")
            print(f"   Tipo: {result.product_type}")
            print(f"   📊 {result.chunks_count} chunks, {result.table_rows} tablas")
            if result.bean_mentions:
                print(f"   🫘 Menciones de porotos:")
                for mention in result.bean_mentions[:2]:
                    print(f"      • {mention}")
            if result.related_crops:
                print(f"   🫘 Leguminosas relacionadas: {', '.join(result.related_crops)}")
            if result.active_ingredient:
                print(f"   🧪 Principio activo: {result.active_ingredient}")
            print(f"   💡 {result.recommendation}")

            while True:
                response = input(f"\n¿Procesar {result.product_name}? (s/n/d=detalles): ").lower().strip()
                if response == 'd':
                    self._show_detailed_info(result)
                    continue
                if response in ['s', 'si', 'y', 'yes']:
                    self._process_and_save(result)
                    break
                if response in ['n', 'no']:
                    print(f"⏭️ Omitiendo {result.product_name}")
                    break
                print("Por favor responde 's' para sí, 'n' para no, o 'd' para ver detalles")

    def _show_detailed_info(self, result: EvaluationResult):
        print(f"\n📋 INFORMACIÓN DETALLADA: {result.product_name}")
        print("=" * 50)
        print(f"Archivo: {result.filename}")
        print(f"Tipo de producto: {result.product_type}")
        print(f"Chunks extraídos: {result.chunks_count}")
        print(f"Registros de aplicación: {result.table_rows}")
        print(f"Prioridad: {result.priority}")
        print()
        if result.bean_mentions:
            print("🫘 MENCIONES DE POROTOS:")
            for mention in result.bean_mentions:
                print(f"   • {mention}")
            print()
        if result.related_crops:
            print(f"🫘 LEGUMINOSAS RELACIONADAS: {', '.join(result.related_crops)}")
            print()
        if result.active_ingredient:
            print(f"🧪 PRINCIPIO ACTIVO: {result.active_ingredient}")
            print()
        if result.dose_info:
            print("💊 INFORMACIÓN DE DOSIS:")
            for dose in result.dose_info[:3]:
                print(f"   • {dose}")
            print()
        print("📝 RAZONES DE LA EVALUACIÓN:")
        for reason in result.reasons:
            print(f"   • {reason}")
        print()

    def _process_and_save(self, result: EvaluationResult):
        print(f"\n💾 PROCESANDO Y GUARDANDO: {result.product_name}")
        print("=" * 50)
        try:
            pdf_path = f"./documents_porotos/{result.filename}"
            initial_count = self.rag.count()
            print(f"Estado inicial: {initial_count} documentos")
            chunks = self.rag.process_pdf(pdf_path)
            print(f"Chunks extraídos: {len(chunks)}")
            print("Guardando en la base de datos...")
            self.rag.add_documents(chunks)
            final_count = self.rag.count()
            added_docs = final_count - initial_count
            print("✅ COMPLETADO EXITOSAMENTE!")
            print(f"   Documentos añadidos: {added_docs}")
            print(f"   Estado final: {final_count} documentos")
            print(f"   {result.product_name} agregado al sistema")
        except Exception as e:
            print(f"❌ Error procesando {result.product_name}: {e}")


def main():
    evaluator = PorotoPesticideEvaluator()
    results = evaluator.evaluate_directory()
    evaluator.interactive_processing(results)
    print(f"\n🎉 EVALUACIÓN COMPLETADA")
    print(f"Estado final de la base: {evaluator.rag.count()} documentos")


if __name__ == "__main__":
    main()
