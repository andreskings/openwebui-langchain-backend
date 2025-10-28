#!/usr/bin/env python3
"""
Evaluador Automático de Pesticidas para Arroz
Procesa múltiples archivos PDF, evalúa si son relevantes para arroz
y permite al usuario decidir cuáles guardar en la base de datos.
"""

import os
import re
import fitz
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rag_arroz_optimizado import RiceRAGSystem


@dataclass
class EvaluationResult:
    """Resultado de la evaluación de un pesticida para arroz"""
    filename: str
    product_name: str
    chunks_count: int
    table_rows: int
    has_rice: bool
    rice_mentions: List[str]
    related_crops: List[str]
    active_ingredient: Optional[str]
    dose_info: List[str]
    product_type: str  # Herbicida, Fungicida, Insecticida
    priority: str  # 'HIGH', 'MEDIUM', 'LOW', 'SKIP'
    recommendation: str
    reasons: List[str]


class RicePesticideEvaluator:
    """Evaluador automático de pesticidas para arroz"""

    def __init__(self):
        self.rag = RiceRAGSystem()
        self.processed_products = self._get_processed_products()

        self.related_crops = [
            'maíz', 'sorgo', 'trigo', 'cebada', 'avena', 'centeno',
            'gramínea', 'gramíneas', 'cereal', 'cereales', 'pastos'
        ]

        self.known_ingredients = [
            # Herbicidas
            'propanil', 'bispiribac', 'oxadiazon', 'penoxsulam', 'clomazone',
            'cyhalofop', 'quinclorac', 'butaclor', 'molinate', 'acetoclor',
            'pretilachlor', 'metsulfuron', 'pyrazosulfuron', 'imazapic',
            'imazapyr', 'pendimethalin', 'thiobencarb',
            # Fungicidas
            'tricyclazole', 'propiconazole', 'azoxystrobin', 'difenoconazole',
            'isoprothiolane', 'thifluzamide', 'tebuconazole', 'fludioxonil',
            'carbendazim', 'hexaconazole',
            # Insecticidas
            'lambda-cyhalothrin', 'fenitrothion', 'etofenprox', 'thiamethoxam',
            'chlorantraniliprole', 'imidacloprid', 'buprofezin', 'fipronil',
            'phenthoate', 'malathion'
        ]

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
        return [
            'propanil', 'facet', 'command', 'strada', 'penoxsulam',
            'bispiribac', 'saturn', 'butaclor', 'molinate', 'oxadiazon'
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

    def _find_rice_mentions(self, text: str) -> List[str]:
        mentions: List[str] = []
        lines = text.split('\n')

        rice_patterns = [
            r'arroz',
            r'oryza',
            r'rice',
            r'cultivo\s+inundado',
            r'campos\s+arroceros',
            r'arroz\s+de\s+riego',
            r'arroz\s+en\s+secano'
        ]

        for i, line in enumerate(lines):
            line_lower = line.lower()
            for pattern in rice_patterns:
                if re.search(pattern, line_lower):
                    mentions.append(f"Línea {i + 1}: {line.strip()}")
                    break

        return mentions

    def _find_related_crops(self, text: str) -> List[str]:
        text_lower = text.lower()
        return sorted({crop for crop in self.related_crops if crop in text_lower})

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

            if 'arroz' in line_lower:
                for pattern in self.dose_patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        dose_info.append(f"ARROZ - Línea {i + 1}: {line.strip()}")
                        break
            else:
                for pattern in self.dose_patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        context_start = max(0, i - 3)
                        context_end = min(len(lines), i + 4)
                        context_text = ' '.join(lines[context_start:context_end]).lower()
                        if 'arroz' in context_text:
                            dose_info.append(f"CONTEXTO ARROZ - Línea {i + 1}: {line.strip()}")
                        break

        return dose_info

    def _extract_rice_doses_from_chunks(self, chunks: List[Dict]) -> List[str]:
        rice_doses: List[str] = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            chunk_text_lower = chunk_text.lower()
            metadata = chunk.get('metadata', {})

            if metadata.get('type') == 'table_row' and 'arroz' in chunk_text_lower:
                lines = chunk_text.split('\n')
                cultivo_info = ''
                dosis_info = ''
                unidad_info = ''

                for line in lines:
                    line_clean = line.strip()
                    line_lower = line_clean.lower()

                    if 'cultivo:' in line_lower and 'arroz' in line_lower:
                        cultivo_info = line_clean
                    elif 'dosis:' in line_lower:
                        dosis_info = line_clean
                    elif 'unidad:' in line_lower:
                        unidad_info = line_clean

                if cultivo_info and (dosis_info or unidad_info):
                    rice_doses.append(
                        f"RAG-TABLA-COMPLETA-{i + 1} - {cultivo_info} | {dosis_info} {unidad_info}"
                    )
                elif 'arroz' in chunk_text_lower:
                    for pattern in self.dose_patterns:
                        if re.search(pattern, chunk_text_lower):
                            rice_doses.append(
                                f"RAG-TABLA-ARROZ-{i + 1} - {chunk_text.strip()[:150]}..."
                            )
                            break

            elif 'arroz' in chunk_text_lower:
                lines = chunk_text.split('\n')
                for line in lines:
                    line_lower = line.lower()
                    line_clean = line.strip()
                    if 'arroz' in line_lower:
                        for pattern in self.dose_patterns:
                            if re.search(pattern, line_lower):
                                rice_doses.append(f"RAG-ARROZ-DOSIS-{i + 1} - {line_clean}")
                                break
                    elif 'dosis:' in line_lower:
                        rice_doses.append(f"RAG-DOSIS-CONTEXTO-{i + 1} - {line_clean}")
                    elif 'unidad:' in line_lower and any(
                        unit in line_lower for unit in ['g/ha', 'l/ha', 'kg/ha', 'ml/ha']
                    ):
                        rice_doses.append(f"RAG-UNIDAD-CONTEXTO-{i + 1} - {line_clean}")

            elif metadata.get('type') == 'table_row':
                if any(word in chunk_text_lower for word in ['cultivo', 'aplicación', 'dosis']) and any(
                    re.search(pattern, chunk_text_lower) for pattern in self.dose_patterns
                ):
                    rice_doses.append(f"RAG-TABLA-GENERAL-{i + 1} - {chunk_text.strip()[:100]}...")

        return rice_doses

    def _calculate_priority(self, result: EvaluationResult) -> Tuple[str, str, List[str]]:
        reasons: List[str] = []

        if self._is_already_processed(result.filename):
            return 'SKIP', 'Ya procesado anteriormente', ['Producto ya en la base de datos']

        if result.has_rice:
            reasons.append('✅ Menciona ARROZ explícitamente')

            arroz_doses = [dose for dose in result.dose_info if 'ARROZ -' in dose or 'RAG-ARROZ' in dose]
            context_doses = [dose for dose in result.dose_info if 'CONTEXTO ARROZ' in dose]

            if result.table_rows > 0:
                reasons.append(f'✅ {result.table_rows} registros de aplicación')

                if arroz_doses:
                    reasons.append(f'✅ {len(arroz_doses)} dosis específicas para arroz')
                    if result.active_ingredient:
                        reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                        return 'HIGH', 'PROCESAR - Dosis específicas para arroz + información completa', reasons
                    return 'HIGH', 'PROCESAR - Dosis específicas para arroz (principio activo a verificar)', reasons

                if context_doses:
                    reasons.append(f'✅ {len(context_doses)} dosis en contexto de arroz')
                    if result.active_ingredient:
                        reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                        return 'MEDIUM', 'PROCESAR - Dosis en contexto de arroz + principio activo', reasons
                    return 'MEDIUM', 'PROCESAR - Dosis en contexto de arroz (verificar principio activo)', reasons

                return 'MEDIUM', 'PROCESAR - Menciona arroz pero sin dosis específicas', reasons

            reasons.append('⚠️ Sin tablas de aplicación')
            if arroz_doses:
                reasons.append(f'✅ {len(arroz_doses)} dosis para arroz')
                return 'MEDIUM', 'PROCESAR - Dosis para arroz pero sin tablas', reasons
            return 'LOW', 'EVALUAR - Menciona arroz pero información limitada', reasons

        if result.related_crops:
            reasons.append(f'⚠️ Menciona cereales relacionados: {", ".join(result.related_crops)}')
            if result.active_ingredient:
                reasons.append(f'✅ Principio activo conocido: {result.active_ingredient}')
                if result.product_type in ['Herbicida', 'Fungicida', 'Insecticida']:
                    reasons.append(f'✅ Tipo de producto: {result.product_type}')
                    return 'MEDIUM', 'PROCESAR OPCIONAL - Cereales relacionados con principio activo conocido', reasons
                return 'LOW', 'EVALUAR MANUALMENTE - Cereales relacionados pero tipo desconocido', reasons
            return 'LOW', 'EVALUAR MANUALMENTE - Cereales relacionados pero principio activo desconocido', reasons

        reasons.append('❌ No menciona arroz ni cereales relacionados')
        if result.chunks_count < 5:
            reasons.append('❌ Información muy limitada')
            return 'SKIP', 'NO PROCESAR - Sin información relevante', reasons
        return 'LOW', 'EVALUAR MANUALMENTE - Sin menciones directas pero documento completo', reasons

    def evaluate_file(self, pdf_path: str) -> EvaluationResult:
        filename = os.path.basename(pdf_path)
        print(f"🔍 Evaluando: {filename}")

        try:
            chunks = self.rag.process_pdf(pdf_path)
            chunks_count = len(chunks)
            table_rows = sum(
                1 for chunk in chunks if chunk.get('metadata', {}).get('type') == 'table_row'
            )
            rag_rice_doses = self._extract_rice_doses_from_chunks(chunks)
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
            chunks = []
            chunks_count = 0
            table_rows = 0
            rag_rice_doses = []

        full_text = self._extract_pdf_text(pdf_path)
        rice_mentions = self._find_rice_mentions(full_text)
        related_crops = self._find_related_crops(full_text)
        active_ingredient = self._find_active_ingredient(full_text)
        dose_info = self._find_dose_info(full_text)
        product_type = self._find_product_type(full_text)
        dose_info.extend(rag_rice_doses)

        result = EvaluationResult(
            filename=filename,
            product_name=self._extract_product_name(filename),
            chunks_count=chunks_count,
            table_rows=table_rows,
            has_rice=len(rice_mentions) > 0,
            rice_mentions=rice_mentions,
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

    def evaluate_directory(self, directory_path: str = './documents_arroz') -> List[EvaluationResult]:
        directory = Path(directory_path)
        pdf_files = list(directory.glob('*.pdf'))

        print("🔍 EVALUADOR AUTOMÁTICO DE PESTICIDAS PARA ARROZ")
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

        if result.has_rice:
            print(f"   🍚 ✅ Menciona ARROZ ({len(result.rice_mentions)} veces)")
        elif result.related_crops:
            print(f"   🍚 ⚠️ Cereales relacionados: {', '.join(result.related_crops)}")
        else:
            print("   ❌ Sin menciones de arroz")

        if result.active_ingredient:
            print(f"   🧪 Principio activo: {result.active_ingredient}")

        if result.dose_info:
            tabla_completa = [dose for dose in result.dose_info if 'RAG-TABLA-COMPLETA' in dose]
            tabla_rice = [dose for dose in result.dose_info if 'RAG-TABLA-ARROZ' in dose]
            arroz_dosis = [dose for dose in result.dose_info if 'RAG-ARROZ-DOSIS' in dose]
            dosis_contexto = [dose for dose in result.dose_info if 'RAG-DOSIS-CONTEXTO' in dose]
            unidad_contexto = [dose for dose in result.dose_info if 'RAG-UNIDAD-CONTEXTO' in dose]
            tabla_general = [dose for dose in result.dose_info if 'RAG-TABLA-GENERAL' in dose]
            text_doses = [dose for dose in result.dose_info if 'ARROZ -' in dose or 'CONTEXTO ARROZ' in dose]

            if tabla_completa:
                print(f"   🎯 TABLA COMPLETA (ARROZ): {len(tabla_completa)} encontradas")
                for dose in tabla_completa[:2]:
                    clean_dose = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      ✅ {clean_dose}")

            if tabla_rice:
                print(f"   📊 TABLA CON ARROZ: {len(tabla_rice)} encontradas")
                for dose in tabla_rice[:1]:
                    clean_dose = dose.split(' - ', 1)[1] if ' - ' in dose else dose
                    print(f"      🍚 {clean_dose}")

            if arroz_dosis:
                print(f"   🎯 ARROZ + DOSIS: {len(arroz_dosis)} encontradas")
                for dose in arroz_dosis[:1]:
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

            if not any([tabla_completa, tabla_rice, arroz_dosis, dosis_contexto, unidad_contexto, text_doses]):
                print(f"   ⚠️ Dosis generales: {len(result.dose_info)} (sin especificar arroz)")

        print(f"   💡 {result.recommendation}")

    def interactive_processing(self, results: List[EvaluationResult]):
        print("\n🎯 RESUMEN DE EVALUACIÓN")
        print("=" * 50)

        by_priority: Dict[str, List[EvaluationResult]] = {}
        for result in results:
            by_priority.setdefault(result.priority, []).append(result)

        for priority in ['HIGH', 'MEDIUM', 'LOW', 'SKIP']:
            if priority in by_priority:
                count = len(by_priority[priority])
                print(f"{priority}: {count} archivos")

        print("\n📋 CANDIDATOS PARA PROCESAMIENTO:")
        print("=" * 40)

        candidates = [r for r in results if r.priority in ['HIGH', 'MEDIUM']]

        if not candidates:
            print("❌ No se encontraron candidatos relevantes para arroz")
            return

        for i, result in enumerate(candidates, 1):
            print(f"\n{i}. {result.product_name} ({result.filename})")
            print(f"   Prioridad: {result.priority}")
            print(f"   Tipo: {result.product_type}")
            print(f"   📊 {result.chunks_count} chunks, {result.table_rows} tablas")

            if result.rice_mentions:
                print("   🍚 Menciones de arroz:")
                for mention in result.rice_mentions[:2]:
                    print(f"      • {mention}")

            if result.related_crops:
                print(f"   🍚 Cereales relacionados: {', '.join(result.related_crops)}")

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

        if result.rice_mentions:
            print("🍚 MENCIONES DE ARROZ:")
            for mention in result.rice_mentions:
                print(f"   • {mention}")
            print()

        if result.related_crops:
            print(f"🍚 CEREALES RELACIONADOS: {', '.join(result.related_crops)}")
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
            pdf_path = f"./documents_arroz/{result.filename}"
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
    evaluator = RicePesticideEvaluator()
    results = evaluator.evaluate_directory()
    evaluator.interactive_processing(results)
    print("\n🎉 EVALUACIÓN COMPLETADA")
    print(f"Estado final de la base: {evaluator.rag.count()} documentos")


if __name__ == "__main__":
    main()
