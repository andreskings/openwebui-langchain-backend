"""
Sistema de Ingesta UNO A UNO para Máxima Calidad
Procesa archivo por archivo con revisión detallada
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
from rag_lentejas_optimized import LentilRAGSystem, HerbicideInfo

class OneByOneIngestSystem:
    """Sistema de ingesta UNO A UNO con máxima calidad"""
    
    def __init__(self):
        self.rag = LentilRAGSystem()
        self.statistics = {
            "files_processed": 0,
            "chunks_approved": 0,
            "chunks_rejected": 0,
            "herbicides_found": 0,
            "tables_found": 0,
            "dose_tables_found": 0
        }
    
    def process_one_by_one(self, documents_dir: str = "./documents"):
        """Procesa archivos UNO A UNO con máxima calidad"""
        documents_path = Path(documents_dir)
        
        if not documents_path.exists():
            print(f"❌ Directorio {documents_dir} no existe")
            return
        
        pdf_files = list(documents_path.glob("*.pdf"))
        
        print(f"🔥 SISTEMA UNO A UNO - MÁXIMA CALIDAD")
        print(f"{'='*80}")
        print(f"📁 Directorio: {documents_dir}")
        print(f"📄 Archivos encontrados: {len(pdf_files)}")
        print(f"💎 Modo: EXTRACCIÓN ULTRA-COMPLETA")
        
        if len(pdf_files) == 0:
            print("⚠️  No hay archivos PDF para procesar")
            return
        
        # Mostrar lista de archivos
        print(f"\n📋 ARCHIVOS DISPONIBLES:")
        for i, pdf_file in enumerate(pdf_files, 1):
            file_size = pdf_file.stat().st_size / 1024 / 1024  # MB
            print(f"  {i:2d}. {pdf_file.name} ({file_size:.1f} MB)")
        
        print(f"\n🎯 CARACTERÍSTICAS DEL MODO UNO A UNO:")
        print(f"  ✅ Extracción ULTRA-completa de tablas")
        print(f"  ✅ Múltiples configuraciones de extracción")
        print(f"  ✅ Interpretación automática de información crítica")
        print(f"  ✅ Revisión detallada de cada chunk")
        print(f"  ✅ Pausa entre archivos para verificar calidad")
        print(f"  ✅ Control total sobre qué se guarda")
        
        # Procesar cada archivo individualmente
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n{'🔥'*80}")
            print(f"📄 ARCHIVO {i}/{len(pdf_files)}: {pdf_file.name}")
            print(f"{'🔥'*80}")
            
            # Mostrar información del archivo
            file_size = pdf_file.stat().st_size / 1024 / 1024
            print(f"📊 Información del archivo:")
            print(f"  • Tamaño: {file_size:.1f} MB")
            print(f"  • Ruta: {pdf_file}")
            
            # Opciones para este archivo
            while True:
                print(f"\n🤔 ¿Qué hacer con {pdf_file.name}?")
                print(f"  [p] Procesar con extracción ultra-completa")
                print(f"  [v] Ver vista previa del contenido")
                print(f"  [s] Saltar este archivo")
                print(f"  [q] Salir del sistema")
                
                choice = input("Opción: ").lower().strip()
                
                if choice == 'q':
                    print("🛑 Saliendo del sistema")
                    self._show_final_stats()
                    return
                elif choice == 's':
                    print(f"⏭️  Saltando {pdf_file.name}")
                    break
                elif choice == 'v':
                    self._show_detailed_preview(str(pdf_file))
                elif choice == 'p':
                    self._process_file_ultra_complete(str(pdf_file))
                    break
                else:
                    print("❌ Opción inválida")
            
            if choice == 's':
                continue
            
            # Pausa obligatoria para verificación
            if i < len(pdf_files):
                print(f"\n⏸️  PAUSA PARA VERIFICACIÓN DE CALIDAD")
                print(f"{'─'*60}")
                print(f"📊 Archivo procesado: {pdf_file.name}")
                self._show_current_stats()
                print(f"🔍 Verifica que la información extraída sea correcta")
                print(f"💾 Revisa el estado de la base de datos")
                
                input(f"\n⏯️  Presiona ENTER para continuar al siguiente archivo...")
        
        # Estadísticas finales
        self._show_final_stats()
    
    def _process_file_ultra_complete(self, pdf_path: str):
        """Procesa un archivo con extracción ultra-completa"""
        filename = os.path.basename(pdf_path)
        
        print(f"\n🚀 INICIANDO EXTRACCIÓN ULTRA-COMPLETA")
        print(f"{'─'*60}")
        print(f"📄 Archivo: {filename}")
        print(f"⏱️  Esto puede tomar unos momentos...")
        
        try:
            # Extraer con máxima completitud
            chunks = self.rag.process_pdf(pdf_path)
            
            if not chunks:
                print(f"❌ No se pudieron extraer chunks de {filename}")
                return False
            
            print(f"\n📊 EXTRACCIÓN COMPLETADA")
            print(f"{'─'*40}")
            print(f"✅ Total chunks extraídos: {len(chunks)}")
            
            # Analizar tipos de chunks
            table_chunks = [c for c in chunks if c.get('metadata', {}).get('type') == 'table']
            text_chunks = [c for c in chunks if c.get('metadata', {}).get('type') == 'text']
            dose_tables = [c for c in table_chunks if c.get('metadata', {}).get('is_dose_table')]
            
            print(f"📋 Chunks de tablas: {len(table_chunks)}")
            print(f"📝 Chunks de texto: {len(text_chunks)}")
            print(f"💊 Tablas de dosis: {len(dose_tables)}")
            
            # Mostrar información detallada
            self._show_extraction_details(chunks, filename)
            
            # Decidir qué hacer con los chunks
            return self._review_extracted_chunks(chunks, filename)
            
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _show_detailed_preview(self, pdf_path: str):
        """Muestra vista previa ultra-detallada"""
        filename = os.path.basename(pdf_path)
        
        print(f"\n👁️  VISTA PREVIA DETALLADA: {filename}")
        print(f"{'='*60}")
        
        try:
            # Información básica del archivo
            import fitz
            doc = fitz.open(pdf_path)
            
            print(f"📊 INFORMACIÓN BÁSICA:")
            print(f"  • Páginas: {doc.page_count}")
            print(f"  • Tamaño: {os.path.getsize(pdf_path) / 1024 / 1024:.1f} MB")
            
            # Muestra de contenido de primeras páginas
            print(f"\n📄 CONTENIDO DE PRIMERAS PÁGINAS:")
            for page_num in range(min(3, doc.page_count)):
                page = doc[page_num]
                text = page.get_text()[:300]
                print(f"\n  Página {page_num + 1}:")
                print(f"  {text}...")
            
            doc.close()
            
            # Detectar tablas rápidamente
            print(f"\n📋 ANÁLISIS RÁPIDO DE TABLAS:")
            tables = self.rag.extract_tables_from_pdf(pdf_path)
            
            if tables:
                print(f"  ✅ Tablas encontradas: {len(tables)}")
                for i, table in enumerate(tables[:5], 1):
                    print(f"    {i}. Página {table['page']} - {'Dosis' if table.get('is_dose_table') else 'General'}")
                    if table.get('critical_info'):
                        print(f"       Info crítica: {table['critical_info'][:100]}...")
            else:
                print(f"  ❌ No se encontraron tablas")
            
            # Detectar herbicidas rápidamente
            print(f"\n🌿 ANÁLISIS RÁPIDO DE HERBICIDAS:")
            sample_text = ""
            doc = fitz.open(pdf_path)
            for page_num in range(min(5, doc.page_count)):
                sample_text += doc[page_num].get_text()
            doc.close()
            
            herbicides = self.rag.extract_herbicide_info(sample_text, filename)
            if herbicides:
                print(f"  ✅ Herbicidas detectados: {len(herbicides)}")
                for herb in herbicides[:3]:
                    print(f"    • {herb.name}")
                    if herb.dose_range:
                        print(f"      Dosis: {herb.dose_range} {herb.dose_unit}")
                    if herb.controlled_weeds:
                        print(f"      Malezas: {', '.join(herb.controlled_weeds[:2])}")
            else:
                print(f"  ❌ No se detectaron herbicidas")
            
        except Exception as e:
            print(f"❌ Error en vista previa: {e}")
        
        input(f"\n📖 Presiona ENTER para volver al menú...")
    
    def _show_extraction_details(self, chunks: List[Dict[str, Any]], filename: str):
        """Muestra detalles de la extracción"""
        print(f"\n📋 DETALLES DE EXTRACCIÓN PARA: {filename}")
        print(f"{'─'*60}")
        
        # Estadísticas por tipo
        herbicides_total = set()
        doses_found = 0
        weeds_found = set()
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            
            # Recopilar herbicidas
            herbicide_names = metadata.get('herbicide_names', [])
            herbicides_total.update(herbicide_names)
            
            # Contar dosis
            if metadata.get('has_dose_info'):
                doses_found += 1
            
            # Recopilar malezas
            controlled_weeds = metadata.get('controlled_weeds', [])
            weeds_found.update(controlled_weeds)
        
        print(f"🌿 HERBICIDAS ÚNICOS DETECTADOS: {len(herbicides_total)}")
        for herbicide in sorted(herbicides_total):
            if herbicide:
                print(f"  • {herbicide}")
        
        print(f"\n💊 INFORMACIÓN DE DOSIS:")
        print(f"  • Chunks con dosis: {doses_found}")
        
        print(f"\n🌱 MALEZAS MENCIONADAS: {len(weeds_found)}")
        for weed in sorted(weeds_found):
            if weed:
                print(f"  • {weed}")
        
        # Mostrar chunks más relevantes
        print(f"\n⭐ TOP 3 CHUNKS MÁS RELEVANTES:")
        sorted_chunks = sorted(chunks, key=lambda x: x.get('metadata', {}).get('content_score', 0), reverse=True)
        
        for i, chunk in enumerate(sorted_chunks[:3], 1):
            metadata = chunk.get('metadata', {})
            score = metadata.get('content_score', 0)
            chunk_type = metadata.get('type', 'N/A')
            herbicides = len(metadata.get('herbicide_names', []))
            
            print(f"  {i}. Tipo: {chunk_type} | Score: {score:.1f} | Herbicidas: {herbicides}")
            
            # Muestra del contenido
            text = chunk.get('text', '')[:150]
            print(f"     Contenido: {text}...")
    
    def _review_extracted_chunks(self, chunks: List[Dict[str, Any]], filename: str) -> bool:
        """Revisa los chunks extraídos"""
        print(f"\n🤔 ¿QUÉ HACER CON LOS CHUNKS EXTRAÍDOS?")
        print(f"{'─'*50}")
        
        while True:
            print(f"\nOpciones:")
            print(f"  [a] Aprobar TODO (guardar todos los chunks)")
            print(f"  [r] Revisar chunk por chunk")
            print(f"  [t] Revisar solo tablas importantes")
            print(f"  [x] Rechazar TODO (no guardar nada)")
            print(f"  [v] Ver estadísticas detalladas")
            
            choice = input("Opción: ").lower().strip()
            
            if choice == 'a':
                self.rag.add_documents(chunks)
                self.statistics["chunks_approved"] += len(chunks)
                self.statistics["files_processed"] += 1
                
                # Actualizar estadísticas específicas
                table_chunks = [c for c in chunks if c.get('metadata', {}).get('type') == 'table']
                dose_tables = [c for c in table_chunks if c.get('metadata', {}).get('is_dose_table')]
                
                self.statistics["tables_found"] += len(table_chunks)
                self.statistics["dose_tables_found"] += len(dose_tables)
                
                print(f"✅ TODOS los chunks aprobados y guardados")
                return Trueprint(f"\n📄 CHUNK {i}/{len(chunks)}")
            print(f"  Tipo: {metadata.get('type', 'N/A')}")
            print(f"  Página: {metadata.get('page_number', 'N/A')}")
            print(f"  Herbicidas: {metadata.get('herbicides_found', 0)}")
            print(f"  Score: {metadata.get('content_score', 0):.1f}")
            
                
            elif choice == 'r':
                return self._review_individual_chunks(chunks, filename)
                
            elif choice == 't':
                return self._review_important_tables(chunks, filename)
                
            elif choice == 'x':
                self.statistics["chunks_rejected"] += len(chunks)
                print(f"❌ TODOS los chunks rechazados")
                return False
                
            elif choice == 'v':
                self._show_detailed_chunk_stats(chunks)
                
            else:
                print("❌ Opción inválida")
    
    def _review_important_tables(self, chunks: List[Dict[str, Any]], filename: str) -> bool:
        """Revisa solo las tablas importantes"""
        table_chunks = [c for c in chunks if c.get('metadata', {}).get('type') == 'table']
        text_chunks = [c for c in chunks if c.get('metadata', {}).get('type') == 'text']
        
        print(f"\n📋 REVISIÓN DE TABLAS IMPORTANTES")
        print(f"{'─'*50}")
        print(f"📊 Tablas encontradas: {len(table_chunks)}")
        print(f"📝 Chunks de texto: {len(text_chunks)}")
        
        approved_chunks = []
        
        # Aprobar automáticamente chunks de texto
        approved_chunks.extend(text_chunks)
        print(f"✅ Auto-aprobados {len(text_chunks)} chunks de texto")
        
        # Revisar tablas una por una
        for i, chunk in enumerate(table_chunks, 1):
            metadata = chunk.get('metadata', {})
            
            print(f"\n📋 TABLA {i}/{len(table_chunks)}")
            print(f"  Página: {metadata.get('page', 'N/A')}")
            print(f"  Es tabla de dosis: {'✅' if metadata.get('is_dose_table') else '❌'}")
            
            if metadata.get('critical_info'):
                print(f"  Info crítica: {metadata['critical_info']}")
            
            # Muestra del contenido
            text = chunk.get('text', '')
            print(f"\n📄 Contenido (primeros 300 chars):")
            print(text[:300] + "...")
            
            while True:
                choice = input(f"\n¿Aprobar esta tabla? (s/n/v=ver completa): ").lower().strip()
                
                if choice == 's':
                    approved_chunks.append(chunk)
                    print("✅ Tabla aprobada")
                    break
                elif choice == 'n':
                    print("❌ Tabla rechazada")
                    break
                elif choice == 'v':
                    print(f"\n📖 CONTENIDO COMPLETO:")
                    print("=" * 80)
                    print(text)
                    print("=" * 80)
                else:
                    print("❌ Opción inválida")
        
        # Guardar chunks aprobados
        if approved_chunks:
            self.rag.add_documents(approved_chunks)
            self.statistics["chunks_approved"] += len(approved_chunks)
            self.statistics["files_processed"] += 1
            print(f"\n✅ Guardados {len(approved_chunks)} chunks")
            return True
        else:
            print(f"\n❌ No se guardó ningún chunk")
            return False
    
    def _review_individual_chunks(self, chunks: List[Dict[str, Any]], filename: str) -> bool:
        """Revisa cada chunk individualmente"""
        print(f"\n🔍 REVISIÓN INDIVIDUAL DE CHUNKS")
        print(f"{'─'*50}")
        
        approved_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            text = chunk.get('text', '')
            
            print(f"\n📄 CHUNK {i}/{len(chunks)}")
            print(f"  Tipo: {metadata.get('type', 'N/A')}")
            print(f"  Página: {metadata.get('page_number', 'N/A')}")
            print(f"  Herbicidas: {metadata.get('herbicides_found', 0)}")
            print(f"  Score: {metadata.get('content_score', 0):.1f}")
            
            print(f"\n📝 Contenido (primeros 200 chars):")
            print(text[:200] + "...")
            
            while True:
                choice = input(f"\n¿Qué hacer? (a=aprobar/r=rechazar/v=ver completo/s=saltar resto): ").lower().strip()
                
                if choice == 'a':
                    approved_chunks.append(chunk)
                    print("✅ Chunk aprobado")
                    break
                elif choice == 'r':
                    print("❌ Chunk rechazado")
                    break
                elif choice == 'v':
                    print(f"\n📖 CONTENIDO COMPLETO:")
                    print("=" * 80)
                    print(text)
                    print("=" * 80)
                elif choice == 's':
                    # Aprobar todos los restantes
                    remaining = chunks[i-1:]
                    approved_chunks.extend(remaining)
                    print(f"✅ Aprobados {len(remaining)} chunks restantes")
                    break
                else:
                    print("❌ Opción inválida")
            
            if choice == 's':
                break
        
        # Guardar chunks aprobados
        if approved_chunks:
            self.rag.add_documents(approved_chunks)
            self.statistics["chunks_approved"] += len(approved_chunks)
            self.statistics["files_processed"] += 1
            print(f"\n✅ Guardados {len(approved_chunks)} chunks")
            return True
        else:
            print(f"\n❌ No se guardó ningún chunk")
            return False
    
    def _show_detailed_chunk_stats(self, chunks: List[Dict[str, Any]]):
        """Muestra estadísticas detalladas de chunks"""
        print(f"\n📊 ESTADÍSTICAS DETALLADAS DE CHUNKS")
        print(f"{'='*60}")
        
        # Por tipo
        types = {}
        for chunk in chunks:
            chunk_type = chunk.get('metadata', {}).get('type', 'unknown')
            types[chunk_type] = types.get(chunk_type, 0) + 1
        
        print(f"📋 Por tipo:")
        for chunk_type, count in types.items():
            print(f"  • {chunk_type}: {count}")
        
        # Por score
        scores = [chunk.get('metadata', {}).get('content_score', 0) for chunk in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"\n⭐ Scores de relevancia:")
        print(f"  • Promedio: {avg_score:.1f}")
        print(f"  • Máximo: {max(scores):.1f}")
        print(f"  • Mínimo: {min(scores):.1f}")
        
        # Con información crítica
        with_dose = len([c for c in chunks if c.get('metadata', {}).get('has_dose_info')])
        with_weeds = len([c for c in chunks if c.get('metadata', {}).get('has_weed_info')])
        with_timing = len([c for c in chunks if c.get('metadata', {}).get('has_timing_info')])
        
        print(f"\n💊 Con información crítica:")
        print(f"  • Con dosis: {with_dose}")
        print(f"  • Con malezas: {with_weeds}")
        print(f"  • Con timing: {with_timing}")
    
    def _show_current_stats(self):
        """Muestra estadísticas actuales"""
        print(f"\n📊 ESTADÍSTICAS ACTUALES:")
        print(f"  📁 Archivos procesados: {self.statistics['files_processed']}")
        print(f"  ✅ Chunks aprobados: {self.statistics['chunks_approved']}")
        print(f"  ❌ Chunks rechazados: {self.statistics['chunks_rejected']}")
        print(f"  📋 Tablas encontradas: {self.statistics['tables_found']}")
        print(f"  💊 Tablas de dosis: {self.statistics['dose_tables_found']}")
        
        total_docs = self.rag.count()
        print(f"  💾 Total en base de datos: {total_docs} documentos")
    
    def _show_final_stats(self):
        """Muestra estadísticas finales"""
        print(f"\n🎉 PROCESAMIENTO UNO A UNO COMPLETADO")
        print(f"{'='*80}")
        print(f"📊 ESTADÍSTICAS FINALES:")
        print(f"  📁 Archivos procesados: {self.statistics['files_processed']}")
        print(f"  ✅ Chunks aprobados: {self.statistics['chunks_approved']}")
        print(f"  ❌ Chunks rechazados: {self.statistics['chunks_rejected']}")
        print(f"  📋 Tablas encontradas: {self.statistics['tables_found']}")
        print(f"  💊 Tablas de dosis: {self.statistics['dose_tables_found']}")
        
        total_docs = self.rag.count()
        print(f"  💾 Total en base de datos: {total_docs} documentos")
        
        if self.statistics['chunks_approved'] > 0:
            approval_rate = (self.statistics['chunks_approved'] / 
                           (self.statistics['chunks_approved'] + self.statistics['chunks_rejected'])) * 100
            print(f"  📈 Tasa de aprobación: {approval_rate:.1f}%")
        
        print(f"\n💎 CALIDAD GARANTIZADA: Sistema RAG con información verificada manualmente")
        print(f"🎯 Listo para consultas sobre herbicidas para lentejas!")

def main():
    """Función principal del sistema UNO A UNO"""
    print("🔥 SISTEMA UNO A UNO - MÁXIMA CALIDAD PARA LENTEJAS")
    print("=" * 80)
    
    ingest_system = OneByOneIngestSystem()
    
    # Verificar estado inicial
    initial_docs = ingest_system.rag.count()
    print(f"📊 Estado inicial: {initial_docs} documentos en la base")
    
    # Procesar UNO A UNO
    ingest_system.process_one_by_one()

if __name__ == "__main__":
    main()
