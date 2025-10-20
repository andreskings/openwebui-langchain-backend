# ingest_documents.py - Script para procesar PDFs y almacenarlos en el vector store
import os
import sys
from pathlib import Path

# Importar el sistema RAG
from pipelines.rag_agricultural_enhanced import EnhancedRAGSystem

def ingest_pdf(pdf_path: str):
    """Ingestar un PDF al sistema RAG"""
    print(f"Procesando: {pdf_path}")
    
    # Verificar que el archivo existe
    if not os.path.exists(pdf_path):
        print(f"ERROR: El archivo no existe: {pdf_path}")
        return False
    
    # Verificar que es un PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"ERROR: El archivo no es un PDF: {pdf_path}")
        return False
    
    try:
        # Inicializar sistema RAG
        print("Inicializando sistema RAG...")
        rag_system = EnhancedRAGSystem()
        
        # Crear estructura de archivo
        file_data = {
            "name": os.path.basename(pdf_path),
            "path": pdf_path,
            "size": os.path.getsize(pdf_path),
            "type": "application/pdf"
        }
        
        print(f"Archivo: {file_data['name']} ({file_data['size']} bytes)")
        
        # Procesar archivo
        print("Extrayendo contenido y creando chunks...")
        chunks = rag_system.process_file(file_data)
        
        if chunks:
            print(f"Chunks creados: {len(chunks)}")
            
            # Añadir al vector store
            print("Almacenando en base vectorial...")
            rag_system.vector_store.add_documents(chunks)
            
            total_docs = rag_system.vector_store.count()
            print(f"✅ EXITO: {len(chunks)} chunks procesados")
            print(f"📦 Total documentos en base: {total_docs}")
            
            # Mostrar algunos chunks como ejemplo
            print("\n--- Ejemplo de chunks creados ---")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\nChunk {i+1} ID: {chunk['id']}")
                print(f"Sección: {chunk['metadata'].get('section_type', 'N/A')}")
                print(f"Relevancia agrícola: {chunk['metadata'].get('agricultural_relevance', 0):.2f}")
                preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                print(f"Contenido: {preview}")
            
            return True
        else:
            print("❌ ERROR: No se pudieron crear chunks del PDF")
            return False
            
    except Exception as e:
        print(f"❌ ERROR procesando {pdf_path}: {e}")
        import traceback
        print(f"Detalles del error:")
        traceback.print_exc()
        return False

def ingest_directory(directory_path: str):
    """Ingestar todos los PDFs de un directorio"""
    print(f"Buscando PDFs en: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"ERROR: El directorio no existe: {directory_path}")
        return
    
    pdf_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory_path, file))
    
    if not pdf_files:
        print("No se encontraron archivos PDF en el directorio")
        return
    
    print(f"Encontrados {len(pdf_files)} archivos PDF")
    
    successful = 0
    failed = 0
    
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        if ingest_pdf(pdf_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"RESUMEN:")
    print(f"✅ Procesados exitosamente: {successful}")
    print(f"❌ Fallos: {failed}")
    print(f"📊 Total: {len(pdf_files)}")

def test_system():
    """Probar que el sistema funciona con una consulta"""
    try:
        rag_system = EnhancedRAGSystem()
        total_docs = rag_system.vector_store.count()
        
        if total_docs == 0:
            print("⚠️  No hay documentos en la base vectorial")
            return
        
        print(f"📚 Documentos en base: {total_docs}")
        
        # Consulta de prueba
        test_query = "numero maximo de aplicaciones del producto zoro"
        print(f"\nProbando consulta: '{test_query}'")
        
        # Buscar y reranquear
        results = rag_system.search_and_rerank(test_query, top_k=5, rerank_k=3)
        
        if results:
            print(f"✅ Encontrados {len(results)} resultados relevantes")
            
            for i, result in enumerate(results):
                print(f"\nResultado {i+1}:")
                print(f"  ID: {result['id']}")
                print(f"  Archivo: {result['metadata'].get('source_file', 'N/A')}")
                print(f"  Similitud: {result.get('similarity', 0):.3f}")
                print(f"  Rerank Score: {result.get('rerank_score', 0):.3f}")
                preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
                print(f"  Contenido: {preview}")
        else:
            print("❌ No se encontraron resultados para la consulta")
            
    except Exception as e:
        print(f"❌ Error probando sistema: {e}")

if __name__ == "__main__":
    print("=== SCRIPT DE INGESTA DE DOCUMENTOS AGRICOLAS ===\n")
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python ingest_documents.py <archivo.pdf>")
        print("  python ingest_documents.py <directorio>")  
        print("  python ingest_documents.py test")
        print("\nEjemplos:")
        print("  python ingest_documents.py bull_cs_-_zoro_07-01-2019.pdf")
        print("  python ingest_documents.py C:/documentos/")
        print("  python ingest_documents.py test")
        sys.exit(1)
    
    target = sys.argv[1]
    
    if target.lower() == "test":
        test_system()
    elif os.path.isfile(target):
        ingest_pdf(target)
    elif os.path.isdir(target):
        ingest_directory(target)
    else:
        print(f"ERROR: '{target}' no es un archivo o directorio válido")
        sys.exit(1)
