import os
import subprocess
import sys
from pathlib import Path

# Configurar la API key de OpenAI
OPENAI_API_KEY = ""

# Directorio que contiene los documentos
DOCUMENTS_DIR = "documents"

def main():
    # Establecer la variable de entorno
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Verificar si el directorio existe
    documents_path = Path(DOCUMENTS_DIR)
    if not documents_path.exists():
        print(f"❌ Error: El directorio '{DOCUMENTS_DIR}' no existe.")
        sys.exit(1)
    
    # Buscar todos los archivos PDF
    pdf_files = list(documents_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No se encontraron archivos PDF en '{DOCUMENTS_DIR}'")
        sys.exit(1)
    
    # Contadores
    total = len(pdf_files)
    success = 0
    failed = 0
    
    print("=" * 60)
    print(f"Iniciando ingesta de {total} documentos PDF")
    print("=" * 60)
    print()
    
    # Procesar cada archivo PDF
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"[{idx}/{total}] Procesando: {pdf_file.name}")
        print("-" * 60)
        
        try:
            # Ejecutar el script de ingesta
            result = subprocess.run(
                [sys.executable, "ingest_documents.py", str(pdf_file)],
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )
            
            if result.returncode == 0:
                print(f"✅ Éxito: {pdf_file.name}")
                success += 1
            else:
                print(f"❌ Error al procesar: {pdf_file.name}")
                if result.stderr:
                    print(f"   Detalle: {result.stderr[:200]}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Excepción al procesar {pdf_file.name}: {str(e)}")
            failed += 1
        
        print()
    
    # Resumen final
    print("=" * 60)
    print("RESUMEN DE INGESTA")
    print("=" * 60)
    print(f"📊 Total de archivos: {total}")
    print(f"✅ Exitosos: {success}")
    print(f"❌ Fallidos: {failed}")
    print(f"📈 Tasa de éxito: {(success/total*100):.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
