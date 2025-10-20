#!/usr/bin/env python3
"""
Script para importar ChromaDB en nuevo computador.
Ejecutar después de copiar la carpeta de exportación.
"""

import sys
import os
from pathlib import Path

# Agregar directorio actual al path
sys.path.append('.')

try:
    from export_chroma_database import import_chroma_database
except ImportError:
    print("❌ Error: No se puede importar export_chroma_database.py")
    print("💡 Asegúrate de que este archivo esté en el mismo directorio")
    sys.exit(1)

def main():
    print("🚀 Importador de ChromaDB - Sistema de Lentejas")
    print("=" * 50)
    
    # Detectar directorio actual
    current_dir = Path.cwd()
    export_dir = current_dir
    
    # Verificar que estamos en directorio correcto
    json_file = export_dir / "chroma_data.json"
    if not json_file.exists():
        print(f"❌ No se encontró chroma_data.json en {export_dir}")
        print("💡 Ejecuta este script desde el directorio de exportación")
        return False
    
    # Importar base de datos
    target_path = "./chroma_lentejas_v2"
    result = import_chroma_database(export_dir, target_path)
    
    if result:
        print("\n🎉 ¡Importación exitosa!")
        print("\n📋 Próximos pasos:")
        print("1. Configurar archivo .env con tus API keys")
        print("2. Instalar dependencias: pip install -r requirements.txt")
        print("3. Probar sistema: python -c \"from pipelines.rag_agricultural_enhanced import RAGAgriculturalEnhanced; rag = RAGAgriculturalEnhanced(); print('✅ Sistema funcionando')\"")
    else:
        print("❌ Error en importación")
    
    return result

if __name__ == "__main__":
    main()
