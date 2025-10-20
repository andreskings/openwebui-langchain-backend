#!/usr/bin/env python3
"""
Script para importar ChromaDB de TRIGO en nuevo computador.
Ejecutar después de copiar la carpeta de exportación.
"""

import sys
import os
from pathlib import Path

# Agregar directorio actual al path
sys.path.append('.')

try:
    from export_trigo_database import import_wheat_database
except ImportError:
    print("❌ Error: No se puede importar export_trigo_database.py")
    print("💡 Asegúrate de que este archivo esté en el mismo directorio")
    sys.exit(1)

def main():
    print("🌾 Importador de ChromaDB - Sistema de TRIGO")
    print("=" * 50)
    
    # Detectar directorio actual
    current_dir = Path.cwd()
    export_dir = current_dir
    
    # Verificar que estamos en directorio correcto
    json_file = export_dir / "trigo_data.json"
    if not json_file.exists():
        print(f"❌ No se encontró trigo_data.json en {export_dir}")
        print("💡 Ejecuta este script desde el directorio de exportación")
        return False
    
    # Importar base de datos DE TRIGO
    target_path = "./chroma_trigo_v1"
    result = import_wheat_database(export_dir, target_path)
    
    if result:
        print("\n🎉 ¡Importación de TRIGO exitosa!")
        print("\n📋 Próximos pasos:")
        print("1. Configurar archivo .env con tus API keys")
        print("2. Instalar dependencias: pip install -r requirements.txt")
        print("3. Probar sistema: python -c \"from rag_trigo_optimizado import WheatRAGSystem; rag = WheatRAGSystem(); print('✅ Sistema de trigo funcionando')\"")
    else:
        print("❌ Error en importación de trigo")
    
    return result

if __name__ == "__main__":
    main()
