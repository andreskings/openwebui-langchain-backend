import os
import sys
from pathlib import Path

print("=" * 60)
print("DIAGNÓSTICO DEL ENTORNO")
print("=" * 60)
print()

# 1. Verificar Python
print(f"🐍 Python: {sys.version}")
print(f"📍 Ejecutable: {sys.executable}")
print()

# 2. Verificar directorio actual
print(f"📁 Directorio actual: {os.getcwd()}")
print()

# 3. Verificar archivos necesarios
print("📄 Verificando archivos:")
files_to_check = [
    "ingest_documents.py",
    "pipelines/rag_agricultural_enhanced.py",
    "pipelines/__init__.py"
]

for file in files_to_check:
    exists = Path(file).exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")
print()

# 4. Verificar estructura de pipelines
print("📦 Estructura de pipelines/:")
pipelines_path = Path("pipelines")
if pipelines_path.exists():
    for item in pipelines_path.iterdir():
        print(f"   - {item.name}")
else:
    print("   ❌ Carpeta 'pipelines' no encontrada")
print()

# 5. Verificar sys.path
print("🔍 Python busca módulos en:")
for p in sys.path[:5]:
    print(f"   - {p}")
print()

# 6. Intentar importar el módulo
print("🧪 Intentando importar el módulo:")
try:
    # Agregar el directorio actual al path
    sys.path.insert(0, os.getcwd())
    from pipelines.rag_agricultural_enhanced import EnhancedRAGSystem
    print("   ✅ Importación exitosa!")
except ModuleNotFoundError as e:
    print(f"   ❌ ModuleNotFoundError: {e}")
except ImportError as e:
    print(f"   ❌ ImportError: {e}")
except Exception as e:
    print(f"   ❌ Error: {type(e).__name__}: {e}")
print()

# 7. Verificar contenido de ingest_documents.py
print("📖 Primeras líneas de ingest_documents.py:")
try:
    with open("ingest_documents.py", "r") as f:
        lines = f.readlines()[:15]
        for i, line in enumerate(lines, 1):
            print(f"   {i:2d}: {line.rstrip()}")
except Exception as e:
    print(f"   ❌ Error al leer archivo: {e}")
print()

print("=" * 60)
print("RECOMENDACIONES:")
print("=" * 60)
print("1. Asegúrate de estar en el directorio correcto")
print("2. Verifica que exista pipelines/rag_agricultural_enhanced.py")
print("3. Verifica que exista pipelines/__init__.py")
print("4. Instala dependencias faltantes con: pip install -r requirements.txt")
print("=" * 60)