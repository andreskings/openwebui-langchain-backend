import importlib.util
import sys

def load_pipelines():
    """
    Carga automáticamente todos los pipelines desde la carpeta 'pipelines'.
    Requiere que cada archivo tenga:
        - pipeline (función)
        - metadata (diccionario con al menos 'name')
    """
    global pipelines_dict
    pipelines_dict.clear()

    PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "pipelines")

    if not os.path.exists(PIPELINES_DIR):
        print(f"[Warning] Directorio {PIPELINES_DIR} no existe")
        return

    for root, dirs, files in os.walk(PIPELINES_DIR):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = os.path.join(root, file)
                # Crear un nombre único de módulo basado en la ruta relativa
                module_name = os.path.relpath(file_path, PIPELINES_DIR).replace(os.sep, ".")[:-3]

                try:
                    # Cargar módulo desde ruta
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Verificar que tiene pipeline y metadata
                    if hasattr(module, "pipeline") and hasattr(module, "metadata"):
                        key_name = module.metadata.get("name", file[:-3])
                        pipelines_dict[key_name] = {
                            "func": module.pipeline,
                            "metadata": module.metadata
                        }
                        print(f"✅ Pipeline {key_name} cargado: {module.metadata.get('description', 'Sin descripción')}")
                    else:
                        print(f"⚠️  {file} no tiene 'pipeline' o 'metadata'")

                except Exception as e:
                    print(f"❌ Error cargando {file}: {e}")

    print(f"🔄 Total pipelines cargados: {len(pipelines_dict)}")
    print(f"📋 Pipelines disponibles: {list(pipelines_dict.keys())}")
