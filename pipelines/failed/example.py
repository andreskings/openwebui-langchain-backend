# pipelines/example_pipeline.py

metadata = {
    "name": "example_pipeline",
    "description": "Pipeline de ejemplo simple para Open WebUI",
    "type": "llm"
}

def pipeline(request):
    """Pipeline simple que procesa texto"""
    try:
        # Extraer input del usuario
        user_input = ""
        if "input" in request:
            user_input = request["input"]
        elif "messages" in request and request["messages"]:
            user_input = request["messages"][-1].get("content", "")
        
        # Procesar (ejemplo simple)
        response = f"🤖 Pipeline Example procesó: '{user_input}' - ¡Funcionando correctamente!"
        
        return {
            "output": response,
            "success": True,
            "pipeline": metadata["name"]
        }
        
    except Exception as e:
        return {
            "output": f"Error en pipeline: {str(e)}",
            "success": False,
            "error": str(e)
        }