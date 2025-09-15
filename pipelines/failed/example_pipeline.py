metadata = {"name": "example_pipeline", "description": "Pipeline de ejemplo para Open WebUI", "type": "llm"} 
def pipeline(data): 
    print(f"[Pipeline] Ejecutando example_pipeline con data: {data}") 
    return {"result": "ok"} 
