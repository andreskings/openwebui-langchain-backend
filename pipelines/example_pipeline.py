# pipelines/example_pipeline.py
from datetime import datetime

metadata = {
    "name": "example_pipeline",
    "description": "Pipeline de ejemplo con procesamiento de chat",
    "type": "llm"
}

def pipeline(data: dict):
    """
    Pipeline de ejemplo que procesa mensajes de chat
    """
    print(f"[Example Pipeline] Ejecutando con data: {data}")
    
    try:
        # Extraer mensajes si existen
        messages = data.get("messages", [])
        user_input = data.get("user_input", "")
        
        # Si hay mensajes, tomar el último mensaje del usuario
        if messages:
            last_message = messages[-1]
            if last_message.get("role") == "user":
                user_input = last_message.get("content", "")
        
        # Generar una respuesta simple basada en el input
        if not user_input:
            response_text = "¡Hola! Soy el pipeline de ejemplo. ¿En qué puedo ayudarte?"
        elif "hola" in user_input.lower():
            response_text = f"¡Hola! Recibí tu mensaje: '{user_input}'. Es un placer saludarte."
        elif "tiempo" in user_input.lower() or "hora" in user_input.lower():
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response_text = f"La hora actual es: {current_time}"
        else:
            response_text = f"Procesé tu mensaje: '{user_input}'. Este es un pipeline de ejemplo que simplemente repite y confirma lo que escribiste."
        
        result = {
            "success": True,
            "response": response_text,
            "processed_input": user_input,
            "timestamp": datetime.now().isoformat(),
            "pipeline_name": "example_pipeline"
        }
        
        print(f"[Example Pipeline] Respuesta: {response_text}")
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "response": f"Error en el pipeline de ejemplo: {str(e)}"
        }
        print(f"[Example Pipeline] Error: {e}")
        return error_result