"""
LLM Router con Fallback Automático
Soporte para OpenAI, Anthropic, Google Gemini, Ollama y más

Autor: AgroIA System
Versión: 1.0.0
"""

import os
import logging
from typing import Dict, Any, List, Optional
import time

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("⚠️  LiteLLM no disponible. Instala con: pip install litellm")

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir logs verbosos de LiteLLM
if LITELLM_AVAILABLE:
    litellm.suppress_debug_info = True
    litellm.set_verbose = False


class LLMRouter:
    """Router inteligente de LLMs con fallback automático"""
    
    def __init__(self):
        self.name = "LLM Router"
        
        # Configuración de modelos
        self.primary_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        # Modelos de fallback (en orden de prioridad)
        fallback_list = []
        
        # Solo agregar modelos si tienen API key configurada
        if os.getenv("ANTHROPIC_API_KEY"):
            fallback_list.append("claude-3-5-sonnet-20241022")
        
        if os.getenv("GOOGLE_API_KEY"):
            fallback_list.append("gemini/gemini-1.5-pro")
        
        # Siempre intentar Ollama local como último recurso
        fallback_list.append("ollama/llama3.1")
        
        self.fallback_models = fallback_list
        
        # API Keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        
        # Ollama config
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Estadísticas
        self.stats = {
            "total_requests": 0,
            "openai_success": 0,
            "fallback_used": 0,
            "total_failures": 0,
            "last_error": None,
            "models_used": {}
        }
        
        # Configurar LiteLLM
        if LITELLM_AVAILABLE:
            litellm.api_key = self.openai_key
            if self.anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = self.anthropic_key
            if self.google_key:
                os.environ["GOOGLE_API_KEY"] = self.google_key
            
            logger.info("[LLM Router] ✅ Inicializado con LiteLLM")
            logger.info(f"[LLM Router] 🎯 Modelo principal: {self.primary_model}")
            logger.info(f"[LLM Router] 🔄 Fallbacks disponibles: {len(self.fallback_models)}")
            for i, model in enumerate(self.fallback_models, 1):
                logger.info(f"[LLM Router]    {i}. {model}")
        else:
            logger.warning("[LLM Router] ⚠️  LiteLLM no disponible - solo OpenAI directo")
    
    def generate(self, 
                messages: List[Dict[str, str]], 
                temperature: float = 0.1,
                max_tokens: int = 1800,
                timeout: int = 60) -> Dict[str, Any]:
        """
        Genera respuesta con fallback automático
        
        Args:
            messages: Lista de mensajes en formato OpenAI
            temperature: Temperatura del modelo
            max_tokens: Máximo de tokens
            timeout: Timeout en segundos
            
        Returns:
            Dict con la respuesta y metadata
        """
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        if not LITELLM_AVAILABLE:
            return self._fallback_to_openai_direct(messages, temperature, max_tokens, timeout)
        
        # Intentar con modelo principal
        try:
            logger.info(f"[LLM Router] 🚀 Usando modelo principal: {self.primary_model}")
            
            response = completion(
                model=self.primary_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=self.openai_key
            )
            
            self.stats["openai_success"] += 1
            self._track_model_usage(self.primary_model)
            elapsed = time.time() - start_time
            
            logger.info(f"[LLM Router] ✅ Respuesta exitosa en {elapsed:.2f}s")
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model_used": self.primary_model,
                "is_fallback": False,
                "elapsed_time": elapsed,
                "raw_response": response
            }
            
        except Exception as e:
            logger.warning(f"[LLM Router] ⚠️  Modelo principal falló: {e}")
            self.stats["last_error"] = str(e)
            
            # Intentar con fallbacks
            return self._try_fallbacks(messages, temperature, max_tokens, timeout, start_time)
    
    def _try_fallbacks(self, messages, temperature, max_tokens, timeout, start_time):
        """Intenta con modelos de fallback en orden"""
        
        for fallback_model in self.fallback_models:
            try:
                logger.info(f"[LLM Router] 🔄 Intentando fallback: {fallback_model}")
                
                # Configurar base URL para Ollama
                kwargs = {
                    "model": fallback_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout
                }
                
                if "ollama" in fallback_model:
                    kwargs["api_base"] = self.ollama_base_url
                
                response = completion(**kwargs)
                
                self.stats["fallback_used"] += 1
                self._track_model_usage(fallback_model)
                elapsed = time.time() - start_time
                
                logger.info(f"[LLM Router] ✅ Fallback exitoso: {fallback_model} ({elapsed:.2f}s)")
                
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "model_used": fallback_model,
                    "is_fallback": True,
                    "elapsed_time": elapsed,
                    "raw_response": response
                }
                
            except Exception as e:
                logger.warning(f"[LLM Router] ❌ Fallback {fallback_model} falló: {e}")
                continue
        
        # Todos los modelos fallaron
        self.stats["total_failures"] += 1
        elapsed = time.time() - start_time
        
        logger.error("[LLM Router] 💥 TODOS LOS MODELOS FALLARON")
        
        return {
            "success": False,
            "content": "❌ El sistema de IA no está disponible temporalmente. Por favor, intenta de nuevo en unos momentos.",
            "model_used": "none",
            "is_fallback": False,
            "elapsed_time": elapsed,
            "error": "All models failed"
        }
    
    def _fallback_to_openai_direct(self, messages, temperature, max_tokens, timeout):
        """Fallback directo a OpenAI si LiteLLM no está disponible"""
        import requests
        
        try:
            logger.info("[LLM Router] 🔧 Usando OpenAI directo (sin LiteLLM)")
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.primary_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self._track_model_usage(self.primary_model)
                return {
                    "success": True,
                    "content": data["choices"][0]["message"]["content"],
                    "model_used": self.primary_model,
                    "is_fallback": False
                }
            else:
                raise Exception(f"OpenAI API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"[LLM Router] ❌ OpenAI directo falló: {e}")
            return {
                "success": False,
                "content": "❌ Error de conexión con el servicio de IA.",
                "error": str(e)
            }
    
    def _track_model_usage(self, model_name: str):
        """Rastrea uso de cada modelo"""
        if model_name not in self.stats["models_used"]:
            self.stats["models_used"][model_name] = 0
        self.stats["models_used"][model_name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de uso"""
        success_rate = 0
        if self.stats["total_requests"] > 0:
            success_rate = ((self.stats["openai_success"] + self.stats["fallback_used"]) / 
                          self.stats["total_requests"]) * 100
        
        return {
            **self.stats,
            "success_rate": f"{success_rate:.1f}%",
            "fallback_rate": f"{(self.stats['fallback_used'] / max(1, self.stats['total_requests'])) * 100:.1f}%"
        }


# Instancia global
llm_router = LLMRouter()


# Función helper para compatibilidad con código existente
def generate_completion(messages: List[Dict[str, str]], 
                       temperature: float = 0.1,
                       max_tokens: int = 1800) -> str:
    """
    Función simple para generar completions
    Retorna solo el contenido de texto
    """
    result = llm_router.generate(messages, temperature, max_tokens)
    return result["content"]
