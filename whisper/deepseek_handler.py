#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenRouter Handler - Corrección de texto en inglés
Integración con Qwen API via OpenRouter para mejorar la calidad del texto traducido
"""

import logging
import time
import threading
import queue
from typing import Dict, Any, Optional, Callable
from openai import OpenAI
import hashlib
import json
import os
from datetime import datetime, timedelta

class DeepSeekHandler:
    """Manejador para corrección de texto usando Qwen API via OpenRouter"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Inicializar OpenRouter Handler"""
        self.logger = logging.getLogger(__name__)
        
        # Obtener configuración
        if config_dict is None:
            from config import config
            config_dict = config.get_deepseek_config()
        
        # Configuración
        self.use_dual_keys = config_dict.get('use_dual_keys', False)
        self.api_key_free = config_dict.get('api_key_free', '')
        self.api_key_paid = config_dict.get('api_key_paid', '')
        # Compatibilidad con configuración antigua
        self.api_key = config_dict.get('api_key', self.api_key_free)
        
        # Modelos separados para API gratuita y de pago
        self.model_free = config_dict.get('model_free', 'qwen/qwen3-235b-a22b-2507:free')
        self.model_paid = config_dict.get('model_paid', 'qwen/qwen3-235b-a22b-2507')
        
        # Mantener compatibilidad con configuración antigua
        self.model = config_dict.get('model', self.model_free)
        self.fallback_model = config_dict.get('fallback_model', self.model_paid)
        self.base_url = config_dict.get('base_url', 'https://openrouter.ai/api/v1')
        self.use_openrouter = config_dict.get('use_openrouter', True)
        self.enabled = config_dict.get('enabled', True)  # Habilitado por defecto con OpenRouter
        self.max_retries = config_dict.get('max_retries', 3)
        self.timeout = config_dict.get('timeout', 30)
        
        # Estado actual de la API key
        self.current_api_key = self._get_initial_api_key()
        self.using_paid_key = False
        
        # Cliente OpenAI para Qwen
        self.client = None
        self.client_paid = None  # Cliente separado para API key de pago
        
        if self.enabled:
            self._initialize_clients()
        
        # Sistema de corrección asíncrona
        self.correction_queue = queue.Queue(maxsize=10)
        self.correction_thread = None
        self.is_processing = False
        
        # Callbacks
        self.on_correction_ready = None
        self.on_error = None
        self.on_progress = None
        
        # Estadísticas
        self.stats = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'failed_corrections': 0,
            'average_response_time': 0.0,
            'total_characters_processed': 0
        }
        
        # Prompt del sistema para traducción directa
        self.system_prompt = (
            "You are a professional Spanish-to-English translator. Your task is to:"
            "1. Translate the Spanish text to natural, fluent English"
            "2. Ensure perfect grammar, spelling, and punctuation"
            "3. Use natural English expressions and idioms when appropriate"
            "4. Maintain the EXACT original meaning and intent"
            "5. Preserve the original tone and style"
            "6. Return ONLY the English translation, no explanations or additions"
            "\n\nIMPORTANT RULES:"
            "- Input will ALWAYS be in Spanish"
            "- Output must ALWAYS be in English"
            "- Do NOT add any commentary or explanations"
            "- Do NOT change the core meaning or add new information"
            "- Focus on producing natural, fluent English that sounds native"
            "- Handle colloquialisms and informal speech appropriately"
        )

        # Cache configuration
        self.enable_cache = True
        self.cache_dir = os.path.join(os.path.dirname(__file__), '.cache', 'corrections')
        self.cache_ttl_hours = 24 * 7  # 1 week cache
        self.max_cache_size = 1000  # Maximum number of cached translations
        
        # Initialize cache directory
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._cleanup_old_cache()
    
    def _get_initial_api_key(self) -> str:
        """Determinar qué API key usar inicialmente"""
        if self.use_dual_keys:
            # Priorizar API key gratuita si está disponible
            if self.api_key_free and self.api_key_free != "sk-or-v1-your-openrouter-api-key-here":
                return self.api_key_free
            elif self.api_key_paid and self.api_key_paid != "sk-or-v1-your-openrouter-api-key-here":
                self.using_paid_key = True
                return self.api_key_paid
        else:
            # Usar configuración antigua
            if self.api_key and self.api_key != "sk-or-v1-your-openrouter-api-key-here":
                return self.api_key
            elif self.api_key_free and self.api_key_free != "sk-or-v1-your-openrouter-api-key-here":
                return self.api_key_free
        return ""
    
    def _initialize_clients(self):
        """Inicializar clientes OpenAI para ambas API keys"""
        try:
            if not self.use_openrouter:
                self.logger.warning("Qwen solo está disponible via OpenRouter")
                self.enabled = False
                return
            
            # Inicializar cliente principal (gratuito o único)
            if self.current_api_key:
                self.client = OpenAI(
                    api_key=self.current_api_key,
                    base_url=self.base_url
                )
                key_type = "pago" if self.using_paid_key else "gratuita"
                current_model = self.model_paid if self.using_paid_key else self.model_free
                self.logger.info(f"Qwen client inicializado via OpenRouter (API {key_type}, modelo: {current_model})")
            
            # Inicializar cliente de pago si está disponible y es diferente
            if (self.use_dual_keys and self.api_key_paid and 
                self.api_key_paid != "sk-or-v1-your-openrouter-api-key-here" and
                self.api_key_paid != self.current_api_key):
                
                self.client_paid = OpenAI(
                    api_key=self.api_key_paid,
                    base_url=self.base_url
                )
                self.logger.info("Cliente de pago Qwen inicializado como fallback")
            
            if not self.client:
                self.logger.warning("No se pudo inicializar ningún cliente Qwen - verifica tus API keys")
                self.enabled = False
                
        except Exception as e:
            self.logger.error(f"Error inicializando clientes Qwen: {e}")
            self.enabled = False
    
    def _switch_to_paid_key(self) -> bool:
        """Cambiar a la API key de pago si está disponible"""
        if (self.use_dual_keys and not self.using_paid_key and 
            self.api_key_paid and self.api_key_paid != "sk-or-v1-your-openrouter-api-key-here"):
            
            self.logger.info("Cambiando a API key de pago debido a límites en la gratuita")
            self.current_api_key = self.api_key_paid
            self.using_paid_key = True
            
            # Usar cliente de pago si ya está inicializado, sino crear uno nuevo
            if self.client_paid:
                self.client = self.client_paid
            else:
                try:
                    self.client = OpenAI(
                        api_key=self.current_api_key,
                        base_url=self.base_url
                    )
                except Exception as e:
                    self.logger.error(f"Error creando cliente de pago: {e}")
                    return False
            
            return True
        return False
    
    @property
    def is_available(self) -> bool:
        """Verificar si Qwen está disponible"""
        if self.use_openrouter:
            # Con OpenRouter, necesitamos API key válida y cliente
            return (self.enabled and self.client is not None and 
                   bool(self.api_key) and self.api_key != "sk-or-v1-your-openrouter-api-key-here")
        else:
            # Qwen solo está disponible via OpenRouter
            return False
    
    def correct_text(self, text: str) -> Dict[str, Any]:
        """
        Corregir y traducir texto español usando OpenRouter/Qwen
        """
        # Check cache first
        cached_translation = self._load_from_cache(text)
        if cached_translation:
            return {
                'original_text': text,
                'corrected_text': cached_translation,
                'success': True,
                'error': None,
                'processing_time': 0.0,
                'from_cache': True
            }
        
        if not self.client:
            self.logger.error("Cliente OpenRouter no configurado")
            return {
                'original_text': text,
                'corrected_text': text,
                'success': False,
                'error': "Cliente OpenRouter no configurado",
                'processing_time': 0.0
            }
        
        start_time = time.time()
        # Usar el modelo apropiado según la API key actual
        current_model = self.model_paid if self.using_paid_key else self.model_free
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Corrigiendo texto con {current_model} (intento {attempt + 1}): {text[:50]}...")
                
                # Preparar headers adicionales para OpenRouter
                extra_headers = {}
                extra_body = {}
                
                if self.use_openrouter:
                    extra_headers = {
                        "HTTP-Referer": "https://voice-translator.local",
                        "X-Title": "Voice Translator AI"
                    }
                
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Correct this text: {text}"}
                    ],
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                    stream=False,
                    timeout=self.timeout
                )
                
                corrected_text = response.choices[0].message.content.strip()
                processing_time = time.time() - start_time
                
                # Actualizar estadísticas
                self.stats['total_corrections'] += 1
                self.stats['successful_corrections'] += 1
                self.stats['total_characters_processed'] += len(text)
                self._update_stats(processing_time)
                
                self.logger.info(f"Texto corregido con {current_model} en {processing_time:.2f}s")
                
                # Save to cache
                self._save_to_cache(text, corrected_text)
                
                return {
                    'original_text': text,
                    'corrected_text': corrected_text,
                    'success': True,
                    'error': None,
                    'processing_time': processing_time,
                    'model_used': current_model
                }
                
            except Exception as e:
                error_str = str(e)
                self.logger.warning(f"Error en intento {attempt + 1} con {current_model}: {e}")
                
                # Detectar errores de límite de requests/quota
                is_rate_limit_error = ("rate limit" in error_str.lower() or 
                                     "quota" in error_str.lower() or 
                                     "free" in error_str.lower() or
                                     "insufficient credits" in error_str.lower() or
                                     "429" in error_str)
                
                # Si es error de límite y no estamos usando la API key de pago
                if is_rate_limit_error and not self.using_paid_key:
                    # Intentar cambiar a API key de pago
                    if self._switch_to_paid_key():
                        self.logger.info("Reintentando con API key de pago")
                        continue  # Reintentar inmediatamente con la API key de pago
                    
                    # Si no hay API key de pago, intentar modelo de pago con la misma key
                    elif current_model == self.model_free and self.model_paid:
                        self.logger.info(f"Límite alcanzado, cambiando a modelo de pago: {self.model_paid}")
                        current_model = self.model_paid
                        continue  # Reintentar con el modelo de pago
                
                if attempt == self.max_retries - 1:
                    # Último intento fallido
                    self.stats['total_corrections'] += 1
                    self.stats['failed_corrections'] += 1
                    
                    self.logger.error(f"Error corrigiendo texto después de {self.max_retries} intentos: {e}")
                    
                    return {
                        'original_text': text,
                        'corrected_text': text,  # Devolver texto original como fallback
                        'success': False,
                        'error': str(e),
                        'processing_time': time.time() - start_time
                    }
                
                # Esperar antes del siguiente intento
                time.sleep(0.5 * (attempt + 1))
    
    def start_correction_worker(self):
        """Iniciar worker de corrección"""
        if self.correction_thread and self.correction_thread.is_alive():
            return
        
        self.is_processing = True
        self.correction_thread = threading.Thread(target=self._correction_worker, daemon=True)
        self.correction_thread.start()
        self.logger.info("Worker de corrección iniciado")
    
    def _correction_worker(self):
        """Worker para corrección asíncrona"""
        while self.is_processing:
            try:
                # Obtener tarea de la cola
                task = self.correction_queue.get(timeout=1.0)
                
                if task is None:  # Señal de parada
                    break
                
                text, callback = task
                
                # Corregir texto
                result = self.correct_text_sync(text)
                
                # Llamar callback con resultado
                if callback:
                    callback(result)
                elif self.on_correction_ready:
                    self.on_correction_ready(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker de corrección: {e}")
                if self.on_error:
                    self.on_error(f"Error de corrección: {e}")
    
    def correct_text_async(self, text: str, callback: Optional[Callable] = None) -> bool:
        """Corregir texto de forma asíncrona"""
        if not self.is_available:
            self.logger.error("OpenRouter no disponible")
            return False
        
        if not text.strip():
            self.logger.warning("Texto vacío")
            return False
        
        try:
            # Iniciar worker si no está activo
            if not self.is_processing:
                self.start_correction_worker()
            
            # Agregar a la cola de corrección
            task = (text, callback)
            self.correction_queue.put_nowait(task)
            return True
            
        except queue.Full:
            self.logger.warning("Cola de corrección llena")
            return False
        except Exception as e:
            self.logger.error(f"Error agregando tarea de corrección: {e}")
            return False
    
    def _update_stats(self, processing_time: float):
        """Actualizar estadísticas"""
        if self.stats['successful_corrections'] > 0:
            total_time = (self.stats['average_response_time'] * (self.stats['successful_corrections'] - 1) + 
                         processing_time) / self.stats['successful_corrections']
            self.stats['average_response_time'] = total_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de uso"""
        return self.stats.copy()
    
    def set_callbacks(self, 
                     on_correction_ready: Optional[Callable] = None,
                     on_error: Optional[Callable] = None,
                     on_progress: Optional[Callable] = None):
        """Establecer callbacks"""
        if on_correction_ready:
            self.on_correction_ready = on_correction_ready
        if on_error:
            self.on_error = on_error
        if on_progress:
            self.on_progress = on_progress
    
    def stop_correction(self):
        """Detener corrección"""
        self.is_processing = False
        
        # Enviar señal de parada
        try:
            self.correction_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Esperar a que termine el thread
        if self.correction_thread and self.correction_thread.is_alive():
            self.correction_thread.join(timeout=2.0)
        
        self.logger.info("Corrección detenida")
    
    def cleanup(self):
        """Limpiar recursos"""
        self.stop_correction()
        self.client = None
        self.logger.info("OpenRouter Handler limpiado")

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the cache file path for a given cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, text: str) -> Optional[str]:
        """Load translation from cache if available and not expired"""
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid
                cached_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cached_time < timedelta(hours=self.cache_ttl_hours):
                    self.logger.info(f"Cache hit for text: '{text[:50]}...'")
                    return cache_data['translation']
                else:
                    # Cache expired, remove it
                    os.remove(cache_path)
            except Exception as e:
                self.logger.warning(f"Error loading cache: {e}")
        
        return None
    
    def _save_to_cache(self, text: str, translation: str):
        """Save translation to cache"""
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'text': text,
                'translation': translation,
                'timestamp': datetime.now().isoformat(),
                'model': self.model
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Saved to cache: '{text[:50]}...'")
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {e}")
    
    def _cleanup_old_cache(self):
        """Clean up old cache files"""
        if not self.enable_cache:
            return
        
        try:
            cache_files = []
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    cache_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Remove old files if we exceed max cache size
            if len(cache_files) > self.max_cache_size:
                for filepath, _ in cache_files[:len(cache_files) - self.max_cache_size]:
                    os.remove(filepath)
                    self.logger.debug(f"Removed old cache file: {filepath}")
            
            # Remove expired files
            current_time = time.time()
            ttl_seconds = self.cache_ttl_hours * 3600
            
            for filepath, mtime in cache_files:
                if current_time - mtime > ttl_seconds:
                    os.remove(filepath)
                    self.logger.debug(f"Removed expired cache file: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Error cleaning up cache: {e}")


if __name__ == "__main__":
    # Prueba del DeepSeek Handler
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    # Configuración de prueba (necesitas una API key válida)
    config = {
        'api_key': 'tu_api_key_aqui',
        'model': 'deepseek-chat',
        'enabled': False,  # Cambiar a True con API key válida
        'max_retries': 3,
        'timeout': 10
    }
    
    # Crear handler
    handler = DeepSeekHandler(config)
    
    if handler.is_available:
        print("DeepSeek disponible")
        
        # Pruebas de corrección
        test_texts = [
            "hello how are you today i am fine",
            "i go to store yesterday and buy some food",
            "the weather is very good today isnt it",
            "can you help me with this problem please",
            "thank you very much for you help"
        ]
        
        for text in test_texts:
            print(f"\nTexto original: '{text}'")
            
            result = handler.correct_text_sync(text)
            
            if result['success']:
                print(f"Texto corregido: '{result['corrected_text']}'")
                print(f"Tiempo: {result['processing_time']:.2f}s")
            else:
                print(f"Error: {result['error']}")
        
        # Mostrar estadísticas
        stats = handler.get_stats()
        print(f"\nEstadísticas: {stats}")
    
    else:
        print("DeepSeek no disponible (verificar API key y configuración)")
    
    # Limpiar
    handler.cleanup()
    print("Prueba completada")