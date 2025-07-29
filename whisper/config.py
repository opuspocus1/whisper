# -*- coding: utf-8 -*-
"""
Configuración principal del Traductor de Voz en Tiempo Real
Español → Inglés usando OpenAI Whisper + ElevenLabs
"""

import os
from pathlib import Path
from typing import Dict, Any
import json

class Config:
    """Clase de configuración principal"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / "settings.json"
        self.load_config()
    
    def load_config(self):
        """Cargar configuración desde archivo o crear valores por defecto"""
        default_config = {
            # APIs
# ElevenLabs removido como solicitado
            
            # Configuración de TTS (Text-to-Speech)
            
            # Whisper
            "whisper": {
                "engine": "whisper",  # "whisper" o "faster-whisper"
                "model_size": "base",  # tiny, base, small, medium, large
                "language": "spanish",
                "task": "translate",  # transcribe o translate - OPTIMIZADO: traducción directa
                "device": "cpu",  # cpu, cuda (para faster-whisper)
                "compute_type": "int8",  # int8, float16, float32 (para faster-whisper)
                "temperature": 0.0,          # Respuesta determinista
                "best_of": 1,               # Solo una mejor opción
                "beam_size": 1,             # Decodificación greedy (más rápida)
                "patience": 1.0,
                "length_penalty": 0.0,      # Sin penalización de longitud
                "suppress_tokens": "-1",
                "initial_prompt": "Concise translation:",  # Traducción breve
                "condition_on_previous_text": False, # No condicionar en texto previo
                "fp16": False,              # Desactivar precisión mixta en CPU
                "compression_ratio_threshold": 1.5,  # Umbral de compresión
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.3,  # Ignorar silencios
                "enable_timestamps": True,  # Para faster-whisper
                "word_timestamps": False  # Timestamps a nivel de palabra
            },
            
            # Audio
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024,
                "format": "int16",
                "input_device": None,  # None = dispositivo por defecto
                "output_device": None,  # None = dispositivo por defecto
                "noise_reduction": True,
                "auto_gain_control": True,
                "echo_cancellation": True,
                "volume_input": 0.8,
                "volume_output": 0.8
            },
            
            # Traducción
            "translation": {
                "source_language": "es",  # español
                "target_language": "en",  # inglés
                "use_context": True,
                "preserve_formatting": True,
                "fallback_translator": "google"  # google, deepl, libre
            },
            
            # Qwen (Corrección de texto con IA)
            "deepseek": {
                "api_key": "sk-or-v1-your-openrouter-api-key-here",  # Obtén tu API key GRATUITA en https://openrouter.ai/
                "model": "qwen/qwen3-235b-a22b-2507:free",  # Modelo GRATUITO de Qwen 3
                "fallback_model": "qwen/qwen3-235b-a22b-2507",  # Modelo de pago como fallback
                "base_url": "https://openrouter.ai/api/v1",
                "use_openrouter": True,  # Usar OpenRouter
                "enabled": True,  # Habilitado por defecto
                "max_retries": 3,
                "timeout": 30,
                "auto_correct": True,  # Corrección automática
                "improve_fluency": True,  # Mejorar fluidez
                "preserve_meaning": True  # Preservar significado original
            },
            
            # Interfaz
            "ui": {
                "theme": "dark",  # dark, light, system
                "language": "es",  # es, en
                "window_size": [800, 600],
                "window_position": [100, 100],
                "always_on_top": False,
                "minimize_to_tray": True,
                "show_waveform": True,
                "show_confidence": True,
                "font_size": 12,
                "font_family": "Segoe UI"
            },
            
            # Rendimiento
            "performance": {
                "max_audio_length": 30,  # segundos
                "processing_timeout": 10,  # segundos
                "max_concurrent_requests": 3,
                "cache_enabled": True,
                "cache_size": 100,  # MB
                "gpu_acceleration": True,
                "cpu_threads": 4
            },
            
            # Logging
            "logging": {
                "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
                "file_enabled": True,
                "console_enabled": True,
                "max_file_size": 10,  # MB
                "backup_count": 5
            },
            
            # Características avanzadas
            "advanced": {
                "real_time_processing": True,
                "voice_activity_detection": True,
                "silence_detection_threshold": 0.01,
                "min_speech_duration": 0.5,  # segundos
                "max_silence_duration": 2.0,  # segundos
                "auto_punctuation": True,
                "speaker_diarization": False,
                "emotion_detection": False
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # Merge con configuración por defecto
                self.config = self._merge_configs(default_config, loaded_config)
            except Exception as e:
                print(f"Error cargando configuración: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Combinar configuración por defecto con la cargada"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self):
        """Guardar configuración actual"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando configuración: {e}")
    
    def get(self, section: str, key: str = None, default=None):
        """Obtener valor de configuración"""
        if key is None:
            return self.config.get(section, default)
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value):
        """Establecer valor de configuración"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
    
    
    def get_whisper_config(self) -> Dict[str, Any]:
        """Obtener configuración de Whisper"""
        return self.config["whisper"]
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Obtener configuración de audio"""
        return self.config["audio"]
    
    def get_translation_config(self) -> Dict[str, Any]:
        """Obtener configuración de traducción"""
        return self.config["translation"]
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Obtener configuración de interfaz de usuario"""
        return self.config["ui"]
    
    def get_deepseek_config(self) -> Dict[str, Any]:
        """Obtener configuración de Qwen (manteniendo nombre deepseek para compatibilidad)"""
        default_deepseek = {
            "api_key_free": "",  # API key para modelos gratuitos
            "api_key_paid": "",  # API key para modelos de pago
            "model": "qwen/qwen3-235b-a22b-2507:free",
            "fallback_model": "qwen/qwen3-235b-a22b-2507",
            "base_url": "https://openrouter.ai/api/v1",
            "use_openrouter": True,
            "enabled": True,
            "max_retries": 3,
            "timeout": 30,
            "auto_correct": True,
            "improve_fluency": True,
            "preserve_meaning": True,
            "use_dual_keys": False  # Habilitar sistema de doble API key
        }
        return self.config.get("deepseek", default_deepseek)
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Obtener configuración de TTS (Text-to-Speech)"""
        default_tts = {
            "engine": "kokoro",
            "fallback_engine": "kokoro",
            "auto_fallback": True,
            "kokoro": {
                "voice": "af_bella",
                "speed": 1.0,
                "server_host": "localhost",
                "server_port": 8880,
                "timeout": 30
            },
            "pyttsx3": {
                "rate": 150,
                "volume": 0.9,
                "voice_index": 0,
                "language": "en"
            }
        }
        return self.config.get("tts", default_tts)
    
    def get_elevenlabs_config(self) -> Dict[str, Any]:
        """Obtener configuración de ElevenLabs (para compatibilidad)"""
        default_elevenlabs = {
            "api_key": "",
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "speech_rate": 1.0,
            "model": "eleven_monolingual_v1",
            "stability": 0.5,
            "similarity_boost": 0.5
        }
        return self.config.get("elevenlabs", default_elevenlabs)
    
    def is_deepseek_configured(self) -> bool:
        """Verificar si DeepSeek está configurado"""
        deepseek_config = self.get_deepseek_config()
        
        if not deepseek_config.get("enabled", True):
            return False
            
        # Si usa sistema de doble API key
        if deepseek_config.get("use_dual_keys", False):
            free_key = deepseek_config.get("api_key_free", "").strip()
            paid_key = deepseek_config.get("api_key_paid", "").strip()
            # Al menos una de las dos keys debe estar configurada
            return len(free_key) > 0 or len(paid_key) > 0
        
        # Compatibilidad con configuración antigua
        if deepseek_config.get("use_openrouter", True):
            # Verificar si tiene alguna API key configurada
            free_key = deepseek_config.get("api_key_free", "").strip()
            old_key = deepseek_config.get("api_key", "").strip()  # Compatibilidad
            return len(free_key) > 0 or len(old_key) > 0
        else:
            # Con API directa, necesitamos API key
            free_key = deepseek_config.get("api_key_free", "").strip()
            old_key = deepseek_config.get("api_key", "").strip()  # Compatibilidad
            return len(free_key) > 0 or len(old_key) > 0
    
    def setup_deepseek_key(self, api_key: str):
        """Configurar la API key de OpenRouter para Qwen (compatibilidad)
        
        Para obtener tu API key GRATUITA:
        1. Ve a https://openrouter.ai/
        2. Regístrate gratis
        3. Ve a 'Keys' y crea una nueva key
        4. Copia la key que empieza con 'sk-or-v1-'
        """
        self.set("deepseek", "api_key_free", api_key.strip())
    
    def setup_deepseek_free_key(self, api_key: str):
        """Configurar la API key gratuita de OpenRouter para Qwen
        
        Para obtener tu API key GRATUITA:
        1. Ve a https://openrouter.ai/
        2. Regístrate gratis
        3. Ve a 'Keys' y crea una nueva key
        4. Copia la key que empieza con 'sk-or-v1-'
        """
        self.set("deepseek", "api_key_free", api_key.strip())
    
    def setup_deepseek_paid_key(self, api_key: str):
        """Configurar la API key de pago de OpenRouter para Qwen
        
        Para obtener tu API key DE PAGO:
        1. Ve a https://openrouter.ai/
        2. Agrega créditos a tu cuenta
        3. Ve a 'Keys' y crea una nueva key
        4. Copia la key que empieza con 'sk-or-v1-'
        """
        self.set("deepseek", "api_key_paid", api_key.strip())
    
    def enable_dual_keys(self, enabled: bool = True):
        """Habilitar/deshabilitar sistema de doble API key"""
        self.set("deepseek", "use_dual_keys", enabled)
    
    def enable_deepseek(self, enabled: bool = True):
        """Habilitar/deshabilitar Qwen (manteniendo nombre deepseek para compatibilidad)"""
        self.set("deepseek", "enabled", enabled)
    
    def get_kokoro_config(self) -> dict:
        """Obtener configuración de Kokoro Voice API"""
        kokoro_section = self.config.get('kokoro', {})
        return {
            'voice': kokoro_section.get('voice', 'af_bella'),
            'speed': float(kokoro_section.get('speed', 1.0)),
            'server_host': kokoro_section.get('server_host', 'localhost'),
            'server_port': int(kokoro_section.get('server_port', 5000)),
            'timeout': float(kokoro_section.get('timeout', 30.0))
        }
    

# Instancia global de configuración
config = Config()

# Variables de entorno
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', '')
LOG_LEVEL = os.getenv('LOG_LEVEL', config.get('logging', 'level'))

# Constantes
APP_NAME = "Traductor de Voz en Tiempo Real"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Tu Nombre"
APP_DESCRIPTION = "Traductor de voz español-inglés usando OpenAI Whisper + ElevenLabs"

# URLs y endpoints

# Límites y timeouts
MAX_AUDIO_DURATION = 30  # segundos
MAX_TEXT_LENGTH = 5000   # caracteres
API_TIMEOUT = 30         # segundos
RECORDING_TIMEOUT = 60   # segundos

if __name__ == "__main__":
    # Prueba de configuración
    print(f"Configuración cargada para {APP_NAME} v{APP_VERSION}")
    print(f"Modelo Whisper: {config.get('whisper', 'model_size')}")
    print(f"Tema UI: {config.get('ui', 'theme')}")