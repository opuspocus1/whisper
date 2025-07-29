# -*- coding: utf-8 -*-
"""
Manejador unificado de TTS (Text-to-Speech)
Permite cambiar entre ElevenLabs y pyttsx3
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

# Importar handlers de TTS
# ElevenLabs removido como solicitado
from kokoro_handler import KokoroHandler

# PYTTSX3 ELIMINADO - Causaba problemas de timeout en Windows SAPI5

from config import config

class TTSManager:
    """Manejador unificado de TTS con soporte para múltiples motores"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        self.tts_config = config.get_tts_config()
        
        # Handlers disponibles
        self.handlers = {}
        self.available_engines = []
        
        # Inicializar handlers
        self._initialize_handlers()
        
        # Motor actual y de respaldo
        self.current_engine = self.tts_config.get('engine', 'kokoro')
        self.fallback_engine = self.tts_config.get('fallback_engine', 'kokoro')
        self.auto_fallback = self.tts_config.get('auto_fallback', True)
        
        # Verificar que el motor principal esté disponible
        if self.current_engine not in self.available_engines:
            self.logger.warning(f"Motor principal '{self.current_engine}' no disponible")
            if self.fallback_engine in self.available_engines:
                self.logger.info(f"Cambiando a motor de respaldo: {self.fallback_engine}")
                self.current_engine = self.fallback_engine
            elif self.available_engines:
                self.current_engine = self.available_engines[0]
                self.logger.info(f"Usando primer motor disponible: {self.current_engine}")
            else:
                self.logger.error("No hay motores TTS disponibles")
                self.current_engine = None
        
        
        
        # Callbacks
        self.on_audio_ready: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_progress: Optional[Callable] = None
        
        # Estadísticas globales
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'engine_switches': 0,
            'fallback_uses': 0
        }
        
        self.logger.info(f"TTSManager inicializado - Motor: {self.current_engine}, Disponibles: {self.available_engines}")
    
    def set_callbacks(self, on_audio_ready: Optional[Callable] = None, 
                     on_error: Optional[Callable] = None, 
                     on_progress: Optional[Callable] = None):
        """Establecer callbacks para eventos"""
        if on_audio_ready:
            self.on_audio_ready = on_audio_ready
        if on_error:
            self.on_error = on_error
        if on_progress:
            self.on_progress = on_progress
    
    def _initialize_handlers(self):
        """Inicializar todos los handlers disponibles"""
        
        # ElevenLabs removido como solicitado
        
        # Inicializar Kokoro Voice API
        try:
            kokoro_handler = KokoroHandler()
            self.handlers['kokoro'] = kokoro_handler
            self.available_engines.append('kokoro')
            self.logger.info("Kokoro Voice API inicializado exitosamente")
        except Exception as e:
            self.logger.warning(f"No se pudo inicializar Kokoro Voice API: {e}")
        
        # PYTTSX3 ELIMINADO - Causaba problemas de timeout en Windows SAPI5
    
    def get_available_engines(self) -> List[str]:
        """Obtener lista de motores disponibles"""
        return self.available_engines.copy()
    
    def get_current_engine(self) -> str:
        """Obtener motor actual"""
        return self.current_engine
    
    def set_engine(self, engine_name: str) -> bool:
        """Cambiar motor TTS"""
        if engine_name not in self.available_engines:
            self.logger.error(f"Motor '{engine_name}' no disponible")
            return False
        
        if engine_name != self.current_engine:
            old_engine = self.current_engine
            self.current_engine = engine_name
            self.stats['engine_switches'] += 1
            self.logger.info(f"Motor cambiado de '{old_engine}' a '{engine_name}'")
            
            # Actualizar configuración
            config.set('tts', 'engine', engine_name)
        
        return True
    
    def get_engine_info(self, engine_name: str = None) -> Dict[str, Any]:
        """Obtener información de un motor específico"""
        engine_name = engine_name or self.current_engine
        
        if engine_name not in self.handlers:
            return {'error': f'Motor {engine_name} no disponible'}
        
        handler = self.handlers[engine_name]
        
        info = {
            'engine': engine_name,
            'available': True,
            'is_current': engine_name == self.current_engine
        }
        
        # Información específica por motor
        if engine_name == 'elevenlabs':
            info.update({
                'type': 'cloud',
                'cost': 'freemium',
                'quality': 'high',
                'languages': 'multilingual',
                'features': ['voice_cloning', 'emotion', 'speed_control']
            })
            if hasattr(handler, 'get_voice_info'):
                info['voice_info'] = handler.get_voice_info()
        elif engine_name == 'kokoro':
            info.update({
                'type': 'local',
                'cost': 'free',
                'quality': 'high',
                'languages': 'multilingual',
                'features': ['multiple_voices', 'local_processing']
            })
            if hasattr(handler, 'get_available_voices'):
                info['available_voices'] = handler.get_available_voices()
        
        # pyttsx3 eliminado por problemas de timeout
        
        
        
        return info
    
    def synthesize_sync(self, text: str, **options) -> Dict[str, Any]:
        """Sintetizar texto de forma síncrona"""
        if not self.current_engine or self.current_engine not in self.handlers:
            return {
                'success': False,
                'error': 'No hay motor TTS disponible',
                'audio_data': None,
                'engine': None
            }
        
        self.stats['total_requests'] += 1
        
        # Intentar con motor principal
        result = self._try_synthesis(self.current_engine, text, options)
        
        # Si falla y auto_fallback está habilitado, intentar con motor de respaldo
        if not result['success'] and self.auto_fallback and self.fallback_engine != self.current_engine:
            if self.fallback_engine in self.available_engines:
                self.logger.warning(f"Síntesis falló con {self.current_engine}, intentando con {self.fallback_engine}")
                result = self._try_synthesis(self.fallback_engine, text, options)
                if result['success']:
                    self.stats['fallback_uses'] += 1
                    result['used_fallback'] = True
        
        # Actualizar estadísticas
        if result['success']:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        return result
    
    def _try_synthesis(self, engine_name: str, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Intentar síntesis con un motor específico"""
        if engine_name not in self.handlers:
            return {
                'success': False,
                'error': f'Motor {engine_name} no disponible',
                'audio_data': None,
                'engine': engine_name
            }
        
        handler = self.handlers[engine_name]
        
        try:
            # Preparar opciones específicas del motor
            engine_options = self._prepare_engine_options(engine_name, options, text)
            
            # Notificar progreso
            if self.on_progress:
                self.on_progress(f"Sintetizando con {engine_name}...")
            
            # Realizar síntesis
            result = handler.synthesize_sync(text, **engine_options)
            
            # Agregar información del motor al resultado
            if result.get('success', False):
                result['engine'] = engine_name
                result['engine_type'] = self.get_engine_info(engine_name).get('type', 'unknown')
            
            return result
            
        except Exception as e:
            error_msg = f"Error en síntesis con {engine_name}: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'audio_data': None,
                'engine': engine_name
            }
    
    def _prepare_engine_options(self, engine_name: str, options: Dict[str, Any], text: str = '') -> Dict[str, Any]:
        """Preparar opciones específicas para cada motor"""
        engine_options = options.copy()
        
        # Obtener configuración específica del motor
        engine_config = self.tts_config.get(engine_name, {})
        
        # Combinar configuración por defecto con opciones proporcionadas
        for key, value in engine_config.items():
            if key not in engine_options:
                engine_options[key] = value
        
        # Agregar optimizaciones específicas para Kokoro
        if engine_name == 'kokoro':
            # Habilitar GPU si está disponible en la configuración
            from config import config
            performance_config = config.get('performance') or {}
            gpu_enabled = performance_config.get('gpu_acceleration', True)
            engine_options.setdefault('use_gpu', gpu_enabled)
            
            # Habilitar streaming para textos largos (más de 100 caracteres)
            text_length = len(text)
            engine_options.setdefault('use_streaming', text_length > 100)
            
            self.logger.info(f"🚀 Kokoro optimizado: GPU={engine_options['use_gpu']}, Streaming={engine_options['use_streaming']}")
            print(f"🚀 KOKORO OPTIMIZADO: GPU={engine_options['use_gpu']}, Streaming={engine_options['use_streaming']}")
        
        return engine_options
    
    def synthesize_async(self, text: str, **options):
        """Sintetizar texto de forma asíncrona"""
        if not self.current_engine or self.current_engine not in self.handlers:
            if self.on_error:
                self.on_error("No hay motor TTS disponible")
            return
        
        # Configurar callbacks para el handler
        handler = self.handlers[self.current_engine]
        
        # Guardar callbacks originales
        original_on_audio_ready = handler.on_audio_ready
        original_on_error = handler.on_error
        
        # Configurar callbacks temporales
        def on_audio_ready_wrapper(result):
            try:
                result['engine'] = self.current_engine
                result['engine_type'] = self.get_engine_info(self.current_engine).get('type', 'unknown')
                if self.on_audio_ready:
                    self.on_audio_ready(result)
            finally:
                # Restaurar callbacks originales
                handler.on_audio_ready = original_on_audio_ready
                handler.on_error = original_on_error
        
        def on_error_wrapper(error):
            try:
                # Intentar fallback si está habilitado
                if self.auto_fallback and self.fallback_engine != self.current_engine:
                    if self.fallback_engine in self.available_engines:
                        self.logger.warning(f"Síntesis asíncrona falló con {self.current_engine}, intentando con {self.fallback_engine}")
                        fallback_handler = self.handlers[self.fallback_engine]
                        fallback_handler.on_audio_ready = on_audio_ready_wrapper
                        fallback_handler.on_error = self.on_error
                        fallback_options = self._prepare_engine_options(self.fallback_engine, options, text)
                        # Filtrar opciones para kokoro_handler si es el fallback
                        if self.fallback_engine == 'kokoro':
                            # Solo pasar voice si existe en las opciones
                            voice = fallback_options.get('voice')
                            if voice:
                                fallback_handler.synthesize_async(text, voice=voice)
                            else:
                                fallback_handler.synthesize_async(text)
                        else:
                            fallback_handler.synthesize_async(text, **fallback_options)
                        self.stats['fallback_uses'] += 1
                        return
                
                if self.on_error:
                    self.on_error(error)
            finally:
                # Restaurar callbacks originales
                handler.on_audio_ready = original_on_audio_ready
                handler.on_error = original_on_error
        
        handler.on_audio_ready = on_audio_ready_wrapper
        handler.on_error = on_error_wrapper
        
        # Preparar opciones y realizar síntesis
        engine_options = self._prepare_engine_options(self.current_engine, options, text)
        
        # Filtrar opciones para kokoro_handler (solo acepta voice)
        if self.current_engine == 'kokoro':
            # Solo pasar voice si existe en las opciones
            voice = engine_options.get('voice')
            if voice:
                handler.synthesize_async(text, voice=voice)
            else:
                handler.synthesize_async(text)
        else:
            handler.synthesize_async(text, **engine_options)
        
        # NO restaurar callbacks aquí - se restaurarán cuando se complete la síntesis
        # Los callbacks se restauran en los wrappers después de procesar el resultado
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas globales"""
        stats = self.stats.copy()
        
        # Agregar estadísticas de cada motor
        stats['engines'] = {}
        for engine_name, handler in self.handlers.items():
            if hasattr(handler, 'get_stats'):
                stats['engines'][engine_name] = handler.get_stats()
        
        # Calcular tasas
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['fallback_rate'] = stats['fallback_uses'] / stats['total_requests']
        else:
            stats['success_rate'] = 0
            stats['fallback_rate'] = 0
        
        stats['current_engine'] = self.current_engine
        stats['available_engines'] = self.available_engines
        
        return stats
    
    def reload_config(self):
        """Recargar configuración desde config.py"""
        try:
            from config import config
            tts_config = config.get_tts_config()
            
            # Actualizar configuración
            new_primary = tts_config.get('engine', 'elevenlabs')
            new_fallback = tts_config.get('fallback_engine', 'elevenlabs')
            new_auto_fallback = tts_config.get('auto_fallback', True)
            
            # Solo cambiar si es diferente
            if (new_primary != self.current_engine or 
                new_fallback != self.fallback_engine or 
                new_auto_fallback != self.auto_fallback):
                
                self.logger.info(f"Recargando configuración TTS: {self.current_engine} -> {new_primary}")
                
                self.current_engine = new_primary
                self.fallback_engine = new_fallback
                self.auto_fallback = new_auto_fallback
                
                # Reconfigurar handlers individuales si es necesario
                for engine_name, handler in self.handlers.items():
                    if hasattr(handler, 'reload_config'):
                        handler.reload_config()
                
                self.logger.info("Configuración TTS recargada exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error recargando configuración TTS: {e}")
    
    def set_speech_rate(self, rate: float):
        """Establecer velocidad de habla (compatibilidad con ElevenLabs)"""
        if self.current_engine in self.handlers:
            handler = self.handlers[self.current_engine]
            if hasattr(handler, 'set_speech_rate'):
                handler.set_speech_rate(rate)
    
    def set_voice(self, voice_id: str):
        """Establecer voz (compatibilidad con ElevenLabs)"""
        self.logger.info(f"🎭 TTSManager.set_voice llamado con: {voice_id}")
        print(f"🎭 TTSManager.set_voice: {voice_id} para engine: {self.current_engine}")
        if self.current_engine in self.handlers:
            handler = self.handlers[self.current_engine]
            if hasattr(handler, 'set_voice'):
                self.logger.info(f"🎭 Llamando handler.set_voice({voice_id}) en {self.current_engine}")
                handler.set_voice(voice_id)
            else:
                self.logger.warning(f"🎭 Handler {self.current_engine} no tiene método set_voice")
        else:
            self.logger.error(f"🎭 Engine {self.current_engine} no encontrado en handlers")
    
    def cleanup(self):
        """Limpiar recursos de todos los handlers"""
        for engine_name, handler in self.handlers.items():
            try:
                if hasattr(handler, 'cleanup'):
                    handler.cleanup()
                self.logger.info(f"Handler {engine_name} limpiado")
            except Exception as e:
                self.logger.error(f"Error limpiando handler {engine_name}: {e}")
        
        self.handlers.clear()
        self.available_engines.clear()
        self.logger.info("TTSManager limpiado")


if __name__ == "__main__":
    # Prueba del TTSManager
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    # Crear manager
    tts_manager = TTSManager()
    
    print(f"Motores disponibles: {tts_manager.get_available_engines()}")
    print(f"Motor actual: {tts_manager.get_current_engine()}")
    
    # Obtener información de motores
    for engine in tts_manager.get_available_engines():
        info = tts_manager.get_engine_info(engine)
        print(f"\n{engine}: {info}")
    
    # Prueba de síntesis
    if tts_manager.get_available_engines():
        test_text = "Hello, this is a test of the unified TTS manager system."
        print(f"\nSintetizando: '{test_text}'")
        
        result = tts_manager.synthesize_sync(test_text)
        
        if result['success']:
            print(f"Síntesis exitosa con {result['engine']}")
            print(f"Tiempo de procesamiento: {result.get('processing_time', 0):.2f}s")
        else:
            print(f"Error en síntesis: {result['error']}")
        
        # Mostrar estadísticas
        stats = tts_manager.get_stats()
        print(f"\nEstadísticas: {stats}")
    
    else:
        print("No hay motores TTS disponibles")
    
    # Limpiar
    tts_manager.cleanup()
    print("Prueba completada")