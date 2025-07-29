# -*- coding: utf-8 -*-
"""
Manejador de OpenAI Whisper para reconocimiento de voz
Convierte audio en español a texto
"""

import whisper
import numpy as np
import torch
import threading
import queue
import time
import tempfile
import os
from typing import Optional, Dict, Any, Callable
import logging
from pathlib import Path
import soundfile as sf

from config import config

class WhisperHandler:
    """Manejador de OpenAI Whisper para reconocimiento de voz"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.whisper_config = config.get_whisper_config()
        
        # Modelo Whisper
        self.model = None
        self.model_size = self.whisper_config['model_size']
        self.device = "cuda" if torch.cuda.is_available() and config.get('performance', 'gpu_acceleration') else "cpu"
        self.engine_type = "whisper"  # Identificador del motor
        
        # Estados
        self.is_loaded = False
        self.is_processing = False
        
        # Cola de procesamiento
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        
        # Callbacks
        self.on_transcription: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_progress: Optional[Callable] = None
        
        # Estadísticas
        self.stats = {
            'total_processed': 0,
            'total_duration': 0.0,
            'average_processing_time': 0.0,
            'errors': 0,
            'last_confidence': 0.0
        }
        
        # Cache de resultados
        self.result_cache = {}
        self.cache_enabled = config.get('performance', 'cache_enabled', True)
        
        self.logger.info(f"WhisperHandler inicializado - Dispositivo: {self.device}")
    
    def load_model(self, model_size: str = None) -> bool:
        """Cargar modelo Whisper"""
        if model_size:
            self.model_size = model_size
        
        try:
            self.logger.info(f"Cargando modelo Whisper '{self.model_size}' en {self.device}...")
            
            # Cargar modelo
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=config.get('whisper', 'model_path', None)
            )
            
            self.is_loaded = True
            self.logger.info(f"Modelo Whisper '{self.model_size}' cargado exitosamente")
            
            # Iniciar thread de procesamiento
            self._start_processing_thread()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo Whisper: {e}")
            self.is_loaded = False
            if self.on_error:
                self.on_error(f"Error cargando modelo: {e}")
            return False
    
    def _start_processing_thread(self):
        """Iniciar thread de procesamiento"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        self.logger.info("Thread de procesamiento iniciado")
    
    def _processing_worker(self):
        """Worker para procesamiento de audio"""
        self.logger.info("Worker de procesamiento iniciado")
        while self.is_processing:
            try:
                # Obtener tarea de la cola
                self.logger.debug("Esperando tarea en la cola...")
                task = self.processing_queue.get(timeout=1.0)
                
                if task is None:  # Señal de parada
                    self.logger.info("Señal de parada recibida")
                    break
                
                audio_data, sample_rate, options, callback = task
                self.logger.info(f"Procesando audio: {len(audio_data)} muestras a {sample_rate} Hz")
                
                # Procesar audio
                result = self._transcribe_audio(audio_data, sample_rate, options)
                self.logger.info(f"Transcripción completada: '{result.get('text', '')[:50]}...'")
                
                # Llamar callback con resultado
                if callback:
                    self.logger.info("Llamando callback específico")
                    callback(result)
                elif self.on_transcription:
                    self.logger.info("Llamando callback on_transcription")
                    self.on_transcription(result)
                else:
                    self.logger.warning("No hay callback configurado para el resultado")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker de procesamiento: {e}")
                if self.on_error:
                    self.on_error(f"Error de procesamiento: {e}")
        
        self.logger.info("Worker de procesamiento terminado")
    
    def transcribe_async(self, 
                        audio_data: np.ndarray, 
                        sample_rate: int = 16000,
                        callback: Optional[Callable] = None,
                        **options) -> bool:
        """Transcribir audio de forma asíncrona"""
        if not self.is_loaded:
            self.logger.error("Modelo no cargado")
            return False
        
        try:
            # Agregar a la cola de procesamiento
            task = (audio_data, sample_rate, options, callback)
            self.processing_queue.put_nowait(task)
            return True
            
        except queue.Full:
            self.logger.warning("Cola de procesamiento llena")
            return False
        except Exception as e:
            self.logger.error(f"Error agregando tarea: {e}")
            return False
    
    def transcribe_sync(self, 
                       audio_data: np.ndarray, 
                       sample_rate: int = 16000,
                       **options) -> Dict[str, Any]:
        """Transcribir audio de forma síncrona"""
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado")
        
        return self._transcribe_audio(audio_data, sample_rate, options)
    
    def _transcribe_audio(self, 
                         audio_data: np.ndarray, 
                         sample_rate: int,
                         options: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribir audio usando Whisper"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Iniciando transcripción de audio: {len(audio_data)} muestras, {sample_rate} Hz")
            
            # Preparar audio
            audio_prepared = self._prepare_audio(audio_data, sample_rate)
            self.logger.info(f"Audio preparado: {len(audio_prepared)} muestras, dtype: {audio_prepared.dtype}")
            
            # Verificar cache
            cache_key = None
            if self.cache_enabled:
                cache_key = self._get_cache_key(audio_prepared, options)
                if cache_key in self.result_cache:
                    self.logger.debug("Resultado obtenido del cache")
                    return self.result_cache[cache_key]
            
            # Configurar opciones de transcripción
            transcribe_options = self._prepare_transcribe_options(options)
            self.logger.info(f"Opciones de transcripción: {transcribe_options}")
            
            # Notificar progreso
            if self.on_progress:
                self.on_progress("Transcribiendo...")
            
            # Transcribir
            self.logger.info("Llamando a model.transcribe()...")
            result = self.model.transcribe(
                audio_prepared,
                **transcribe_options
            )
            self.logger.info(f"Whisper devolvió resultado: {result.get('text', '')[:100]}...")
            
            # Procesar resultado
            processed_result = self._process_result(result, audio_data, sample_rate)
            
            # Guardar en cache
            if self.cache_enabled and cache_key:
                self.result_cache[cache_key] = processed_result
                self._cleanup_cache()
            
            # Actualizar estadísticas
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(audio_data) / sample_rate)
            
            self.logger.info(f"Transcripción completada en {processing_time:.2f}s: '{processed_result.get('text', '')}'")
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Error en transcripción: {e}")
            self.stats['errors'] += 1
            
            if self.on_error:
                self.on_error(f"Error de transcripción: {e}")
            
            return {
                'text': '',
                'language': 'es',
                'confidence': 0.0,
                'segments': [],
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _prepare_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preparar audio para Whisper"""
        # Convertir a float32 y normalizar
        if audio_data.dtype != np.float32:
            audio_float = audio_data.astype(np.float32)
            if audio_data.dtype == np.int16:
                audio_float /= 32768.0
            elif audio_data.dtype == np.int32:
                audio_float /= 2147483648.0
        else:
            audio_float = audio_data.copy()
        
        # Asegurar que esté en el rango [-1, 1]
        audio_float = np.clip(audio_float, -1.0, 1.0)
        
        # Remuestrear a 16kHz si es necesario
        if sample_rate != 16000:
            audio_float = self._resample_audio(audio_float, sample_rate, 16000)
        
        # Asegurar que sea mono
        if len(audio_float.shape) > 1:
            audio_float = np.mean(audio_float, axis=1)
        
        return audio_float
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Remuestrear audio"""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Remuestreo simple si librosa no está disponible
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    def _prepare_transcribe_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Preparar opciones de transcripción"""
        # Opciones por defecto del config
        transcribe_options = {
            'language': options.get('language', self.whisper_config.get('language', 'spanish')),
            'task': options.get('task', self.whisper_config.get('task', 'transcribe')),
            'temperature': options.get('temperature', self.whisper_config.get('temperature', 0.0)),
            'best_of': options.get('best_of', self.whisper_config.get('best_of', 5)),
            'beam_size': options.get('beam_size', self.whisper_config.get('beam_size', 5)),
            'patience': options.get('patience', self.whisper_config.get('patience', 1.0)),
            'length_penalty': options.get('length_penalty', self.whisper_config.get('length_penalty', 0.05)),
            'suppress_tokens': options.get('suppress_tokens', self.whisper_config.get('suppress_tokens', '-1')),
            'initial_prompt': options.get('initial_prompt', self.whisper_config.get('initial_prompt')),
            'condition_on_previous_text': options.get('condition_on_previous_text', 
                                                    self.whisper_config.get('condition_on_previous_text', True)),
            'fp16': options.get('fp16', self.whisper_config.get('fp16', True) and self.device == 'cuda'),
            'compression_ratio_threshold': options.get('compression_ratio_threshold', 
                                                     self.whisper_config.get('compression_ratio_threshold', 2.4)),
            'logprob_threshold': options.get('logprob_threshold', 
                                           self.whisper_config.get('logprob_threshold', -1.0)),
            'no_speech_threshold': options.get('no_speech_threshold', 
                                             self.whisper_config.get('no_speech_threshold', 0.6))
        }
        
        # Filtrar valores None
        return {k: v for k, v in transcribe_options.items() if v is not None}
    
    def _process_result(self, result: Dict[str, Any], audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Procesar resultado de Whisper"""
        # Calcular confianza promedio
        confidence = 0.0
        if 'segments' in result and result['segments']:
            confidences = []
            for segment in result['segments']:
                if 'avg_logprob' in segment:
                    # Convertir log probability a confianza
                    conf = np.exp(segment['avg_logprob'])
                    confidences.append(conf)
            
            if confidences:
                confidence = np.mean(confidences)
        
        # Detectar idioma si no se especificó
        detected_language = result.get('language', 'es')
        
        # Preparar resultado procesado
        processed_result = {
            'text': result.get('text', '').strip(),
            'language': detected_language,
            'confidence': float(confidence),
            'segments': result.get('segments', []),
            'audio_duration': len(audio_data) / sample_rate,
            'processing_time': 0.0,  # Se actualizará en el caller
            'model_size': self.model_size,
            'device': self.device
        }
        
        # Actualizar estadísticas
        self.stats['last_confidence'] = confidence
        
        return processed_result
    
    def _get_cache_key(self, audio_data: np.ndarray, options: Dict[str, Any]) -> str:
        """Generar clave de cache para audio y opciones"""
        import hashlib
        
        # Hash del audio
        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()[:16]
        
        # Hash de las opciones
        options_str = str(sorted(options.items()))
        options_hash = hashlib.md5(options_str.encode()).hexdigest()[:8]
        
        return f"{audio_hash}_{options_hash}_{self.model_size}"
    
    def _cleanup_cache(self):
        """Limpiar cache si es muy grande"""
        max_cache_size = 100  # máximo número de entradas
        
        if len(self.result_cache) > max_cache_size:
            # Remover las entradas más antiguas
            items_to_remove = len(self.result_cache) - max_cache_size + 10
            keys_to_remove = list(self.result_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.result_cache[key]
            
            self.logger.debug(f"Cache limpiado: {items_to_remove} entradas removidas")
    
    def _update_stats(self, processing_time: float, audio_duration: float):
        """Actualizar estadísticas"""
        self.stats['total_processed'] += 1
        self.stats['total_duration'] += audio_duration
        
        # Calcular tiempo promedio de procesamiento
        total_time = (self.stats['average_processing_time'] * (self.stats['total_processed'] - 1) + 
                     processing_time) / self.stats['total_processed']
        self.stats['average_processing_time'] = total_time
    
    def get_available_models(self) -> list:
        """Obtener lista de modelos disponibles"""
        return ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
    
    def get_model_info(self, model_size: str = None) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if model_size is None:
            model_size = self.model_size
        
        model_info = {
            'tiny': {'params': '39M', 'vram': '~1GB', 'speed': '~10x', 'multilingual': True},
            'base': {'params': '74M', 'vram': '~1GB', 'speed': '~7x', 'multilingual': True},
            'small': {'params': '244M', 'vram': '~2GB', 'speed': '~4x', 'multilingual': True},
            'medium': {'params': '769M', 'vram': '~5GB', 'speed': '~2x', 'multilingual': True},
            'large': {'params': '1550M', 'vram': '~10GB', 'speed': '1x', 'multilingual': True},
            'turbo': {'params': '809M', 'vram': '~6GB', 'speed': '~8x', 'multilingual': True}
        }
        
        return model_info.get(model_size, {})
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de uso"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Limpiar cache de resultados"""
        self.result_cache.clear()
        self.logger.info("Cache limpiado")
    
    def set_callbacks(self, 
                     on_transcription: Optional[Callable] = None,
                     on_error: Optional[Callable] = None,
                     on_progress: Optional[Callable] = None):
        """Establecer callbacks"""
        if on_transcription:
            self.on_transcription = on_transcription
        if on_error:
            self.on_error = on_error
        if on_progress:
            self.on_progress = on_progress
    
    def stop_processing(self):
        """Detener procesamiento"""
        self.is_processing = False
        
        # Enviar señal de parada
        try:
            self.processing_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Esperar a que termine el thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        self.logger.info("Procesamiento detenido")
    
    def cleanup(self):
        """Limpiar recursos"""
        self.logger.info("Limpiando WhisperHandler...")
        
        self.stop_processing()
        self.clear_cache()
        
        # Liberar modelo
        if self.model:
            del self.model
            self.model = None
        
        # Limpiar memoria GPU si está disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        self.logger.info("WhisperHandler limpiado")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

if __name__ == "__main__":
    # Prueba del WhisperHandler
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    # Crear handler
    whisper_handler = WhisperHandler()
    
    # Cargar modelo
    if whisper_handler.load_model('base'):
        print("Modelo cargado exitosamente")
        
        # Mostrar información del modelo
        model_info = whisper_handler.get_model_info()
        print(f"Información del modelo: {model_info}")
        
        # Crear audio de prueba (silencio)
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 segundo de silencio
        
        # Transcribir
        result = whisper_handler.transcribe_sync(test_audio, 16000)
        print(f"Resultado: {result}")
        
        # Mostrar estadísticas
        stats = whisper_handler.get_stats()
        print(f"Estadísticas: {stats}")
    
    else:
        print("Error cargando modelo")
    
    # Limpiar
    whisper_handler.cleanup()
    print("Prueba completada")