import logging
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import torch
from faster_whisper import WhisperModel
from config import Config

class FasterWhisperHandler:
    """
    Handler para faster-whisper con la misma interfaz que WhisperHandler
    Proporciona traducción directa de español a inglés con mejor rendimiento
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuración
        self.whisper_config = config.get_whisper_config()
        self.model_size = self.whisper_config.get("model_size", "base")
        self.device = self.whisper_config.get("device", "cpu")
        self.compute_type = self.whisper_config.get("compute_type", "int8")
        self.language = "es"  # Forzar español
        self.task = self.whisper_config.get("task", "translate")  # Usar task de configuración
        self.engine_type = "faster-whisper"  # Identificador del motor
        
        # Optimizaciones específicas para modelo tiny
        self.is_tiny_model = self.model_size == "tiny"
        if self.is_tiny_model:
            self.logger.info("Modelo 'tiny' detectado - Aplicando optimizaciones de velocidad")
            # Usar int8 para máxima velocidad en modelo tiny
            if self.device == "cpu":
                self.compute_type = "int8"
            # Configurar para máxima velocidad
            self.optimize_for_speed = True
        else:
            self.optimize_for_speed = False
        
        # Modelo
        self.model = None
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        
        # Callbacks
        self.on_transcription_ready = None
        self.on_error = None
        self.on_progress = None
        
        # Estadísticas
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        # Control de procesamiento
        self.processing = False
        self.stop_requested = False
        
        # Cola de procesamiento y worker thread
        self.processing_queue = queue.Queue(maxsize=10)
        self.worker_thread = None
        self.worker_running = False
        
        self.logger.info(f"FasterWhisperHandler inicializado - Modelo: {self.model_size}, Device: {self.device}")
    
    def load_model(self, model_size: str = None) -> bool:
        """
        Cargar el modelo faster-whisper
        """
        # Actualizar tamaño del modelo si se proporciona
        if model_size:
            self.model_size = model_size
            self.logger.info(f"Cambiando modelo a: {self.model_size}")
        
        if self.model_loaded:
            # Si ya está cargado pero con un modelo diferente, recargar
            if hasattr(self, '_current_model_size') and self._current_model_size != self.model_size:
                self.logger.info(f"Recargando modelo: {self._current_model_size} → {self.model_size}")
                self.cleanup()
            else:
                return True
            
        with self.loading_lock:
            if self.model_loaded:
                return True
                
            try:
                self.logger.info(f"Cargando modelo faster-whisper: {self.model_size}")
                
                # Inicializar modelo faster-whisper
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=None,  # Usar directorio por defecto
                    local_files_only=False
                )
                
                self.model_loaded = True
                self._current_model_size = self.model_size
                self.logger.info(f"Modelo faster-whisper {self.model_size} cargado exitosamente")
                return True
                
            except Exception as e:
                self.logger.error(f"Error cargando modelo faster-whisper: {e}")
                self.model = None
                self.model_loaded = False
                return False
    
    def set_callbacks(self, on_transcription: Callable = None, 
                     on_error: Callable = None, on_progress: Callable = None):
        """
        Establecer callbacks para eventos
        """
        self.on_transcription_ready = on_transcription
        self.on_error = on_error
        self.on_progress = on_progress
    
    def transcribe_async(self, audio_data: np.ndarray, 
                        sample_rate: int = 16000,
                        callback: Optional[Callable] = None,
                        **options) -> bool:
        """
        Transcribir audio de forma asíncrona
        """
        if not self.model_loaded:
            self.logger.error("Modelo no cargado")
            return False
        
        try:
            # Iniciar worker thread si no está corriendo
            if not self.worker_running:
                self._start_worker()
            
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
    
    def _start_worker(self):
        """Iniciar worker thread para procesamiento"""
        if self.worker_running:
            return
            
        self.worker_running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="FasterWhisperWorker"
        )
        self.worker_thread.start()
        self.logger.info("Worker de procesamiento iniciado")
    
    def _worker_loop(self):
        """Loop principal del worker thread"""
        self.logger.info("Worker de procesamiento iniciado")
        
        while self.worker_running and not self.stop_requested:
            try:
                # Obtener tarea de la cola (timeout de 1 segundo)
                task = self.processing_queue.get(timeout=1.0)
                
                if task is None:  # Señal de parada
                    break
                
                audio_data, sample_rate, options, callback = task
                
                # Procesar audio
                result = self._transcribe_audio(audio_data, sample_rate, options)
                
                # Llamar callback si se proporcionó
                if callback:
                    try:
                        callback(result)
                    except Exception as e:
                        self.logger.error(f"Error en callback personalizado: {e}")
                
                # Llamar callback de transcripción lista
                if self.on_transcription_ready and result.get('text') and not result.get('error'):
                    try:
                        self.on_transcription_ready(result)
                    except Exception as e:
                        import traceback
                        self.logger.error(f"Error en callback de transcripción: {e}")
                        self.logger.error(f"Stack trace: {traceback.format_exc()}")
                
                # Marcar tarea como completada
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker loop: {e}")
                if self.on_error:
                    try:
                        self.on_error(str(e))
                    except Exception as callback_error:
                        self.logger.error(f"Error en callback de error: {callback_error}")
        
        self.worker_running = False
        self.logger.info("Worker de procesamiento terminado")
    
    def _transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, options: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribir audio usando faster-whisper"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Iniciando transcripción de audio: {len(audio_data)} muestras, {sample_rate} Hz")
            
            # Cargar modelo si no está cargado
            if not self.model_loaded:
                if not self.load_model():
                    raise Exception("No se pudo cargar el modelo faster-whisper")
            
            # Preparar audio
            audio_prepared = self._prepare_audio(audio_data, sample_rate)
            self.logger.info(f"Audio preparado: {len(audio_prepared)} muestras, dtype: {audio_prepared.dtype}")
            
            # Notificar progreso
            if self.on_progress:
                self.on_progress("Transcribiendo...")
            
            # Transcribir
            self.logger.info("Llamando a model.transcribe()...")
            result = self._transcribe_sync(audio_prepared, options)
            
            # Procesar resultado
            processed_result = self._process_result(result)
            
            # Actualizar estadísticas
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=True)
            
            processed_result['processing_time'] = processing_time
            
            self.logger.info(f"Transcripción completada en {processing_time:.2f}s: '{processed_result.get('text', '')}'")
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Error en transcripción faster-whisper: {e}")
            self._update_stats(time.time() - start_time, success=False)
            
            return {
                'text': '',
                'language': 'es',
                'confidence': 0.0,
                'segments': [],
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _prepare_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preparar audio para faster-whisper
        """
        # Convertir a float32 si es necesario
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Asegurar que esté en el rango correcto
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data
    
    def _transcribe_sync(self, audio_data: np.ndarray, options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Transcripción síncrona con faster-whisper optimizada para tiempo real
        """
        if options is None:
            options = {}
            
        try:
            # Configuración optimizada para tiempo real
            task = options.get('task', self.task)
            
            # Parámetros base optimizados (compatibles con faster-whisper)
            transcribe_params = {
                'language': self.language,
                'task': task,
                'temperature': 0.0,          # Respuesta determinista
                'beam_size': 1,             # Decodificación greedy (más rápida)
                'best_of': 1,               # Solo una mejor opción
                'initial_prompt': "Concise translation:",  # Traducción breve
                'without_timestamps': True, # Sin marcas de tiempo
                'word_timestamps': False,   # Sin timestamps por palabra
                'compression_ratio_threshold': 1.5,  # Umbral de compresión
                'condition_on_previous_text': False, # No condicionar en texto previo
                'no_speech_threshold': 0.3  # Ignorar silencios
            }
            
            # Optimizaciones adicionales para modelo tiny
            if self.is_tiny_model:
                transcribe_params.update({
                    'beam_size': 1,        # Mínimo beam size
                    'best_of': 1,          # Solo una opción
                    'temperature': 0.0,    # Determinista
                    'compression_ratio_threshold': 2.0,  # Más permisivo
                    'no_speech_threshold': 0.4,  # Más permisivo con silencios
                    'initial_prompt': "Translate to English:"  # Prompt más simple
                })
                self.logger.debug("Aplicando optimizaciones específicas para modelo tiny")
            
            # Ejecutar transcripción/traducción con parámetros optimizados
            segments, info = self.model.transcribe(audio_data, **transcribe_params)
            
            # Convertir segmentos a lista
            segments_list = []
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": getattr(segment, 'avg_logprob', 0.0),
                    "no_speech_prob": getattr(segment, 'no_speech_prob', 0.0)
                }
                
                # Agregar timestamps de palabras si están disponibles
                if hasattr(segment, 'words') and segment.words:
                    segment_dict["words"] = [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": getattr(word, 'probability', 0.0)
                        }
                        for word in segment.words
                    ]
                
                segments_list.append(segment_dict)
            
            return {
                "segments": segments_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration
            }
            
        except Exception as e:
            self.logger.error(f"Error en transcripción síncrona: {e}")
            raise
    
    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar resultado de faster-whisper
        """
        try:
            segments = result.get("segments", [])
            
            # Combinar texto de todos los segmentos
            full_text = " ".join([seg["text"] for seg in segments if seg["text"].strip()])
            
            # Calcular confianza promedio
            confidences = []
            for seg in segments:
                if "avg_logprob" in seg:
                    # Convertir log probability a confianza (0-1)
                    confidence = np.exp(seg["avg_logprob"])
                    confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "text": full_text,
                "language": result.get("language", "es"),
                "confidence": float(avg_confidence),
                "segments": segments,
                "duration": result.get("duration", 0.0),
                "language_probability": result.get("language_probability", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error procesando resultado: {e}")
            return {"error": str(e)}
    
    def _update_stats(self, processing_time: float, success: bool = True):
        """
        Actualizar estadísticas de procesamiento
        """
        self.stats["total_requests"] += 1
        self.stats["total_processing_time"] += processing_time
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Calcular tiempo promedio
        if self.stats["total_requests"] > 0:
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_requests"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de uso
        """
        return self.stats.copy()
    
    def stop_processing(self):
        """
        Detener procesamiento actual
        """
        self.stop_requested = True
        self.logger.info("Deteniendo procesamiento faster-whisper")
    
    def cleanup(self):
        """
        Limpiar recursos
        """
        self.logger.info("Limpiando FasterWhisperHandler...")
        
        # Detener procesamiento
        self.stop_processing()
        
        # Detener worker thread
        if self.worker_running:
            self.worker_running = False
            # Agregar señal de parada a la cola
            try:
                self.processing_queue.put_nowait(None)
            except queue.Empty:
                pass
            
            # Esperar a que termine el worker
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
                if self.worker_thread.is_alive():
                    self.logger.warning("Worker thread no terminó en el tiempo esperado")
        
        # Limpiar cola
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
                self.processing_queue.task_done()
            except queue.Empty:
                break
        
        # Limpiar modelo
        if self.model:
            try:
                # faster-whisper no requiere limpieza explícita
                self.model = None
                self.model_loaded = False
                if hasattr(self, '_current_model_size'):
                    delattr(self, '_current_model_size')
                self.logger.info("Modelo faster-whisper liberado")
            except Exception as e:
                self.logger.error(f"Error liberando modelo: {e}")
        
        self.logger.info("FasterWhisperHandler limpiado")
    
    def get_available_models(self) -> List[str]:
        """
        Obtener modelos disponibles para faster-whisper
        """
        return ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo actual
        """
        return {
            "engine": "faster-whisper",
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "task": self.task,
            "loaded": self.model_loaded,
            "optimized_for_speed": self.optimize_for_speed,
            "is_tiny_model": self.is_tiny_model
        }
    
    def transcribe_realtime(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcripción optimizada específicamente para tiempo real
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Transcripción en tiempo real: {len(audio_data)} muestras")
            
            # Cargar modelo si no está cargado
            if not self.model_loaded:
                if not self.load_model():
                    raise Exception("No se pudo cargar el modelo faster-whisper")
            
            # Preparar audio
            audio_prepared = self._prepare_audio(audio_data, sample_rate)
            
            # Configuración ultra-optimizada para tiempo real
            realtime_params = {
                'language': self.language,
                'task': 'translate',
                'temperature': 0.0,
                'beam_size': 1,
                'best_of': 1,
                'initial_prompt': "Translate:",
                'without_timestamps': True,
                'word_timestamps': False,
                'compression_ratio_threshold': 2.0,
                'condition_on_previous_text': False,
                'no_speech_threshold': 0.5,  # Más permisivo para tiempo real
                'patience': 1,  # Mínima paciencia para decodificación
                'length_penalty': 0.0  # Sin penalización de longitud
            }
            
            # Ejecutar transcripción
            segments, info = self.model.transcribe(audio_prepared, **realtime_params)
            
            # Procesar resultado rápidamente
            result = self._process_result_fast(segments, info)
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['realtime_optimized'] = True
            
            self.logger.info(f"Transcripción en tiempo real completada en {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en transcripción en tiempo real: {e}")
            return {
                'text': '',
                'language': 'es',
                'confidence': 0.0,
                'segments': [],
                'error': str(e),
                'processing_time': time.time() - start_time,
                'realtime_optimized': True
            }
    
    def _process_result_fast(self, segments, info) -> Dict[str, Any]:
        """
        Procesamiento rápido del resultado para tiempo real
        """
        try:
            # Combinar texto de todos los segmentos rápidamente
            full_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
            
            # Calcular confianza promedio simplificado
            confidences = []
            for seg in segments:
                if hasattr(seg, 'avg_logprob'):
                    confidence = max(0.0, min(1.0, np.exp(seg.avg_logprob)))
                    confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "text": full_text,
                "language": getattr(info, 'language', 'es'),
                "confidence": float(avg_confidence),
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                        "avg_logprob": getattr(seg, 'avg_logprob', 0.0)
                    }
                    for seg in segments
                ],
                "duration": getattr(info, 'duration', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error procesando resultado rápido: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """
        Destructor para limpiar recursos
        """
        try:
            self.cleanup()
        except:
            pass


# Test básico
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from config import Config
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Test básico
    config = Config()
    handler = FasterWhisperHandler(config)
    
    print(f"Modelos disponibles: {handler.get_available_models()}")
    print(f"Información del modelo: {handler.get_model_info()}")
    
    # Test de carga de modelo
    if handler.load_model():
        print("Modelo cargado exitosamente")
    else:
        print("Error cargando modelo")