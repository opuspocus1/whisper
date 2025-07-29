# -*- coding: utf-8 -*-
"""
Gestor de Audio para el Traductor de Voz en Tiempo Real
Maneja captura, reproducción y procesamiento de audio
"""

import pyaudio
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
import io
from typing import Optional, Callable, List, Tuple, Dict
import logging
from pathlib import Path

try:
    import noisereduce as nr
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    NOISE_REDUCTION_AVAILABLE = False
    logging.warning("noisereduce no disponible. Reducción de ruido deshabilitada.")

from config import config

class AudioManager:
    """Gestor principal de audio"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.audio_config = config.get_audio_config()
        
        # PyAudio para captura
        self.pyaudio = pyaudio.PyAudio()
        
        # Configuración de audio
        self.sample_rate = self.audio_config['sample_rate']
        self.channels = self.audio_config['channels']
        self.chunk_size = self.audio_config['chunk_size']
        self.format = pyaudio.paInt16
        
        # Estados
        self.is_recording = False
        self.is_playing = False
        
        # Streams
        self.input_stream = None
        self.output_stream = None
        
        # Buffers y colas
        self.audio_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        
        # Callbacks
        self.on_audio_data: Optional[Callable] = None
        self.on_silence_detected: Optional[Callable] = None
        self.on_speech_detected: Optional[Callable] = None
        
        # Detección de voz
        self.silence_threshold = config.get('advanced', 'silence_detection_threshold', 0.01)
        self.min_speech_duration = config.get('advanced', 'min_speech_duration', 0.5)
        self.max_silence_duration = config.get('advanced', 'max_silence_duration', 2.0)
        
        # Buffer para detección de actividad de voz
        self.vad_buffer = []
        self.speech_start_time = None
        self.last_speech_time = None
        
        # Threads
        self.recording_thread = None
        self.playback_thread = None
        self.processing_thread = None
        
        # Dispositivos
        self.input_device = self.audio_config.get('input_device')
        self.output_device = self.audio_config.get('output_device')
        
        self.logger.info("AudioManager inicializado")
    
    def set_callbacks(self, on_audio_data=None, on_error=None, on_silence_detected=None, on_speech_detected=None):
        """Establecer callbacks para eventos de audio"""
        if on_audio_data:
            self.on_audio_data = on_audio_data
        if on_error:
            self.on_error = on_error
        if on_silence_detected:
            self.on_silence_detected = on_silence_detected
        if on_speech_detected:
            self.on_speech_detected = on_speech_detected
    
    def get_audio_devices(self) -> Dict[str, List[Dict]]:
        """Obtener lista de dispositivos de audio disponibles"""
        devices = {
            'input': [],
            'output': []
        }
        
        try:
            device_count = self.pyaudio.get_device_count()
            
            for i in range(device_count):
                device_info = self.pyaudio.get_device_info_by_index(i)
                
                device_data = {
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'] if device_info['maxInputChannels'] > 0 else device_info['maxOutputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                }
                
                if device_info['maxInputChannels'] > 0:
                    devices['input'].append(device_data)
                
                if device_info['maxOutputChannels'] > 0:
                    devices['output'].append(device_data)
            
            self.logger.info(f"Encontrados {len(devices['input'])} dispositivos de entrada y {len(devices['output'])} de salida")
            
        except Exception as e:
            self.logger.error(f"Error obteniendo dispositivos de audio: {e}")
        
        return devices
    
    def get_input_devices(self) -> Dict[int, str]:
        """Obtener dispositivos de entrada como diccionario {index: name}"""
        devices = self.get_audio_devices()
        return {device['index']: device['name'] for device in devices['input']}
    
    def get_output_devices(self) -> Dict[int, str]:
        """Obtener dispositivos de salida como diccionario {index: name}"""
        devices = self.get_audio_devices()
        return {device['index']: device['name'] for device in devices['output']}
    
    def set_input_device(self, device_index: Optional[int]):
        """Establecer dispositivo de entrada"""
        self.input_device = device_index
        config.set('audio', 'input_device', device_index)
        self.logger.info(f"Dispositivo de entrada establecido: {device_index}")
    
    def set_output_device(self, device_index: Optional[int]):
        """Establecer dispositivo de salida"""
        self.output_device = device_index
        config.set('audio', 'output_device', device_index)
        self.logger.info(f"Dispositivo de salida establecido: {device_index}")
    
    def start_recording(self, callback: Optional[Callable] = None):
        """Iniciar grabación de audio"""
        if self.is_recording:
            self.logger.warning("Ya se está grabando")
            return
        
        if callback:
            self.on_audio_data = callback
            self.logger.info(f"Callback configurado: {callback}")
        else:
            self.logger.info(f"Usando callback previamente configurado: {self.on_audio_data}")
        
        self.is_recording = True
        
        try:
            # Configurar stream de entrada
            self.input_stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Iniciar thread de procesamiento
            self.processing_thread = threading.Thread(target=self._process_audio_data, daemon=True)
            self.processing_thread.start()
            
            self.logger.info("Grabación iniciada")
            
        except Exception as e:
            self.logger.error(f"Error iniciando grabación: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self):
        """Detener grabación de audio"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        try:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
            
            self.logger.info("Grabación detenida")
            
        except Exception as e:
            self.logger.error(f"Error deteniendo grabación: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback para datos de audio entrantes"""
        if status:
            self.logger.warning(f"Estado de audio: {status}")
        
        # Convertir datos a numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Agregar a la cola para procesamiento
        try:
            self.audio_queue.put_nowait((audio_data, time.time()))
        except queue.Full:
            self.logger.warning("Cola de audio llena, descartando datos")
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_data(self):
        """Procesar datos de audio en thread separado"""
        audio_buffer = []
        self.logger.info("Thread de procesamiento de audio iniciado")
        
        while self.is_recording:
            try:
                # Obtener datos de la cola
                audio_data, timestamp = self.audio_queue.get(timeout=0.1)
                self.logger.debug(f"Audio recibido: {len(audio_data)} muestras")
                
                # Aplicar procesamiento de audio
                processed_audio = self._apply_audio_processing(audio_data)
                
                # Agregar al buffer
                audio_buffer.extend(processed_audio)
                
                # Detectar actividad de voz
                voice_detected = self._detect_voice_activity(processed_audio)
                if voice_detected:
                    if self.speech_start_time is None:
                        self.speech_start_time = timestamp
                        self.logger.info("Inicio de voz detectado")
                        if self.on_speech_detected:
                            self.on_speech_detected()
                    
                    self.last_speech_time = timestamp
                
                # Verificar si hay suficiente audio para procesar
                if len(audio_buffer) >= self.sample_rate * 2:  # 2 segundos de audio
                    self.logger.debug(f"Buffer de audio: {len(audio_buffer)} muestras ({len(audio_buffer)/self.sample_rate:.2f}s)")
                    
                    # Verificar si hay silencio prolongado
                    if (self.last_speech_time and 
                        timestamp - self.last_speech_time > self.max_silence_duration):
                        
                        self.logger.info(f"Silencio prolongado detectado. Procesando {len(audio_buffer)} muestras")
                        
                        # Procesar el audio acumulado
                        if len(audio_buffer) > self.sample_rate * self.min_speech_duration:
                            audio_array = np.array(audio_buffer, dtype=np.int16)
                            
                            self.logger.info(f"Llamando callback on_audio_data con {len(audio_array)} muestras")
                            if self.on_audio_data:
                                self.on_audio_data(audio_array, self.sample_rate)
                            else:
                                self.logger.warning("Callback on_audio_data no está configurado")
                            
                            if self.on_silence_detected:
                                self.on_silence_detected()
                        else:
                            self.logger.info(f"Audio muy corto ({len(audio_buffer)/self.sample_rate:.2f}s), descartando")
                        
                        # Resetear buffer y estados
                        audio_buffer = []
                        self.speech_start_time = None
                        self.last_speech_time = None
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error procesando audio: {e}")
        
        # Procesar cualquier audio restante en la cola
        self.logger.info("Procesando audio restante en la cola...")
        try:
            while True:
                audio_data, timestamp = self.audio_queue.get_nowait()
                processed_audio = self._apply_audio_processing(audio_data)
                audio_buffer.extend(processed_audio)
                self.logger.debug(f"Audio adicional procesado: {len(audio_data)} muestras")
        except queue.Empty:
            pass
        
        # Procesar audio restante cuando se detiene la grabación
        self.logger.info(f"Grabación detenida. Procesando audio restante: {len(audio_buffer)} muestras")
        if len(audio_buffer) > self.sample_rate * self.min_speech_duration:
            audio_array = np.array(audio_buffer, dtype=np.int16)
            self.logger.info(f"Procesando audio final con {len(audio_array)} muestras ({len(audio_array)/self.sample_rate:.2f}s)")
            if self.on_audio_data:
                self.on_audio_data(audio_array, self.sample_rate)
            else:
                self.logger.warning("Callback on_audio_data no está configurado")
        else:
            self.logger.info(f"Audio restante muy corto ({len(audio_buffer)/self.sample_rate:.2f}s), descartando")
        
        self.logger.info("Thread de procesamiento de audio terminado")
    
    def _apply_audio_processing(self, audio_data: np.ndarray) -> np.ndarray:
        """Aplicar procesamiento de audio (reducción de ruido, etc.)"""
        processed = audio_data.copy()
        
        try:
            # Normalizar volumen
            if self.audio_config.get('auto_gain_control', True):
                processed = self._normalize_audio(processed)
            
            # Reducción de ruido
            if (self.audio_config.get('noise_reduction', True) and 
                NOISE_REDUCTION_AVAILABLE):
                processed = self._reduce_noise(processed)
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento de audio: {e}")
            return audio_data
        
        return processed
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalizar volumen del audio"""
        if len(audio_data) == 0:
            return audio_data
        
        # Calcular RMS
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        
        if rms > 0:
            # Normalizar a un nivel objetivo
            target_rms = 3000  # Ajustar según necesidades
            gain = target_rms / rms
            gain = min(gain, 3.0)  # Limitar ganancia máxima
            
            normalized = audio_data.astype(np.float32) * gain
            normalized = np.clip(normalized, -32768, 32767)
            return normalized.astype(np.int16)
        
        return audio_data
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Aplicar reducción de ruido"""
        try:
            # Convertir a float para noisereduce
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Aplicar reducción de ruido
            reduced = nr.reduce_noise(
                y=audio_float,
                sr=self.sample_rate,
                stationary=False,
                prop_decrease=0.8
            )
            
            # Convertir de vuelta a int16
            reduced_int = (reduced * 32768.0).astype(np.int16)
            return reduced_int
            
        except Exception as e:
            self.logger.error(f"Error en reducción de ruido: {e}")
            return audio_data
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Detectar actividad de voz en el audio"""
        if len(audio_data) == 0:
            return False
        
        # Calcular energía del audio
        energy = np.mean(np.abs(audio_data.astype(np.float32)))
        
        # Comparar con umbral
        return energy > self.silence_threshold * 32768
    
    def play_audio_data(self, audio_data: np.ndarray, sample_rate: int = None):
        """Reproducir datos de audio"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # Agregar a la cola de reproducción
            self.playback_queue.put((audio_data, sample_rate))
            
            # Iniciar thread de reproducción si no está activo
            if not self.is_playing:
                self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
                self.playback_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error reproduciendo audio: {e}")
    
    def play_audio_file(self, file_path: str):
        """Reproducir archivo de audio"""
        try:
            audio_data, sample_rate = sf.read(file_path)
            
            # Convertir a formato correcto si es necesario
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Convertir a mono si es estéreo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1).astype(np.int16)
            
            self.play_audio_data(audio_data, sample_rate)
            
        except Exception as e:
            self.logger.error(f"Error reproduciendo archivo {file_path}: {e}")
    
    def _playback_worker(self):
        """Worker para reproducción de audio"""
        self.is_playing = True
        self.logger.info("DEBUG: _playback_worker iniciado")
        
        try:
            while True:
                try:
                    self.logger.info("DEBUG: Esperando audio en la cola...")
                    audio_data, sample_rate = self.playback_queue.get(timeout=1.0)
                    self.logger.info(f"DEBUG: Audio obtenido de la cola: {len(audio_data)} muestras a {sample_rate} Hz")
                    self.logger.info(f"DEBUG: Tipo de datos: {type(audio_data)}, dtype: {audio_data.dtype if hasattr(audio_data, 'dtype') else 'N/A'}")
                    self.logger.info(f"DEBUG: Dispositivo de salida: {self.output_device}")
                    
                    # Verificar que los datos sean válidos
                    if len(audio_data) == 0:
                        self.logger.warning("DEBUG: Audio vacío, saltando reproducción")
                        continue
                    
                    # Reproducir usando sounddevice
                    self.logger.info("DEBUG: Iniciando reproducción con sounddevice...")
                    sd.play(
                        audio_data,
                        samplerate=sample_rate,
                        device=self.output_device,
                        blocking=True
                    )
                    self.logger.info("DEBUG: Reproducción completada")
                    
                except queue.Empty:
                    self.logger.info("DEBUG: Cola vacía, terminando worker")
                    break
                except Exception as e:
                    self.logger.error(f"Error en reproducción: {e}")
                    import traceback
                    self.logger.error(f"DEBUG Traceback: {traceback.format_exc()}")
                    
        finally:
            self.is_playing = False
            self.logger.info("DEBUG: _playback_worker terminado")
    
    def save_audio(self, audio_data: np.ndarray, file_path: str, sample_rate: int = None):
        """Guardar audio en archivo"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # Normalizar datos para guardado
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            sf.write(file_path, audio_data, sample_rate)
            self.logger.info(f"Audio guardado en: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error guardando audio: {e}")
    
    def get_audio_level(self) -> float:
        """Obtener nivel actual de audio (0.0 - 1.0)"""
        try:
            if not self.audio_queue.empty():
                audio_data, _ = self.audio_queue.queue[-1]
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                return min(rms / 10000.0, 1.0)  # Normalizar a 0-1
        except:
            pass
        return 0.0
    
    def set_volume(self, input_volume: float = None, output_volume: float = None):
        """Establecer volúmenes de entrada y salida"""
        if input_volume is not None:
            config.set('audio', 'volume_input', max(0.0, min(1.0, input_volume)))
        
        if output_volume is not None:
            config.set('audio', 'volume_output', max(0.0, min(1.0, output_volume)))
    
    def cleanup(self):
        """Limpiar recursos"""
        self.logger.info("Limpiando AudioManager...")
        
        self.stop_recording()
        
        if self.pyaudio:
            self.pyaudio.terminate()
        
        # Detener threads
        self.is_playing = False
        
        self.logger.info("AudioManager limpiado")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

if __name__ == "__main__":
    # Prueba del AudioManager
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    audio_manager = AudioManager()
    
    print("Dispositivos de audio disponibles:")
    devices = audio_manager.get_audio_devices()
    
    print("\nDispositivos de entrada:")
    for device in devices['input']:
        print(f"  {device['index']}: {device['name']} ({device['channels']} canales, {device['sample_rate']} Hz)")
    
    print("\nDispositivos de salida:")
    for device in devices['output']:
        print(f"  {device['index']}: {device['name']} ({device['channels']} canales, {device['sample_rate']} Hz)")
    
    # Prueba de grabación
    def on_audio(audio_data, sample_rate):
        print(f"Audio recibido: {len(audio_data)} muestras a {sample_rate} Hz")
        # Reproducir el audio grabado
        audio_manager.play_audio_data(audio_data, sample_rate)
    
    print("\nIniciando grabación de prueba por 5 segundos...")
    audio_manager.start_recording(callback=on_audio)
    time.sleep(5)
    audio_manager.stop_recording()
    
    print("Prueba completada")
    audio_manager.cleanup()