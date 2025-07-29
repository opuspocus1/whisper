# -*- coding: utf-8 -*-
"""
Interfaz gráfica principal para el traductor de voz en tiempo real
Español a Inglés usando Whisper + ElevenLabs
"""

# Importar parches de compatibilidad primero
import httpcore_compat

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import logging
from pathlib import Path
import json
import os
from typing import Optional, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Importar componentes del proyecto
from config import config, Config
from audio_manager import AudioManager
from whisper_handler import WhisperHandler
from faster_whisper_handler import FasterWhisperHandler
from tts_manager import TTSManager
from kokoro_handler import OptimizedKokoroHandler

from translator import Translator
from deepseek_handler import DeepSeekHandler

class VoiceTranslatorGUI:
    """Interfaz gráfica principal del traductor de voz"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuración
        self.ui_config = config.get_ui_config()
        
        # Estados
        self.is_recording = False
        self.is_processing = False
        self.is_playing = False
        
        # Configuración dinámica de Whisper
        self.whisper_task = "transcribe"  # Por defecto: modo IA habilitado
        
        # Componentes del sistema
        self.audio_manager = None
        self.whisper_handler = None
        self.tts_manager = None
        self.kokoro_handler = None
        self.translator = None
        
        # Performance monitoring
        self.performance_stats = {}
        self.timing_history = []
        
        # Parallel processing
        self.enable_parallel_processing = True
        self.processing_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ParallelProc")
        
        # Cola de mensajes para UI
        self.ui_queue = queue.Queue()
        
        # Crear ventana principal
        self.root = tk.Tk()
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        
        # Inicializar componentes
        self.initialize_components()
        
        # Configurar eventos
        self.setup_events()
        
        # Iniciar procesamiento de cola UI
        self.process_ui_queue()
        
        # Iniciar actualización periódica de estadísticas de rendimiento
        self.update_performance_stats_periodic()
        
        self.logger.info("GUI inicializada")
    
    def setup_window(self):
        """Configurar ventana principal"""
        self.root.title("Traductor de Voz en Tiempo Real - Español a Inglés")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Centrar ventana
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.root.winfo_screenheight() // 2) - (600 // 2)
        self.root.geometry(f"800x600+{x}+{y}")
        
        # Configurar cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configurar estilos de la interfaz"""
        style = ttk.Style()
        
        # Configurar tema
        available_themes = style.theme_names()
        preferred_theme = self.ui_config.get('theme', 'clam')
        
        if preferred_theme in available_themes:
            style.theme_use(preferred_theme)
        else:
            style.theme_use('clam')
        
        # Estilos personalizados
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
        style.configure('Record.TButton', font=('Arial', 12, 'bold'))
        style.configure('Action.TButton', font=('Arial', 10))
    
    def create_widgets(self):
        """Crear widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Traductor de Voz en Tiempo Real", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # === SECCIÓN DE CONTROL ===
        control_frame = ttk.LabelFrame(main_frame, text="Control de Grabación", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Botón de grabación
        self.record_button = ttk.Button(control_frame, text="🎤 Iniciar Grabación", 
                                       style='Record.TButton', command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=(0, 10))
        
        # Estado de grabación
        self.status_label = ttk.Label(control_frame, text="Listo para grabar", style='Status.TLabel')
        self.status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Botón de configuración
        config_button = ttk.Button(control_frame, text="⚙️ Configuración", 
                                  style='Action.TButton', command=self.show_config)
        config_button.grid(row=0, column=2)
        
        # Checkbox para habilitar/deshabilitar traducción IA
        self.ai_translation_var = tk.BooleanVar(value=True)  # Habilitado por defecto
        self.ai_translation_check = ttk.Checkbutton(control_frame, 
                                                   text="🤖 Habilitar traducción IA",
                                                   variable=self.ai_translation_var,
                                                   command=self.on_ai_translation_toggle)
        self.ai_translation_check.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        # === SECCIÓN DE AUDIO ===
        audio_frame = ttk.LabelFrame(main_frame, text="Configuración de Audio", padding="10")
        audio_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        audio_frame.columnconfigure(1, weight=1)
        
        # Dispositivo de entrada
        ttk.Label(audio_frame, text="Micrófono:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(audio_frame, textvariable=self.input_device_var, 
                                              state="readonly", width=40)
        self.input_device_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.input_device_combo.bind('<<ComboboxSelected>>', self.on_input_device_changed)
        
        # Dispositivo de salida
        ttk.Label(audio_frame, text="Altavoces:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(audio_frame, textvariable=self.output_device_var, 
                                               state="readonly", width=40)
        self.output_device_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.output_device_combo.bind('<<ComboboxSelected>>', self.on_output_device_changed)
        
        # Volumen
        ttk.Label(audio_frame, text="Volumen:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.volume_var = tk.DoubleVar(value=0.8)
        volume_scale = ttk.Scale(audio_frame, from_=0.0, to=1.0, variable=self.volume_var, 
                                orient=tk.HORIZONTAL, command=self.on_volume_changed)
        volume_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.volume_label = ttk.Label(audio_frame, text="80%")
        self.volume_label.grid(row=2, column=2)
        
        # === SECCIÓN DE TEXTO ===
        text_frame = ttk.LabelFrame(main_frame, text="Texto y Traducción", padding="10")
        text_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(1, weight=1)
        text_frame.rowconfigure(1, weight=1)
        
        # Texto original (español)
        ttk.Label(text_frame, text="Texto Original (Español):", style='Subtitle.TLabel').grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.original_text = tk.Text(text_frame, height=8, wrap=tk.WORD, font=('Arial', 11))
        self.original_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Scrollbar para texto original
        original_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.original_text.yview)
        original_scroll.grid(row=1, column=0, sticky=(tk.E, tk.N, tk.S))
        self.original_text.configure(yscrollcommand=original_scroll.set)
        
        # Texto traducido (inglés)
        ttk.Label(text_frame, text="Texto Traducido (Inglés):", style='Subtitle.TLabel').grid(
            row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        self.translated_text = tk.Text(text_frame, height=8, wrap=tk.WORD, font=('Arial', 11))
        self.translated_text.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Scrollbar para texto traducido
        translated_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.translated_text.yview)
        translated_scroll.grid(row=1, column=1, sticky=(tk.E, tk.N, tk.S))
        self.translated_text.configure(yscrollcommand=translated_scroll.set)
        
        # === SECCIÓN DE TIEMPOS DE PROCESAMIENTO ===
        timing_frame = ttk.LabelFrame(main_frame, text="⚡ Tiempos de Procesamiento en Tiempo Real", padding="10")
        timing_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        timing_frame.columnconfigure(1, weight=1)
        timing_frame.columnconfigure(3, weight=1)
        
        # Primera fila - Tiempos principales
        ttk.Label(timing_frame, text="🎤 Transcripción:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.transcription_time_label = ttk.Label(timing_frame, text="--", font=('Arial', 10, 'bold'))
        self.transcription_time_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(timing_frame, text="🤖 IA/Traducción:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.correction_time_label = ttk.Label(timing_frame, text="--", font=('Arial', 10, 'bold'))
        self.correction_time_label.grid(row=0, column=3, sticky=tk.W)
        
        # Segunda fila - Síntesis y total
        ttk.Label(timing_frame, text="🔊 Síntesis:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.synthesis_time_label = ttk.Label(timing_frame, text="--", font=('Arial', 10, 'bold'))
        self.synthesis_time_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(timing_frame, text="⏱️ Tiempo Total:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5))
        self.total_time_label = ttk.Label(timing_frame, text="--", font=('Arial', 10, 'bold'), foreground='blue')
        self.total_time_label.grid(row=1, column=3, sticky=tk.W)
        
        # Tercera fila - Métricas detalladas de Kokoro
        ttk.Label(timing_frame, text="🚀 Primer Byte:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.first_byte_label = ttk.Label(timing_frame, text="--", font=('Arial', 9, 'bold'), foreground='green')
        self.first_byte_label.grid(row=2, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(timing_frame, text="📊 Promedio:").grid(row=2, column=2, sticky=tk.W, padx=(0, 5))
        self.avg_latency_label = ttk.Label(timing_frame, text="--", font=('Arial', 9, 'bold'), foreground='purple')
        self.avg_latency_label.grid(row=2, column=3, sticky=tk.W)
        
        # Cuarta fila - Estadísticas de rendimiento
        ttk.Label(timing_frame, text="📈 Requests:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        self.requests_count_label = ttk.Label(timing_frame, text="--", font=('Arial', 9, 'bold'), foreground='orange')
        self.requests_count_label.grid(row=3, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(timing_frame, text="🎯 P95:").grid(row=3, column=2, sticky=tk.W, padx=(0, 5))
        self.p95_latency_label = ttk.Label(timing_frame, text="--", font=('Arial', 9, 'bold'), foreground='red')
        self.p95_latency_label.grid(row=3, column=3, sticky=tk.W)
        
        # === SECCIÓN DE ACCIONES ===
        actions_frame = ttk.Frame(main_frame)
        actions_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        actions_frame.columnconfigure(1, weight=1)
        
        # Botones de acción
        self.play_button = ttk.Button(actions_frame, text="🔊 Reproducir", 
                                     style='Action.TButton', command=self.play_audio, state=tk.DISABLED)
        self.play_button.grid(row=0, column=0, padx=(0, 10))
        
        self.clear_button = ttk.Button(actions_frame, text="🗑️ Limpiar", 
                                      style='Action.TButton', command=self.clear_text)
        self.clear_button.grid(row=0, column=1, padx=(0, 10))
        
        self.save_button = ttk.Button(actions_frame, text="💾 Guardar", 
                                     style='Action.TButton', command=self.save_translation)
        self.save_button.grid(row=0, column=2)
        
        # === BARRA DE ESTADO ===
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))
        status_frame.columnconfigure(1, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           mode='determinate', length=200)
        self.progress_bar.grid(row=0, column=0, padx=(0, 10))
        
        self.status_text = ttk.Label(status_frame, text="Listo")
        self.status_text.grid(row=0, column=1, sticky=tk.W)
        
        # Configurar pesos de filas
        main_frame.rowconfigure(3, weight=1)
    
    def _create_whisper_handler(self):
        """Factory para crear el handler de Whisper apropiado según configuración"""
        whisper_config = config.get_whisper_config()
        engine = whisper_config.get("engine", "whisper")
        
        if engine == "faster-whisper":
            self.logger.info("Inicializando FasterWhisperHandler")
            return FasterWhisperHandler(config)
        else:
            self.logger.info("Inicializando WhisperHandler (original)")
            return WhisperHandler()
    
    def initialize_components(self):
        """Inicializar componentes del sistema"""
        try:
            # Inicializar AudioManager
            self.update_status("Inicializando sistema de audio...")
            self.audio_manager = AudioManager()
            self.audio_manager.set_callbacks(
                on_audio_data=self.on_audio_data,
                on_error=self.on_audio_error
            )
            
            # Cargar dispositivos de audio
            self.load_audio_devices()
            
            # Inicializar WhisperHandler (usando factory)
            whisper_config = config.get_whisper_config()
            engine = whisper_config.get("engine", "whisper")
            self.update_status(f"Cargando modelo {engine}...")
            
            self.whisper_handler = self._create_whisper_handler()
            self.whisper_handler.set_callbacks(
                on_transcription=self.on_transcription_ready,
                on_error=self.on_whisper_error
            )
            
            # Cargar modelo Whisper
            if not self.whisper_handler.load_model():
                raise RuntimeError(f"No se pudo cargar el modelo {engine}")
            
            # Translator deshabilitado - solo usamos OpenRouter/Qwen para traducción
            # self.update_status("Inicializando traductor...")
            # self.translator = Translator()
            # self.translator.set_callbacks(
            #     on_translation_ready=self.on_translation_ready,
            #     on_error=self.on_translation_error
            # )
            self.translator = None  # Deshabilitado
            
            # Inicializar DeepSeekHandler
            self.update_status("Inicializando corrector de texto...")
            self.deepseek_handler = DeepSeekHandler()
            
            # Inicializar TTS Manager (reemplaza ElevenLabsHandler)
            self.update_status("Inicializando síntesis de voz...")
            self.tts_manager = TTSManager()
            self.tts_manager.set_callbacks(
                on_audio_ready=self.on_synthesis_ready,
                on_error=self.on_synthesis_error
            )
            
            # Inicializar KokoroHandler optimizado para latencia
            self.update_status("Inicializando KokoroHandler optimizado...")
            self.kokoro_handler = OptimizedKokoroHandler(
                api_url="http://localhost:5000",
                voice="af_heart",
                max_workers=4,
                connection_pool_size=10,
                chunk_size=512,
                pre_warm_models=True
            )
            self.kokoro_handler.on_audio_ready = self.on_kokoro_synthesis_ready
            self.kokoro_handler.on_error = self.on_kokoro_synthesis_error
            
            self.update_status("Sistema listo")
            
        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {e}")
            self.update_status(f"Error: {e}")
            messagebox.showerror("Error de Inicialización", 
                               f"No se pudo inicializar el sistema:\n{e}")
    
    def load_audio_devices(self):
        """Cargar dispositivos de audio disponibles"""
        try:
            # Dispositivos de entrada
            input_devices = self.audio_manager.get_input_devices()
            input_names = [f"{i}: {name}" for i, name in input_devices.items()]
            self.input_device_combo['values'] = input_names
            
            if input_names:
                self.input_device_combo.current(0)
                self.on_input_device_changed()
            
            # Dispositivos de salida
            output_devices = self.audio_manager.get_output_devices()
            output_names = [f"{i}: {name}" for i, name in output_devices.items()]
            self.output_device_combo['values'] = output_names
            
            if output_names:
                self.output_device_combo.current(0)
                self.on_output_device_changed()
            
        except Exception as e:
            self.logger.error(f"Error cargando dispositivos de audio: {e}")
    
    def setup_events(self):
        """Configurar eventos de la interfaz"""
        # Atajos de teclado
        self.root.bind('<F1>', lambda e: self.toggle_recording())
        
        self.root.bind('<F3>', lambda e: self.clear_text())
        self.root.bind('<Control-s>', lambda e: self.save_translation())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
    
    def process_ui_queue(self):
        """Procesar cola de mensajes para UI"""
        try:
            while True:
                message = self.ui_queue.get_nowait()
                message_type = message.get('type')
                
                if message_type == 'status':
                    self.status_text.config(text=message['text'])
                elif message_type == 'progress':
                    self.progress_var.set(message['value'])
                elif message_type == 'transcription':
                    self.display_transcription(message['text'])
                elif message_type == 'translation':
                    self.display_translation(message['text'])
                elif message_type == 'error':
                    self.show_error(message['text'])
                
        except queue.Empty:
            pass
        
        # Programar siguiente verificación
        self.root.after(100, self.process_ui_queue)
    
    def update_status(self, text: str):
        """Actualizar texto de estado"""
        self.ui_queue.put({'type': 'status', 'text': text})
    
    def update_progress(self, value: float):
        """Actualizar barra de progreso"""
        self.ui_queue.put({'type': 'progress', 'value': value})
    
    def show_error(self, error_text: str):
        """Mostrar error en la interfaz"""
        self.logger.error(error_text)
        messagebox.showerror("Error", error_text)
    
    def update_timing_display(self, timing_type: str, time_value: float):
        """Actualizar tiempos en la interfaz con formato mejorado"""
        # Mostrar en segundos con 3 decimales para mayor precisión
        time_text = f"{time_value:.3f}s"
        
        # Para latencias muy bajas, mostrar en milisegundos
        if time_value < 1.0:
            time_text = f"{time_value*1000:.1f}ms"
        
        if timing_type == 'transcription':
            self.transcription_time_label.config(text=time_text, foreground='green')
        elif timing_type == 'correction':
            self.correction_time_label.config(text=time_text, foreground='orange')
        elif timing_type == 'synthesis':
            self.synthesis_time_label.config(text=time_text, foreground='purple')
        elif timing_type == 'total':
            self.total_time_label.config(text=time_text, foreground='blue')
        elif timing_type == 'first_byte':
            self.first_byte_label.config(text=time_text, foreground='green')
        
        # Actualizar estadísticas de rendimiento si hay KokoroHandler
        self.update_performance_stats()
    
    def reset_timing_display(self):
        """Resetear tiempos en la interfaz"""
        self.transcription_time_label.config(text="--", foreground='black')
        self.correction_time_label.config(text="--", foreground='black')
        self.synthesis_time_label.config(text="--", foreground='black')
        self.total_time_label.config(text="--", foreground='black')
        self.first_byte_label.config(text="--", foreground='black')
        self.avg_latency_label.config(text="--", foreground='black')
        self.requests_count_label.config(text="--", foreground='black')
        self.p95_latency_label.config(text="--", foreground='black')
    
    def update_performance_stats(self):
        """Actualizar estadísticas de rendimiento del KokoroHandler"""
        if self.kokoro_handler:
            try:
                stats = self.kokoro_handler.get_performance_stats()
                if stats:
                    # Actualizar etiquetas con estadísticas
                    avg_latency = stats.get('avg_first_byte_latency', 0)
                    total_requests = stats.get('total_requests', 0)
                    p95_latency = stats.get('p95_first_byte_latency', 0)
                    
                    self.avg_latency_label.config(text=f"{avg_latency*1000:.1f}ms", foreground='purple')
                    self.requests_count_label.config(text=str(total_requests), foreground='orange')
                    self.p95_latency_label.config(text=f"{p95_latency*1000:.1f}ms", foreground='red')
                    
                    # Guardar estadísticas para uso posterior
                    self.performance_stats = stats
                    
            except Exception as e:
                self.logger.error(f"Error actualizando estadísticas: {e}")
    
    def update_first_byte_latency(self, latency: float):
        """Actualizar latencia de primer byte específicamente"""
        self.update_timing_display('first_byte', latency)
    
    def update_performance_stats_periodic(self):
        """Actualizar estadísticas de rendimiento periódicamente"""
        try:
            self.update_performance_stats()
        except Exception as e:
            self.logger.error(f"Error en actualización periódica: {e}")
        
        # Programar próxima actualización en 2 segundos
        self.root.after(2000, self.update_performance_stats_periodic)
    
    # === EVENTOS DE AUDIO ===
    
    def on_input_device_changed(self, event=None):
        """Cambio de dispositivo de entrada"""
        if self.audio_manager and self.input_device_var.get():
            device_id = int(self.input_device_var.get().split(':')[0])
            self.audio_manager.set_input_device(device_id)
    
    def on_output_device_changed(self, event=None):
        """Cambio de dispositivo de salida"""
        if self.audio_manager and self.output_device_var.get():
            device_id = int(self.output_device_var.get().split(':')[0])
            self.audio_manager.set_output_device(device_id)
    
    def on_volume_changed(self, value):
        """Cambio de volumen"""
        volume = float(value)
        self.volume_label.config(text=f"{int(volume * 100)}%")
        if self.audio_manager:
            self.audio_manager.set_volume(volume)
    
    def toggle_recording(self):
        """Alternar grabación"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Iniciar grabación"""
        try:
            if not self.audio_manager:
                raise RuntimeError("Sistema de audio no inicializado")
            
            # Resetear tiempos en la interfaz
            self.reset_timing_display()
            
            # Pasar el callback explícitamente para asegurar que se use
            self.audio_manager.start_recording(callback=self.on_audio_data)
            self.is_recording = True
            
            self.record_button.config(text="⏹️ Detener Grabación")
            self.status_label.config(text="Grabando...")
            self.update_status("Grabando audio...")
            
            self.logger.info("Grabación iniciada")
            
        except Exception as e:
            self.logger.error(f"Error iniciando grabación: {e}")
            self.show_error(f"No se pudo iniciar la grabación:\n{e}")
    
    def stop_recording(self):
        """Detener grabación"""
        try:
            if self.audio_manager:
                self.audio_manager.stop_recording()
            
            self.is_recording = False
            
            self.record_button.config(text="🎤 Iniciar Grabación")
            self.status_label.config(text="Procesando...")
            self.update_status("Procesando audio...")
            
            self.logger.info("Grabación detenida")
            
        except Exception as e:
            self.logger.error(f"Error deteniendo grabación: {e}")
            self.show_error(f"Error deteniendo grabación:\n{e}")
    
    def on_audio_data(self, audio_data: np.ndarray, sample_rate: int = None):
        """Callback para datos de audio"""
        if self.whisper_handler and not self.is_processing:
            self.is_processing = True
            
            # Usar sample_rate del AudioManager si se proporciona
            if sample_rate is None:
                sample_rate = self.audio_manager.sample_rate
            
            # Iniciar medición de tiempo total
            self.process_start_time = time.time()
            self.logger.info(f"Audio recibido para procesamiento: {len(audio_data)} muestras a {sample_rate} Hz")
            self.logger.info("=== INICIANDO PROCESO DE TRADUCCIÓN ===")
            
            # Procesar en thread separado
            threading.Thread(
                target=self.process_audio_async,
                args=(audio_data, sample_rate),
                daemon=True
            ).start()
    
    def process_audio_async(self, audio_data: np.ndarray, sample_rate: int):
        """Procesar audio de forma asíncrona"""
        try:
            # Iniciar medición de tiempo de transcripción
            transcription_start = time.time()
            
            # Transcribir con Whisper
            self.update_status("Transcribiendo audio...")
            self.logger.info(f"Enviando audio a Whisper: {len(audio_data)} muestras a {sample_rate} Hz")
            # Pasar la tarea dinámica basada en el toggle de traducción IA
            success = self.whisper_handler.transcribe_async(audio_data, sample_rate=sample_rate, task=self.whisper_task)
            
            if not success:
                self.logger.error("No se pudo agregar audio a la cola de procesamiento")
                self.ui_queue.put({'type': 'error', 'text': "Error: Cola de procesamiento llena"})
                self.is_processing = False
            else:
                self.logger.info("Audio enviado exitosamente a la cola de Whisper")
                # Guardar tiempo de inicio para usar en callback
                self.transcription_start_time = transcription_start
            
        except Exception as e:
            self.logger.error(f"Error procesando audio: {e}")
            self.ui_queue.put({'type': 'error', 'text': f"Error procesando audio: {e}"})
            self.is_processing = False
    
    def on_audio_error(self, error_text: str):
        """Callback para errores de audio"""
        self.ui_queue.put({'type': 'error', 'text': f"Error de audio: {error_text}"})
    
    # === EVENTOS DE TRANSCRIPCIÓN ===
    
    def on_ai_translation_toggle(self):
        """Callback para el toggle de traducción IA"""
        is_enabled = self.ai_translation_var.get()
        status = "habilitada" if is_enabled else "deshabilitada"
        self.logger.info(f"Traducción IA {status}")
        
        # Guardar el modo actual para usar en las opciones de transcripción
        if is_enabled:
            # Modo IA: Whisper solo transcribe (español -> español)
            self.whisper_task = "transcribe"
        else:
            # Modo directo: Whisper traduce directamente (español -> inglés)
            self.whisper_task = "translate"
        
        self.logger.info(f"Whisper configurado para: {self.whisper_task}")
    
    def on_transcription_ready(self, result: Dict[str, Any]):
        """Callback para transcripción lista con procesamiento paralelo optimizado"""
        # Calcular tiempo de transcripción
        if hasattr(self, 'transcription_start_time'):
            transcription_time = time.time() - self.transcription_start_time
            self.logger.info(f"⏱️ TRANSCRIPCIÓN completada en {transcription_time:.2f} segundos")
            self.update_timing_display('transcription', transcription_time)
        
        # El resultado viene directamente de Whisper con 'text', 'language', etc.
        text = result.get('text', '').strip()
        if text and not result.get('error'):
            # Verificar si la traducción IA está habilitada
            ai_translation_enabled = self.ai_translation_var.get()
            
            if self.enable_parallel_processing:
                # Proceso paralelo optimizado
                self.logger.info("🚀 Usando procesamiento paralelo para traducción y síntesis")
                
                def parallel_process():
                    try:
                        if ai_translation_enabled:
                            # Modo IA: Whisper transcribió español -> español, usar OpenRouter/Qwen
                            self.ui_queue.put({'type': 'transcription', 'text': text})
                            self.update_status("Traduciendo con IA...")
                            self.correction_start_time = time.time()
                            
                            # Pre-calentar Kokoro mientras se traduce
                            if self.kokoro_handler and hasattr(self.kokoro_handler, '_pre_warm_models_async'):
                                self.kokoro_handler._pre_warm_models_async()
                            
                            self.correct_and_synthesize(text)
                        else:
                            # Modo directo: Whisper ya tradujo español -> inglés
                            self.ui_queue.put({'type': 'transcription', 'text': f"[Español detectado]"}) 
                            self.ui_queue.put({'type': 'translation', 'text': text})
                            
                            # Marcar tiempo de corrección como 0 (no se usa IA)
                            self.correction_start_time = time.time()
                            correction_time = 0
                            self.logger.info(f"⏱️ TRADUCCIÓN/IA (modo directo) completada en {correction_time:.2f} segundos")
                            self.update_timing_display('correction', correction_time)
                            
                            # Síntesis paralela con pre-calentamiento
                            self.synthesis_start_time = time.time()
                            self.update_status("Sintetizando voz con Kokoro optimizado...")
                            if self.kokoro_handler:
                                self.kokoro_handler.synthesize_async(
                                    text, 
                                    optimize_for_latency=True,
                                    use_streaming=True
                                )
                            else:
                                self.tts_manager.synthesize_async(text)
                    except Exception as e:
                        self.logger.error(f"Error en procesamiento paralelo: {e}")
                        self.is_processing = False
                
                # Ejecutar en thread pool para evitar bloqueos
                self.processing_executor.submit(parallel_process)
            else:
                # Procesamiento secuencial tradicional
                if ai_translation_enabled:
                    # Modo IA: Whisper transcribió español -> español, usar OpenRouter/Qwen
                    self.ui_queue.put({'type': 'transcription', 'text': text})
                    self.update_status("Traduciendo con IA...")
                    self.correction_start_time = time.time()
                    threading.Thread(
                        target=self.correct_and_synthesize,
                        args=(text,),
                        daemon=True
                    ).start()
                else:
                    # Modo directo: Whisper ya tradujo español -> inglés, ir directo a síntesis
                    self.ui_queue.put({'type': 'transcription', 'text': f"[Español detectado]"}) 
                    self.ui_queue.put({'type': 'translation', 'text': text})
                    
                    # Marcar tiempo de corrección como 0 (no se usa IA)
                    self.correction_start_time = time.time()
                    correction_time = 0
                    self.logger.info(f"⏱️ TRADUCCIÓN/IA (modo directo) completada en {correction_time:.2f} segundos")
                    self.update_timing_display('correction', correction_time)
                    
                    # Ir directo a síntesis con el texto ya traducido por Whisper
                    self.synthesis_start_time = time.time()
                    self.update_status("Sintetizando voz...")
                    self.tts_manager.synthesize_async(text)
        else:
            error_msg = result.get('error', 'No se detectó texto')
            self.update_status(f"Error: {error_msg}")
            self.is_processing = False
    
    def on_whisper_error(self, error_text: str):
        """Callback para errores de Whisper"""
        # Calcular tiempo total incluso en caso de error
        if hasattr(self, 'process_start_time'):
            total_time = time.time() - self.process_start_time
            self.logger.info(f"❌ PROCESO TOTAL (con error) completado en {total_time:.2f} segundos")
            self.logger.info("=== FIN DEL PROCESO DE TRADUCCIÓN (ERROR) ===")
            self.update_timing_display('total', total_time)
        
        self.ui_queue.put({'type': 'error', 'text': f"Error de transcripción: {error_text}"})
        self.is_processing = False
    
    # === EVENTOS DE SÍNTESIS ===
    
    def on_synthesis_ready(self, result: Dict[str, Any]):
        """Callback para síntesis de voz lista (TTS Manager)"""
        try:
            if result.get('success', False):
                # Calcular tiempo de síntesis
                if hasattr(self, 'synthesis_start_time'):
                    synthesis_time = time.time() - self.synthesis_start_time
                    self.logger.info(f"🔊 SÍNTESIS (TTS Manager) completada en {synthesis_time:.3f} segundos")
                    self.update_timing_display('synthesis', synthesis_time)
                
                # Reproducir audio si está disponible
                if 'audio_path' in result and self.audio_manager:
                    self.audio_manager.play_audio_file(result['audio_path'])
                    self.logger.info("Audio reproducido exitosamente")
                elif 'audio_data' in result and self.audio_manager:
                    # Fallback para compatibilidad
                    if isinstance(result['audio_data'], str):  # Es una ruta de archivo
                        self.audio_manager.play_audio_file(result['audio_data'])
                    else:  # Son datos de audio
                        self.audio_manager.play_audio_data(result['audio_data'])
                    self.logger.info("Audio reproducido exitosamente")
                
                # Calcular tiempo total
                if hasattr(self, 'process_start_time'):
                    total_time = time.time() - self.process_start_time
                    self.logger.info(f"🏁 PROCESO TOTAL completado en {total_time:.3f} segundos")
                    self.logger.info("=== FIN DEL PROCESO DE TRADUCCIÓN ===")
                    self.update_timing_display('total', total_time)
                
                self.update_status("Proceso completado")
            else:
                error_msg = result.get('error', 'Error desconocido en síntesis')
                self.logger.error(f"Error en síntesis: {error_msg}")
                self.update_status(f"Error en síntesis: {error_msg}")
                
                # Calcular tiempo total incluso con error
                if hasattr(self, 'process_start_time'):
                    total_time = time.time() - self.process_start_time
                    self.logger.info(f"❌ PROCESO TOTAL (con error de síntesis) completado en {total_time:.3f} segundos")
                    self.update_timing_display('total', total_time)
            
            self.is_processing = False
            
        except Exception as e:
            self.logger.error(f"Error procesando resultado de síntesis: {e}")
            self.update_status(f"Error procesando síntesis: {e}")
            self.is_processing = False
    
    def on_kokoro_synthesis_ready(self, result: Dict[str, Any]):
        """Callback para síntesis de voz lista (KokoroHandler optimizado)"""
        try:
            if result.get('success', False):
                # Calcular tiempo de síntesis
                if hasattr(self, 'synthesis_start_time'):
                    synthesis_time = time.time() - self.synthesis_start_time
                    self.logger.info(f"🚀 SÍNTESIS (Kokoro) completada en {synthesis_time:.3f} segundos")
                    self.update_timing_display('synthesis', synthesis_time)
                
                # Actualizar latencia de primer byte si está disponible
                if 'latency' in result:
                    latency = result['latency']
                    self.logger.info(f"⚡ Latencia de primer byte: {latency:.3f}s")
                    self.update_first_byte_latency(latency)
                
                # Reproducir audio si está disponible
                if 'audio_path' in result and self.audio_manager:
                    self.audio_manager.play_audio_file(result['audio_path'])
                    self.logger.info("Audio Kokoro reproducido exitosamente")
                elif 'audio_data' in result and self.audio_manager:
                    # Fallback para compatibilidad
                    if isinstance(result['audio_data'], str):  # Es una ruta de archivo
                        self.audio_manager.play_audio_file(result['audio_data'])
                    else:  # Son datos de audio
                        self.audio_manager.play_audio_data(result['audio_data'])
                    self.logger.info("Audio Kokoro reproducido exitosamente")
                
                # Calcular tiempo total
                if hasattr(self, 'process_start_time'):
                    total_time = time.time() - self.process_start_time
                    self.logger.info(f"🏁 PROCESO TOTAL (Kokoro) completado en {total_time:.3f} segundos")
                    self.logger.info("=== FIN DEL PROCESO DE TRADUCCIÓN (KOKORO) ===")
                    self.update_timing_display('total', total_time)
                
                self.update_status("Proceso completado (Kokoro optimizado)")
            else:
                error_msg = result.get('error', 'Error desconocido en síntesis Kokoro')
                self.logger.error(f"Error en síntesis Kokoro: {error_msg}")
                self.update_status(f"Error en síntesis Kokoro: {error_msg}")
                
                # Calcular tiempo total incluso con error
                if hasattr(self, 'process_start_time'):
                    total_time = time.time() - self.process_start_time
                    self.logger.info(f"❌ PROCESO TOTAL (con error de síntesis Kokoro) completado en {total_time:.3f} segundos")
                    self.update_timing_display('total', total_time)
            
            self.is_processing = False
            
        except Exception as e:
            self.logger.error(f"Error procesando resultado de síntesis Kokoro: {e}")
            self.update_status(f"Error procesando síntesis Kokoro: {e}")
            self.is_processing = False
    
    def on_kokoro_synthesis_error(self, error_text: str):
        """Callback para errores de síntesis Kokoro"""
        self.logger.error(f"Error de síntesis Kokoro: {error_text}")
        
        # Calcular tiempo total incluso en caso de error
        if hasattr(self, 'process_start_time'):
            total_time = time.time() - self.process_start_time
            self.logger.info(f"❌ PROCESO TOTAL (con error de síntesis Kokoro) completado en {total_time:.3f} segundos")
            self.logger.info("=== FIN DEL PROCESO DE TRADUCCIÓN (ERROR KOKORO) ===")
            self.update_timing_display('total', total_time)
        
        self.update_status(f"Error de síntesis Kokoro: {error_text}")
        self.is_processing = False
    
    def on_synthesis_error(self, error_text: str):
        """Callback para errores de síntesis"""
        self.logger.error(f"Error de síntesis: {error_text}")
        
        # Calcular tiempo total incluso en caso de error
        if hasattr(self, 'process_start_time'):
            total_time = time.time() - self.process_start_time
            self.logger.info(f"❌ PROCESO TOTAL (con error de síntesis) completado en {total_time:.2f} segundos")
            self.logger.info("=== FIN DEL PROCESO DE TRADUCCIÓN (ERROR) ===")
            self.update_timing_display('total', total_time)
        
        self.update_status(f"Error de síntesis: {error_text}")
        self.is_processing = False
    
    def display_transcription(self, text: str):
        """Mostrar transcripción en la interfaz"""
        self.original_text.insert(tk.END, f"{text}\n\n")
        self.original_text.see(tk.END)
    
    # === EVENTOS DE TRADUCCIÓN (DESHABILITADOS) ===
    # Los siguientes métodos ya no se usan porque el traductor está deshabilitado
    # La traducción se hace directamente con OpenRouter/Qwen
    
    # def on_translation_ready(self, result: Dict[str, Any]):
    #     """Callback para traducción lista"""
    #     if result['success']:
    #         self.ui_queue.put({'type': 'translation', 'text': result['translated_text']})
    #         
    #         # Corregir texto con DeepSeek antes de sintetizar
    #         self.update_status("Corrigiendo texto...")
    #         threading.Thread(
    #             target=self.correct_and_synthesize,
    #             args=(result['translated_text'],),
    #             daemon=True
    #         ).start()
    #     else:
    #         self.ui_queue.put({'type': 'error', 'text': f"Error de traducción: {result['error']}"})
    #         self.is_processing = False
    # 
    # def on_translation_error(self, error_text: str):
    #     """Callback para errores de traducción"""
    #     self.ui_queue.put({'type': 'error', 'text': f"Error de traducción: {error_text}"})
    #     self.is_processing = False
    
    def correct_and_synthesize(self, spanish_text: str):
        """Traducir y corregir texto español con OpenRouter/Qwen, luego sintetizar con ElevenLabs - Optimizado para paralelo"""
        try:
            # Verificar si OpenRouter/Qwen está habilitado
            deepseek_config = config.get_deepseek_config()
            
            if self.enable_parallel_processing:
                # Pre-calentar Kokoro en paralelo mientras se traduce
                kokoro_warmup_future = None
                if self.kokoro_handler and hasattr(self.kokoro_handler, '_pre_warm_models_async'):
                    self.logger.info("🔥 Pre-calentando Kokoro en paralelo con traducción")
                    kokoro_warmup_future = self.processing_executor.submit(
                        self.kokoro_handler._pre_warm_models_async
                    )
            
            if deepseek_config.get('enabled', False) and deepseek_config.get('api_key'):
                # Traducir y corregir texto español directamente con OpenRouter/Qwen
                corrected_result = self.deepseek_handler.correct_text(spanish_text)
                
                # Calcular tiempo de corrección después de que termine OpenRouter/Qwen
                if hasattr(self, 'correction_start_time'):
                    correction_time = time.time() - self.correction_start_time
                    self.logger.info(f"⏱️ TRADUCCIÓN/IA completada en {correction_time:.2f} segundos")
                    self.update_timing_display('correction', correction_time)
                
                if corrected_result['success']:
                    english_text = corrected_result['corrected_text']
                    self.logger.info(f"Texto traducido y corregido por OpenRouter/Qwen: '{spanish_text}' -> '{english_text}'")
                    
                    # Mostrar traducción en la UI
                    self.ui_queue.put({
                        'type': 'translation', 
                        'text': english_text
                    })
                    
                    # Usar texto traducido para síntesis
                    final_text = english_text
                else:
                    self.logger.warning(f"Error traduciendo texto: {corrected_result['error']}")
                    # Usar texto español original si hay error
                    final_text = spanish_text
                    self.ui_queue.put({
                        'type': 'translation', 
                        'text': f"[Error - texto original] {spanish_text}"
                    })
            else:
                # OpenRouter/Qwen deshabilitado, usar texto español original
                final_text = spanish_text
                
                # Si OpenRouter/Qwen está deshabilitado, mostrar tiempo de corrección como 0
                if hasattr(self, 'correction_start_time'):
                    correction_time = time.time() - self.correction_start_time
                    self.logger.info(f"⏱️ TRADUCCIÓN/IA (deshabilitada) completada en {correction_time:.2f} segundos")
                    self.update_timing_display('correction', correction_time)
                
                self.ui_queue.put({
                    'type': 'translation', 
                    'text': f"[Sin traducción] {spanish_text}"
                })
            
            # Iniciar medición de tiempo de síntesis
            self.synthesis_start_time = time.time()
            
            # Sintetizar voz con texto final usando KokoroHandler optimizado
            self.update_status("Sintetizando voz con Kokoro optimizado...")
            if self.kokoro_handler:
                # Usar KokoroHandler optimizado para mejor rendimiento
                self.kokoro_handler.synthesize_async(
                    final_text, 
                    optimize_for_latency=True
                )
                self.logger.info("🚀 Usando KokoroHandler optimizado para síntesis")
            else:
                # Fallback al TTS Manager si KokoroHandler no está disponible
                self.tts_manager.synthesize_async(final_text)
                self.logger.info("⚠️ Usando TTS Manager como fallback")
            
        except Exception as e:
            self.logger.error(f"Error en traducción y síntesis: {e}")
            # En caso de error, usar texto español original
            self.synthesis_start_time = time.time()
            self.update_status("Sintetizando voz con Kokoro optimizado...")
            if self.kokoro_handler:
                self.kokoro_handler.synthesize_async(
                    spanish_text, 
                    optimize_for_latency=True
                )
            else:
                self.tts_manager.synthesize_async(spanish_text)
            # Calcular tiempo total después de traducción
            if hasattr(self, 'process_start_time'):
                total_time = time.time() - self.process_start_time
                self.logger.info(f"🏁 PROCESO TOTAL completado en {total_time:.3f} segundos")
                self.logger.info("=== FIN DEL PROCESO DE TRADUCCIÓN ===")
                self.update_timing_display('total', total_time)
            self.update_status("Traducción completada")
            self.is_processing = False
    
    def display_translation(self, text: str):
        """Mostrar traducción en la interfaz"""
        self.translated_text.insert(tk.END, f"{text}\n\n")
        self.translated_text.see(tk.END)
        self.play_button.config(state=tk.NORMAL)
    
    # Síntesis y reproducción removidas como solicitado
    
    # === ACCIONES DE INTERFAZ ===
    
    def play_translation(self):
        """Reproducir última traducción"""
        # Obtener último texto traducido
        translated_content = self.translated_text.get("1.0", tk.END).strip()
        
        if not translated_content:
            messagebox.showwarning("Advertencia", "No hay texto traducido para reproducir")
            return
        
        # Obtener última línea no vacía
        lines = [line.strip() for line in translated_content.split('\n') if line.strip()]
        if lines:
            last_translation = lines[-1]
            self.update_status("Sintetizando voz...")
            self.tts_manager.synthesize_async(last_translation)
    
    def clear_text(self):
        """Limpiar textos"""
        self.original_text.delete("1.0", tk.END)
        self.translated_text.delete("1.0", tk.END)
        self.play_button.config(state=tk.DISABLED)
        self.update_status("Textos limpiados")
    
    def save_translation(self):
        """Guardar traducción a archivo"""
        original_content = self.original_text.get("1.0", tk.END).strip()
        translated_content = self.translated_text.get("1.0", tk.END).strip()
        
        if not original_content and not translated_content:
            messagebox.showwarning("Advertencia", "No hay contenido para guardar")
            return
        
        # Seleccionar archivo
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
            title="Guardar traducción"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=== TRADUCTOR DE VOZ EN TIEMPO REAL ===\n")
                    f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("=== TEXTO ORIGINAL (ESPAÑOL) ===\n")
                    f.write(original_content)
                    f.write("\n\n=== TEXTO TRADUCIDO (INGLÉS) ===\n")
                    f.write(translated_content)
                
                self.update_status(f"Guardado en: {filename}")
                messagebox.showinfo("Éxito", f"Traducción guardada en:\n{filename}")
                
            except Exception as e:
                self.logger.error(f"Error guardando archivo: {e}")
                messagebox.showerror("Error", f"No se pudo guardar el archivo:\n{e}")
    
    def play_audio(self):
        """Reproducir el último audio grabado"""
        try:
            if hasattr(self, 'last_audio_file') and self.last_audio_file:
                if os.path.exists(self.last_audio_file):
                    self.audio_manager.play_audio_file(self.last_audio_file)
                    self.update_status("Reproduciendo audio...")
                else:
                    messagebox.showwarning("Advertencia", "No se encontró el archivo de audio")
            else:
                messagebox.showinfo("Información", "No hay audio para reproducir")
        except Exception as e:
            self.logger.error(f"Error reproduciendo audio: {e}")
            messagebox.showerror("Error", f"No se pudo reproducir el audio:\n{e}")
    
    def show_config(self):
        """Mostrar ventana de configuración"""
        ConfigWindow(self.root, self)
    
    def on_closing(self):
        """Manejar cierre de aplicación"""
        try:
            self.logger.info("Cerrando aplicación...")
            
            # Detener grabación si está activa
            if self.is_recording:
                self.stop_recording()
            
            # Limpiar componentes
            if self.audio_manager:
                self.audio_manager.cleanup()
            
            if self.whisper_handler:
                self.whisper_handler.cleanup()
            
            if self.tts_manager:
                self.tts_manager.cleanup()
            
            # if self.translator:
            #     self.translator.cleanup()
            
            # Shutdown thread pool executor
            if hasattr(self, 'processing_executor') and self.processing_executor:
                self.logger.info("Cerrando thread pool executor...")
                self.processing_executor.shutdown(wait=True, cancel_futures=True)
            
            # Guardar configuración
            config.save_config()
            
            self.root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error cerrando aplicación: {e}")
            self.root.destroy()
    
    def run(self):
        """Ejecutar aplicación"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()

class ConfigWindow:
    """Ventana de configuración"""
    
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        
        # Crear ventana
        self.window = tk.Toplevel(parent)
        self.window.title("Configuración")
        self.window.geometry("600x700")
        self.window.resizable(True, True)
        self.window.minsize(600, 700)
        
        # Centrar ventana
        self.window.transient(parent)
        self.window.grab_set()
        
        # Inicializar variables de TTS
        self.tts_engine_var = tk.StringVar(value="kokoro")
        self.tts_fallback_var = tk.StringVar(value="kokoro")
        self.auto_fallback_var = tk.BooleanVar(value=True)
        
        self.create_widgets()
        self.load_current_config()
    
    def create_widgets(self):
        """Crear widgets de configuración"""
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        
        # === TAB WHISPER ===
        whisper_frame = ttk.Frame(notebook)
        notebook.add(whisper_frame, text="Whisper")
        
        # Motor de Whisper
        ttk.Label(whisper_frame, text="Motor:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.engine_var = tk.StringVar()
        engine_combo = ttk.Combobox(whisper_frame, textvariable=self.engine_var, 
                                   values=["whisper", "faster-whisper"], width=47, state="readonly")
        engine_combo.grid(row=0, column=1, padx=5, pady=5)
        engine_combo.bind('<<ComboboxSelected>>', self.on_engine_change)
        
        # Descripción del motor
        self.engine_desc_label = ttk.Label(whisper_frame, text="", wraplength=400, justify=tk.LEFT, foreground='blue')
        self.engine_desc_label.grid(row=1, column=0, columnspan=2, padx=5, pady=(0, 10))
        
        # Modelo
        ttk.Label(whisper_frame, text="Modelo:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(whisper_frame, textvariable=self.model_var, 
                                  values=["tiny", "base", "small", "medium", "large"], width=47)
        model_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Configuración específica de faster-whisper
        self.faster_whisper_frame = ttk.LabelFrame(whisper_frame, text="Configuración Faster-Whisper", padding="5")
        self.faster_whisper_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)
        
        # Dispositivo
        ttk.Label(self.faster_whisper_frame, text="Dispositivo:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.device_var = tk.StringVar()
        device_combo = ttk.Combobox(self.faster_whisper_frame, textvariable=self.device_var, 
                                   values=["cpu", "cuda"], width=20, state="readonly")
        device_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Tipo de cómputo
        ttk.Label(self.faster_whisper_frame, text="Tipo de Cómputo:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.compute_type_var = tk.StringVar()
        compute_combo = ttk.Combobox(self.faster_whisper_frame, textvariable=self.compute_type_var, 
                                    values=["int8", "float16", "float32"], width=20, state="readonly")
        compute_combo.grid(row=0, column=3, padx=5, pady=2)
        
        # Información de rendimiento
        perf_text = "🚀 Faster-Whisper: Hasta 4x más rápido, menor uso de memoria, traducción directa ES→EN"
        perf_label = ttk.Label(self.faster_whisper_frame, text=perf_text, wraplength=400, justify=tk.LEFT, foreground='green')
        perf_label.grid(row=1, column=0, columnspan=4, padx=5, pady=5)
        
        # Inicialmente ocultar configuración de faster-whisper
        self.faster_whisper_frame.grid_remove()
        
        # === TAB QWEN ===
        deepseek_frame = ttk.Frame(notebook)
        notebook.add(deepseek_frame, text="Qwen (OpenRouter)")
        
        # Título y descripción
        title_label = ttk.Label(deepseek_frame, text="Qwen 3 - Corrección de Texto con IA", font=('Arial', 12, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(5, 0))
        
        desc_text = "Usa el modelo GRATUITO Qwen 3 via OpenRouter para corregir estructura y gramática."
        desc_label = ttk.Label(deepseek_frame, text=desc_text, wraplength=450, justify=tk.LEFT, foreground='blue')
        desc_label.grid(row=1, column=0, columnspan=2, padx=5, pady=(0, 10))
        
        # Habilitar Qwen
        self.deepseek_enabled_var = tk.BooleanVar()
        deepseek_check = ttk.Checkbutton(deepseek_frame, text="Habilitar corrección de texto con Qwen 3 (GRATUITO + Fallback de pago)", 
                                        variable=self.deepseek_enabled_var)
        deepseek_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Sistema de claves duales
        self.use_dual_keys_var = tk.BooleanVar()
        dual_keys_check = ttk.Checkbutton(deepseek_frame, text="Usar sistema de claves duales (API gratuita + API de pago)", 
                                         variable=self.use_dual_keys_var, command=self.on_dual_keys_toggle)
        dual_keys_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Instrucciones para API Key
        api_info_text = "🔑 API Key de OpenRouter (GRATUITA):"
        ttk.Label(deepseek_frame, text=api_info_text, font=('Arial', 10, 'bold')).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))
        
        instructions_text = "1. Ve a https://openrouter.ai/ y regístrate gratis\n2. Ve a 'Keys' y crea una nueva API key\n3. Copia tu API key (empieza con 'sk-or-v1-')"
        instructions_label = ttk.Label(deepseek_frame, text=instructions_text, wraplength=450, justify=tk.LEFT, foreground='green')
        instructions_label.grid(row=5, column=0, columnspan=2, padx=5, pady=(0, 5))
        
        # API Key Gratuita
        ttk.Label(deepseek_frame, text="API Key Gratuita:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.deepseek_api_key_free_var = tk.StringVar()
        self.deepseek_api_key_free_entry = ttk.Entry(deepseek_frame, textvariable=self.deepseek_api_key_free_var, width=50, show="*")
        self.deepseek_api_key_free_entry.grid(row=6, column=1, padx=5, pady=5)
        
        # API Key de Pago (solo visible si se habilita sistema dual)
        self.paid_key_label = ttk.Label(deepseek_frame, text="API Key de Pago:")
        self.paid_key_label.grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.deepseek_api_key_paid_var = tk.StringVar()
        self.deepseek_api_key_paid_entry = ttk.Entry(deepseek_frame, textvariable=self.deepseek_api_key_paid_var, width=50, show="*")
        self.deepseek_api_key_paid_entry.grid(row=7, column=1, padx=5, pady=5)
        
        # Inicialmente ocultar campos de API de pago
        self.paid_key_label.grid_remove()
        self.deepseek_api_key_paid_entry.grid_remove()
        
        # Mantener compatibilidad con API key única (legacy)
        self.deepseek_api_key_var = self.deepseek_api_key_free_var  # Alias para compatibilidad
        
        # Modelo Gratuito (siempre visible)
        ttk.Label(deepseek_frame, text="Modelo Gratuito:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.deepseek_model_free_var = tk.StringVar(value="qwen/qwen3-235b-a22b-2507:free")
        deepseek_model_free_entry = ttk.Entry(deepseek_frame, textvariable=self.deepseek_model_free_var, width=50, state="readonly")
        deepseek_model_free_entry.grid(row=8, column=1, padx=5, pady=5)
        
        # Modelo de Pago (solo visible si se habilita sistema dual)
        self.paid_model_label = ttk.Label(deepseek_frame, text="Modelo de Pago:")
        self.paid_model_label.grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        self.deepseek_model_paid_var = tk.StringVar(value="qwen/qwen3-235b-a22b-2507")
        self.deepseek_model_paid_entry = ttk.Entry(deepseek_frame, textvariable=self.deepseek_model_paid_var, width=50, state="readonly")
        self.deepseek_model_paid_entry.grid(row=9, column=1, padx=5, pady=5)
        
        # Inicialmente ocultar campos de modelo de pago
        self.paid_model_label.grid_remove()
        self.deepseek_model_paid_entry.grid_remove()
        
        # Mantener compatibilidad con modelo único (legacy)
        self.deepseek_model_var = self.deepseek_model_free_var  # Alias para compatibilidad
        
        # Información adicional
        info_text = "✨ Qwen 3 corrige SOLO estructura y gramática sin agregar información extra.\n💰 Modelo GRATUITO con fallback automático a modelo de pago si se agotan requests gratuitos."
        info_label = ttk.Label(deepseek_frame, text=info_text, wraplength=450, justify=tk.LEFT, foreground='darkgreen')
        info_label.grid(row=10, column=0, columnspan=2, padx=5, pady=10)
        
        # PYTTSX3 ELIMINADO - Causaba problemas de timeout en Windows SAPI5
        
        # === TAB KOKORO VOICE API ===
        kokoro_frame = ttk.Frame(notebook)
        notebook.add(kokoro_frame, text="Kokoro Voice")
        
        # Motor TTS
        ttk.Label(kokoro_frame, text="Motor TTS:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.tts_engine_var = tk.StringVar(value="kokoro")
        tts_engine_combo = ttk.Combobox(kokoro_frame, textvariable=self.tts_engine_var, values=["kokoro"], state="readonly", width=47)
        tts_engine_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Filtro por género
        ttk.Label(kokoro_frame, text="Género:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.gender_filter_var = tk.StringVar(value="all")
        gender_combo = ttk.Combobox(kokoro_frame, textvariable=self.gender_filter_var, values=["all", "female", "male"], state="readonly", width=47)
        gender_combo.grid(row=1, column=1, padx=5, pady=5)
        gender_combo.bind("<<ComboboxSelected>>", self.on_gender_change)
        
        # Voz
        ttk.Label(kokoro_frame, text="Voz:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.kokoro_voice_var = tk.StringVar(value="af_bella")
        self.voice_combo = ttk.Combobox(kokoro_frame, textvariable=self.kokoro_voice_var, state="readonly", width=47)
        self.voice_combo.grid(row=2, column=1, padx=5, pady=5)
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_change)
        
        # Velocidad
        ttk.Label(kokoro_frame, text="Velocidad:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.kokoro_speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(kokoro_frame, from_=0.5, to=2.0, variable=self.kokoro_speed_var, orient=tk.HORIZONTAL)
        speed_scale.grid(row=3, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.kokoro_speed_label = ttk.Label(kokoro_frame, text="1.00x")
        self.kokoro_speed_label.grid(row=3, column=2, padx=5, pady=5)
        
        speed_scale.config(command=self.on_kokoro_speed_change)
        
        # Servidor
        ttk.Label(kokoro_frame, text="Servidor:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.kokoro_host_var = tk.StringVar(value="localhost")
        host_entry = ttk.Entry(kokoro_frame, textvariable=self.kokoro_host_var, width=25)
        host_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(kokoro_frame, text="Puerto:").grid(row=4, column=1, padx=(200, 5), pady=5, sticky=tk.W)
        self.kokoro_port_var = tk.IntVar(value=5000)
        port_entry = ttk.Entry(kokoro_frame, textvariable=self.kokoro_port_var, width=10)
        port_entry.grid(row=4, column=1, padx=(250, 5), pady=5, sticky=tk.W)
        
        # Información
        info_text = "🎤 Kokoro Voice API - Síntesis de voz neural de alta calidad\n🔊 Voces en inglés americano y británico disponibles"
        info_label = ttk.Label(kokoro_frame, text=info_text, wraplength=450, justify=tk.LEFT, foreground='darkblue')
        info_label.grid(row=5, column=0, columnspan=3, padx=5, pady=10)
        
        # Inicializar lista de voces
        self.update_voice_list()
        
        # === BOTONES ===
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Guardar", command=self.save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=self.window.destroy).pack(side=tk.RIGHT)
    
    
    
    def on_dual_keys_toggle(self):
        """Manejar toggle del sistema de claves duales"""
        if self.use_dual_keys_var.get():
            # Mostrar campos de API de pago
            self.paid_key_label.grid()
            self.deepseek_api_key_paid_entry.grid()
            # Mostrar campos de modelo de pago
            self.paid_model_label.grid()
            self.deepseek_model_paid_entry.grid()
        else:
            # Ocultar campos de API de pago
            self.paid_key_label.grid_remove()
            self.deepseek_api_key_paid_entry.grid_remove()
            # Ocultar campos de modelo de pago
            self.paid_model_label.grid_remove()
            self.deepseek_model_paid_entry.grid_remove()
    
    def on_engine_change(self, event=None):
        """Manejar cambio de motor de Whisper"""
        engine = self.engine_var.get()
        
        if engine == "faster-whisper":
            self.faster_whisper_frame.grid()
            desc_text = "Faster-Whisper: Motor optimizado con mejor rendimiento y menor latencia"
        else:
            self.faster_whisper_frame.grid_remove()
            desc_text = "Whisper Original: Motor estándar de OpenAI con máxima compatibilidad"
        
        self.engine_desc_label.config(text=desc_text)
    
    def on_kokoro_speed_change(self, value):
        """Manejar cambio de velocidad de Kokoro"""
        speed = float(value)
        self.kokoro_speed_label.config(text=f"{speed:.2f}x")
    
    def on_gender_change(self, event=None):
        """Manejar cambio de filtro de género"""
        self.update_voice_list()
    
    def on_voice_change(self, event=None):
        """Manejar cambio de voz seleccionada"""
        voice_display = self.voice_combo.get()
        if ' - ' in voice_display:
            voice_id = voice_display.split(' - ')[0]
            self.kokoro_voice_var.set(voice_id)
            print(f"🎭 CAMBIO DE VOZ: {voice_id}")
            if hasattr(self.main_app, 'logger') and self.main_app.logger:
                self.main_app.logger.info(f"🎭 Voz cambiada a: {voice_id}")
            # Actualizar la voz en el TTS Manager
            if hasattr(self.main_app, 'tts_manager') and self.main_app.tts_manager:
                self.main_app.tts_manager.set_voice(voice_id)
                print(f"🎭 VOZ ENVIADA AL TTS MANAGER: {voice_id}")
    
    def update_voice_list(self):
        """Actualizar lista de voces según el filtro de género"""
        # Definir voces de Kokoro
        kokoro_voices = {
            'af_heart': {'gender': 'female', 'description': 'Heart ❤️ (US Female)'},
            'af_bella': {'gender': 'female', 'description': 'Bella 🔥 (US Female)'},
            'af_nicole': {'gender': 'female', 'description': 'Nicole 🎧 (US Female)'},
            'af_aoede': {'gender': 'female', 'description': 'Aoede (US Female)'},
            'af_kore': {'gender': 'female', 'description': 'Kore (US Female)'},
            'af_sarah': {'gender': 'female', 'description': 'Sarah (US Female)'},
            'af_nova': {'gender': 'female', 'description': 'Nova (US Female)'},
            'af_sky': {'gender': 'female', 'description': 'Sky (US Female)'},
            'af_alloy': {'gender': 'female', 'description': 'Alloy (US Female)'},
            'af_jessica': {'gender': 'female', 'description': 'Jessica (US Female)'},
            'af_river': {'gender': 'female', 'description': 'River (US Female)'},
            'am_michael': {'gender': 'male', 'description': 'Michael (US Male)'},
            'am_fenrir': {'gender': 'male', 'description': 'Fenrir (US Male)'},
            'am_puck': {'gender': 'male', 'description': 'Puck (US Male)'},
            'am_echo': {'gender': 'male', 'description': 'Echo (US Male)'},
            'am_eric': {'gender': 'male', 'description': 'Eric (US Male)'},
            'am_liam': {'gender': 'male', 'description': 'Liam (US Male)'},
            'am_onyx': {'gender': 'male', 'description': 'Onyx (US Male)'},
            'am_santa': {'gender': 'male', 'description': 'Santa (US Male)'},
            'am_adam': {'gender': 'male', 'description': 'Adam (US Male)'},
            'bf_emma': {'gender': 'female', 'description': 'Emma (UK Female)'},
            'bf_isabella': {'gender': 'female', 'description': 'Isabella (UK Female)'},
            'bf_alice': {'gender': 'female', 'description': 'Alice (UK Female)'},
            'bf_lily': {'gender': 'female', 'description': 'Lily (UK Female)'},
            'bm_george': {'gender': 'male', 'description': 'George (UK Male)'},
            'bm_fable': {'gender': 'male', 'description': 'Fable (UK Male)'},
            'bm_lewis': {'gender': 'male', 'description': 'Lewis (UK Male)'},
            'bm_daniel': {'gender': 'male', 'description': 'Daniel (UK Male)'}
        }
        
        # Filtrar voces según género seleccionado
        gender_filter = self.gender_filter_var.get()
        filtered_voices = []
        
        for voice_id, voice_info in kokoro_voices.items():
            if gender_filter == "all" or voice_info['gender'] == gender_filter:
                filtered_voices.append(f"{voice_id} - {voice_info['description']}")
        
        # Actualizar combobox
        self.voice_combo['values'] = filtered_voices
        
        # Mantener selección actual si es válida
        current_voice = self.kokoro_voice_var.get()
        current_display = None
        for voice_display in filtered_voices:
            if voice_display.startswith(current_voice):
                current_display = voice_display
                break
        
        if current_display:
            self.voice_combo.set(current_display)
        elif filtered_voices:
            self.voice_combo.set(filtered_voices[0])
            # Extraer ID de voz del display
            voice_id = filtered_voices[0].split(' - ')[0]
            self.kokoro_voice_var.set(voice_id)
    
    def load_current_config(self):
        """Cargar configuración actual"""
        # Cargar configuración de Kokoro
        kokoro_config = config.get_kokoro_config()
        self.kokoro_voice_var.set(kokoro_config.get('voice', 'af_bella'))
        
        # Cargar velocidad de voz
        speed = kokoro_config.get('speed', 1.0)
        self.kokoro_speed_var.set(speed)
        self.kokoro_speed_label.config(text=f"{speed:.2f}x")
        
        # Cargar configuración del servidor
        self.kokoro_host_var.set(kokoro_config.get('server_host', 'localhost'))
        self.kokoro_port_var.set(kokoro_config.get('server_port', 5000))
        
        # Actualizar lista de voces
        self.update_voice_list()
        
        # Cargar configuración de Whisper
        whisper_config = config.get_whisper_config()
        self.engine_var.set(whisper_config.get('engine', 'whisper'))
        self.model_var.set(whisper_config.get('model_size', 'base'))
        self.device_var.set(whisper_config.get('device', 'cpu'))
        self.compute_type_var.set(whisper_config.get('compute_type', 'int8'))
        
        # Actualizar UI según el motor seleccionado
        self.on_engine_change()
        
        # Cargar configuración de Qwen
        deepseek_config = config.get_deepseek_config()
        self.deepseek_enabled_var.set(deepseek_config.get('enabled', True))  # Habilitado por defecto
        
        # Cargar configuración de claves duales
        use_dual_keys = deepseek_config.get('use_dual_keys', False)
        self.use_dual_keys_var.set(use_dual_keys)
        
        # Cargar API keys
        if use_dual_keys:
            self.deepseek_api_key_free_var.set(deepseek_config.get('api_key_free', ''))
            self.deepseek_api_key_paid_var.set(deepseek_config.get('api_key_paid', ''))
        else:
            # Compatibilidad con configuración antigua
            old_api_key = deepseek_config.get('api_key', '')
            self.deepseek_api_key_free_var.set(old_api_key)
            self.deepseek_api_key_paid_var.set('')
        
        # Cargar modelos
        self.deepseek_model_free_var.set(deepseek_config.get('model_free', 'qwen/qwen3-235b-a22b-2507:free'))
        self.deepseek_model_paid_var.set(deepseek_config.get('model_paid', 'qwen/qwen3-235b-a22b-2507'))
        
        # Mantener compatibilidad con modelo único (legacy)
        if not use_dual_keys:
            legacy_model = deepseek_config.get('model', 'qwen/qwen3-235b-a22b-2507:free')
            self.deepseek_model_free_var.set(legacy_model)
        
        # Actualizar visibilidad de campos
        self.on_dual_keys_toggle()
        
        # === CARGAR CONFIGURACIÓN TTS ===
        tts_config = config.get_tts_config()
        
        # Configuración general de TTS
        self.tts_engine_var.set(tts_config.get('engine', 'elevenlabs'))
        self.tts_fallback_var.set(tts_config.get('fallback_engine', 'elevenlabs'))
        self.auto_fallback_var.set(tts_config.get('auto_fallback', True))
        
        # PYTTSX3 ELIMINADO - Causaba problemas de timeout en Windows SAPI5
        
        
    
    def save_config(self):
        """Guardar configuración"""
        try:
            # Guardar configuración de Kokoro
            # Extraer ID de voz del display seleccionado
            voice_display = self.voice_combo.get()
            voice_id = voice_display.split(' - ')[0] if ' - ' in voice_display else self.kokoro_voice_var.get()
            
            config.set('kokoro', 'voice', voice_id)
            config.set('kokoro', 'speed', self.kokoro_speed_var.get())
            config.set('kokoro', 'server_host', self.kokoro_host_var.get())
            config.set('kokoro', 'server_port', self.kokoro_port_var.get())
            config.set('tts', 'engine', 'kokoro')
            
            
            # Guardar configuración de Whisper
            config.set('whisper', 'engine', self.engine_var.get())
            config.set('whisper', 'model_size', self.model_var.get())
            config.set('whisper', 'device', self.device_var.get())
            config.set('whisper', 'compute_type', self.compute_type_var.get())
            
            # Guardar configuración de Qwen
            config.set('deepseek', 'enabled', self.deepseek_enabled_var.get())
            config.set('deepseek', 'use_dual_keys', self.use_dual_keys_var.get())
            
            if self.use_dual_keys_var.get():
                # Guardar claves duales
                config.set('deepseek', 'api_key_free', self.deepseek_api_key_free_var.get())
                config.set('deepseek', 'api_key_paid', self.deepseek_api_key_paid_var.get())
                # Mantener compatibilidad con api_key antigua
                config.set('deepseek', 'api_key', self.deepseek_api_key_free_var.get())
            else:
                # Modo de clave única (compatibilidad)
                api_key = self.deepseek_api_key_free_var.get()
                config.set('deepseek', 'api_key', api_key)
                config.set('deepseek', 'api_key_free', api_key)
                config.set('deepseek', 'api_key_paid', '')
            
            # Guardar modelos
            config.set('deepseek', 'model_free', self.deepseek_model_free_var.get())
            config.set('deepseek', 'model_paid', self.deepseek_model_paid_var.get())
            
            # Mantener compatibilidad con modelo único (legacy)
            if self.use_dual_keys_var.get():
                config.set('deepseek', 'model', self.deepseek_model_free_var.get())
            else:
                config.set('deepseek', 'model', self.deepseek_model_free_var.get())
            
            # Aplicar configuración al TTS manager si está disponible
            if hasattr(self.main_app, 'tts_manager') and self.main_app.tts_manager:
                # Reinicializar TTS manager con nueva configuración
                self.main_app.tts_manager.reload_config()
                
            # Mantener compatibilidad con elevenlabs_handler
            if hasattr(self.main_app, 'elevenlabs_handler') and self.main_app.elevenlabs_handler:
                if hasattr(self.main_app.elevenlabs_handler, 'set_speech_rate'):
                    self.main_app.elevenlabs_handler.set_speech_rate(self.speech_rate_var.get())
                if hasattr(self.main_app.elevenlabs_handler, 'set_voice'):
                    self.main_app.elevenlabs_handler.set_voice(self.voice_var.get())
            
            # Recargar handler de Whisper si cambió el motor o modelo
            if hasattr(self.main_app, 'whisper_handler') and self.main_app.whisper_handler:
                current_engine = getattr(self.main_app.whisper_handler, 'engine_type', 'whisper')
                current_model = self.main_app.whisper_handler.model_size
                new_engine = self.engine_var.get()
                new_model = self.model_var.get()
                
                # Si cambió el motor, reinicializar completamente
                if current_engine != new_engine:
                    self.main_app.update_status(f"Cambiando motor: {current_engine} → {new_engine}...")
                    # Limpiar handler actual
                    self.main_app.whisper_handler.cleanup()
                    # Crear nuevo handler
                    self.main_app.whisper_handler = self.main_app._create_whisper_handler()
                    self.main_app.whisper_handler.set_callbacks(
                        on_transcription=self.main_app.on_transcription_ready,
                        on_error=self.main_app.on_whisper_error
                    )
                    # Cargar modelo
                    if self.main_app.whisper_handler.load_model():
                        self.main_app.update_status(f"Motor cambiado a '{new_engine}' exitosamente")
                    else:
                        self.main_app.update_status(f"Error cargando motor '{new_engine}'")
                # Si solo cambió el modelo, recargar
                elif current_model != new_model:
                    self.main_app.update_status(f"Recargando modelo: {current_model} → {new_model}...")
                    if self.main_app.whisper_handler.load_model(new_model):
                        self.main_app.update_status(f"Modelo cambiado a '{new_model}' exitosamente")
                    else:
                        self.main_app.update_status(f"Error cargando modelo '{new_model}'")
            
            # Reinicializar OpenRouter handler con nueva configuración
            if hasattr(self.main_app, 'deepseek_handler'):
                from deepseek_handler import DeepSeekHandler
                self.main_app.deepseek_handler = DeepSeekHandler()
            
            # === GUARDAR CONFIGURACIÓN TTS ===
            # Configuración general de TTS
            
            
            # PYTTSX3 ELIMINADO - Causaba problemas de timeout en Windows SAPI5
            
            
            
            # Guardar archivo
            config.save_config()
            
            messagebox.showinfo("Éxito", "Configuración guardada correctamente")
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error guardando configuración:\n{e}")
    
    

def main():
    """Función principal"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('voice_translator.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Iniciando Traductor de Voz en Tiempo Real")
    
    try:
        # Crear y ejecutar aplicación
        app = VoiceTranslatorGUI()
        app.run()
        
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        messagebox.showerror("Error Fatal", f"Error fatal en la aplicación:\n{e}")
    
    logger.info("Aplicación cerrada")

if __name__ == "__main__":
    main()