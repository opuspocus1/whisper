# -*- coding: utf-8 -*-
"""
Traductor de español a inglés
Soporte para múltiples servicios de traducción
"""

import requests
import time
import threading
import queue
from typing import Optional, Dict, Any, Callable, List
import logging
import json
from pathlib import Path

# Importar traductores disponibles
try:
    from deep_translator import GoogleTranslator, MicrosoftTranslator, LibreTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    GoogleTranslator = None
    MicrosoftTranslator = None
    LibreTranslator = None

try:
    from googletrans import Translator as GoogleTransAPI
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    GoogleTransAPI = None

from config import config

class Translator:
    """Traductor de español a inglés con múltiples servicios"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.translation_config = config.get_translation_config()
        
        # Configuración
        self.source_language = self.translation_config.get('source_language', 'es')
        self.target_language = self.translation_config.get('target_language', 'en')
        self.fallback_translator = self.translation_config.get('fallback_translator', 'google')
        
        # Servicios de traducción disponibles
        self.translators = {}
        self.current_translator = None
        
        # Estados
        self.is_available = False
        self.is_processing = False
        
        # Cola de procesamiento
        self.translation_queue = queue.Queue()
        self.translation_thread = None
        
        # Callbacks
        self.on_translation_ready: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_progress: Optional[Callable] = None
        
        # Cache de traducciones
        self.translation_cache = {}
        self.cache_enabled = config.get('performance', 'cache_enabled', True)
        
        # Estadísticas
        self.stats = {
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'total_characters': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'translator_usage': {}
        }
        
        # Inicializar traductores
        self._initialize_translators()
        
        self.logger.info(f"Translator inicializado - Disponible: {self.is_available}")
    
    def _initialize_translators(self):
        """Inicializar servicios de traducción"""
        # Google Translator (deep-translator)
        if DEEP_TRANSLATOR_AVAILABLE:
            try:
                self.translators['google_deep'] = GoogleTranslator(
                    source=self.source_language,
                    target=self.target_language
                )
                self.logger.info("Google Translator (deep-translator) inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando Google Translator (deep): {e}")
        
        # Google Translate API (googletrans)
        if GOOGLETRANS_AVAILABLE:
            try:
                self.translators['google_api'] = GoogleTransAPI()
                self.logger.info("Google Translate API (googletrans) inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando Google Translate API: {e}")
        
        # Microsoft Translator (deep-translator)
        if DEEP_TRANSLATOR_AVAILABLE:
            try:
                self.translators['microsoft'] = MicrosoftTranslator(
                    source=self.source_language,
                    target=self.target_language
                )
                self.logger.info("Microsoft Translator inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando Microsoft Translator: {e}")
        
        # LibreTranslate (deep-translator)
        if DEEP_TRANSLATOR_AVAILABLE:
            try:
                self.translators['libre'] = LibreTranslator(
                    source=self.source_language,
                    target=self.target_language,
                    base_url="https://libretranslate.de"
                )
                self.logger.info("LibreTranslate inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando LibreTranslate: {e}")
        
        # Traductor simple de fallback
        self.translators['simple'] = SimpleTranslator()
        self.logger.info("Traductor simple de fallback inicializado")
        
        # Establecer traductor por defecto
        if self.translators:
            # Prioridad: google_deep > google_api > microsoft > libre > simple
            for preferred in ['google_deep', 'google_api', 'microsoft', 'libre', 'simple']:
                if preferred in self.translators:
                    self.current_translator = preferred
                    self.is_available = True
                    break
            
            self.logger.info(f"Traductor activo: {self.current_translator}")
            self.logger.info(f"Traductores disponibles: {list(self.translators.keys())}")
        else:
            self.logger.error("No hay traductores disponibles")
            self.is_available = False
    
    def set_translator(self, translator_name: str) -> bool:
        """Establecer traductor activo"""
        if translator_name in self.translators:
            self.current_translator = translator_name
            self.logger.info(f"Traductor cambiado a: {translator_name}")
            return True
        else:
            self.logger.warning(f"Traductor no disponible: {translator_name}")
            return False
    
    def get_available_translators(self) -> List[str]:
        """Obtener lista de traductores disponibles"""
        return list(self.translators.keys())
    
    def set_languages(self, source: str, target: str):
        """Establecer idiomas de origen y destino"""
        self.source_language = source
        self.target_language = target
        
        config.set('translation', 'source_language', source)
        config.set('translation', 'target_language', target)
        
        # Reinicializar traductores con nuevos idiomas
        self._initialize_translators()
        
        self.logger.info(f"Idiomas establecidos: {source} -> {target}")
    
    def start_translation_worker(self):
        """Iniciar worker de traducción"""
        if self.translation_thread and self.translation_thread.is_alive():
            return
        
        self.is_processing = True
        self.translation_thread = threading.Thread(target=self._translation_worker, daemon=True)
        self.translation_thread.start()
        self.logger.info("Worker de traducción iniciado")
    
    def _translation_worker(self):
        """Worker para traducción"""
        while self.is_processing:
            try:
                # Obtener tarea de la cola
                task = self.translation_queue.get(timeout=1.0)
                
                if task is None:  # Señal de parada
                    break
                
                text, options, callback = task
                
                # Traducir texto
                result = self._translate_text(text, options)
                
                # Llamar callback con resultado
                if callback:
                    callback(result)
                elif self.on_translation_ready:
                    self.on_translation_ready(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker de traducción: {e}")
                if self.on_error:
                    self.on_error(f"Error de traducción: {e}")
    
    def translate_async(self, 
                       text: str,
                       callback: Optional[Callable] = None,
                       **options) -> bool:
        """Traducir texto de forma asíncrona"""
        if not self.is_available:
            self.logger.error("Traductor no disponible")
            return False
        
        if not text.strip():
            self.logger.warning("Texto vacío")
            return False
        
        try:
            # Iniciar worker si no está activo
            if not self.is_processing:
                self.start_translation_worker()
            
            # Agregar a la cola de traducción
            task = (text, options, callback)
            self.translation_queue.put_nowait(task)
            return True
            
        except queue.Full:
            self.logger.warning("Cola de traducción llena")
            return False
        except Exception as e:
            self.logger.error(f"Error agregando tarea de traducción: {e}")
            return False
    
    def translate_sync(self, text: str, **options) -> Dict[str, Any]:
        """Traducir texto de forma síncrona"""
        if not self.is_available:
            raise RuntimeError("Traductor no disponible")
        
        return self._translate_text(text, options)
    
    def _translate_text(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Traducir texto usando el traductor activo"""
        start_time = time.time()
        
        try:
            # Verificar cache
            cache_key = None
            if self.cache_enabled:
                cache_key = self._get_cache_key(text, options)
                if cache_key in self.translation_cache:
                    self.stats['cache_hits'] += 1
                    self.logger.debug("Traducción obtenida del cache")
                    return self.translation_cache[cache_key]
            
            # Obtener traductor a usar
            translator_name = options.get('translator', self.current_translator)
            
            if translator_name not in self.translators:
                translator_name = self.current_translator
            
            translator = self.translators[translator_name]
            
            # Notificar progreso
            if self.on_progress:
                self.on_progress("Traduciendo texto...")
            
            # Realizar traducción
            translated_text = self._perform_translation(translator, translator_name, text, options)
            
            # Preparar resultado
            result = {
                'original_text': text,
                'translated_text': translated_text,
                'source_language': self.source_language,
                'target_language': self.target_language,
                'translator_used': translator_name,
                'processing_time': time.time() - start_time,
                'success': True,
                'error': None
            }
            
            # Guardar en cache
            if self.cache_enabled and cache_key:
                self.translation_cache[cache_key] = result
                self._cleanup_cache()
            
            # Actualizar estadísticas
            self.stats['total_translations'] += 1
            self.stats['successful_translations'] += 1
            self.stats['total_characters'] += len(text)
            
            if translator_name not in self.stats['translator_usage']:
                self.stats['translator_usage'][translator_name] = 0
            self.stats['translator_usage'][translator_name] += 1
            
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            self.logger.info(f"Traducción completada en {processing_time:.2f}s usando {translator_name}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error en traducción: {e}")
            
            # Intentar con traductor de fallback
            if translator_name != self.fallback_translator and self.fallback_translator in self.translators:
                self.logger.info(f"Intentando con traductor de fallback: {self.fallback_translator}")
                try:
                    fallback_translator = self.translators[self.fallback_translator]
                    translated_text = self._perform_translation(
                        fallback_translator, 
                        self.fallback_translator, 
                        text, 
                        options
                    )
                    
                    result = {
                        'original_text': text,
                        'translated_text': translated_text,
                        'source_language': self.source_language,
                        'target_language': self.target_language,
                        'translator_used': self.fallback_translator,
                        'processing_time': time.time() - start_time,
                        'success': True,
                        'error': f"Traductor principal falló, usado fallback: {str(e)}"
                    }
                    
                    self.stats['total_translations'] += 1
                    self.stats['successful_translations'] += 1
                    
                    return result
                    
                except Exception as fallback_error:
                    self.logger.error(f"Error en traductor de fallback: {fallback_error}")
            
            # Error total
            self.stats['total_translations'] += 1
            self.stats['failed_translations'] += 1
            
            if self.on_error:
                self.on_error(f"Error de traducción: {e}")
            
            return {
                'original_text': text,
                'translated_text': text,  # Devolver texto original como fallback
                'source_language': self.source_language,
                'target_language': self.target_language,
                'translator_used': translator_name,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _perform_translation(self, translator, translator_name: str, text: str, options: Dict[str, Any]) -> str:
        """Realizar traducción con el traductor especificado"""
        if translator_name == 'google_deep':
            return translator.translate(text)
        
        elif translator_name == 'google_api':
            result = translator.translate(text, src=self.source_language, dest=self.target_language)
            return result.text
        
        elif translator_name == 'microsoft':
            return translator.translate(text)
        
        elif translator_name == 'libre':
            return translator.translate(text)
        
        elif translator_name == 'simple':
            return translator.translate(text, self.source_language, self.target_language)
        
        else:
            raise ValueError(f"Traductor desconocido: {translator_name}")
    
    def _get_cache_key(self, text: str, options: Dict[str, Any]) -> str:
        """Generar clave de cache"""
        import hashlib
        
        # Crear string único con texto y opciones relevantes
        cache_data = {
            'text': text,
            'source': self.source_language,
            'target': self.target_language,
            'translator': options.get('translator', self.current_translator)
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Limpiar cache si es muy grande"""
        max_cache_size = 100  # máximo número de entradas
        
        if len(self.translation_cache) > max_cache_size:
            # Remover las entradas más antiguas
            items_to_remove = len(self.translation_cache) - max_cache_size + 20
            keys_to_remove = list(self.translation_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.translation_cache[key]
            
            self.logger.debug(f"Cache de traducción limpiado: {items_to_remove} entradas removidas")
    
    def _update_stats(self, processing_time: float):
        """Actualizar estadísticas"""
        # Calcular tiempo promedio de respuesta
        if self.stats['successful_translations'] > 0:
            total_time = (self.stats['average_response_time'] * (self.stats['successful_translations'] - 1) + 
                         processing_time) / self.stats['successful_translations']
            self.stats['average_response_time'] = total_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de uso"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Limpiar cache de traducciones"""
        self.translation_cache.clear()
        self.logger.info("Cache de traducción limpiado")
    
    def set_callbacks(self, 
                     on_translation_ready: Optional[Callable] = None,
                     on_error: Optional[Callable] = None,
                     on_progress: Optional[Callable] = None):
        """Establecer callbacks"""
        if on_translation_ready:
            self.on_translation_ready = on_translation_ready
        if on_error:
            self.on_error = on_error
        if on_progress:
            self.on_progress = on_progress
    
    def stop_translation(self):
        """Detener traducción"""
        self.is_processing = False
        
        # Enviar señal de parada
        try:
            self.translation_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Esperar a que termine el thread
        if self.translation_thread and self.translation_thread.is_alive():
            self.translation_thread.join(timeout=2.0)
        
        self.logger.info("Traducción detenida")
    
    def cleanup(self):
        """Limpiar recursos"""
        self.logger.info("Limpiando Translator...")
        
        self.stop_translation()
        self.clear_cache()
        
        self.logger.info("Translator limpiado")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

class SimpleTranslator:
    """Traductor simple de fallback con diccionario básico"""
    
    def __init__(self):
        # Diccionario básico español-inglés
        self.dictionary = {
            # Saludos y cortesía
            'hola': 'hello',
            'adiós': 'goodbye',
            'gracias': 'thank you',
            'por favor': 'please',
            'disculpe': 'excuse me',
            'perdón': 'sorry',
            'de nada': 'you\'re welcome',
            
            # Palabras comunes
            'sí': 'yes',
            'no': 'no',
            'tal vez': 'maybe',
            'quizás': 'perhaps',
            'siempre': 'always',
            'nunca': 'never',
            'aquí': 'here',
            'allí': 'there',
            'ahora': 'now',
            'después': 'later',
            'antes': 'before',
            'hoy': 'today',
            'mañana': 'tomorrow',
            'ayer': 'yesterday',
            
            # Números
            'uno': 'one',
            'dos': 'two',
            'tres': 'three',
            'cuatro': 'four',
            'cinco': 'five',
            'seis': 'six',
            'siete': 'seven',
            'ocho': 'eight',
            'nueve': 'nine',
            'diez': 'ten',
            
            # Familia
            'familia': 'family',
            'padre': 'father',
            'madre': 'mother',
            'hijo': 'son',
            'hija': 'daughter',
            'hermano': 'brother',
            'hermana': 'sister',
            
            # Colores
            'rojo': 'red',
            'azul': 'blue',
            'verde': 'green',
            'amarillo': 'yellow',
            'negro': 'black',
            'blanco': 'white',
            
            # Verbos comunes
            'ser': 'to be',
            'estar': 'to be',
            'tener': 'to have',
            'hacer': 'to do',
            'ir': 'to go',
            'venir': 'to come',
            'ver': 'to see',
            'hablar': 'to speak',
            'comer': 'to eat',
            'beber': 'to drink',
            
            # Preguntas
            'qué': 'what',
            'quién': 'who',
            'cuándo': 'when',
            'dónde': 'where',
            'por qué': 'why',
            'cómo': 'how',
            'cuánto': 'how much',
            'cuántos': 'how many'
        }
    
    def translate(self, text: str, source_lang: str = 'es', target_lang: str = 'en') -> str:
        """Traducir usando diccionario simple"""
        if source_lang != 'es' or target_lang != 'en':
            return text  # Solo soporta español a inglés
        
        # Convertir a minúsculas para búsqueda
        text_lower = text.lower().strip()
        
        # Buscar traducción exacta
        if text_lower in self.dictionary:
            return self.dictionary[text_lower]
        
        # Buscar palabras individuales
        words = text_lower.split()
        translated_words = []
        
        for word in words:
            # Limpiar puntuación
            clean_word = word.strip('.,!?;:')
            
            if clean_word in self.dictionary:
                translated_words.append(self.dictionary[clean_word])
            else:
                translated_words.append(word)  # Mantener palabra original
        
        return ' '.join(translated_words)

if __name__ == "__main__":
    # Prueba del Translator
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    # Crear traductor
    translator = Translator()
    
    if translator.is_available:
        print("Traductor disponible")
        print(f"Traductores disponibles: {translator.get_available_translators()}")
        print(f"Traductor activo: {translator.current_translator}")
        
        # Pruebas de traducción
        test_texts = [
            "Hola, ¿cómo estás?",
            "Me llamo Juan y soy de España.",
            "¿Puedes ayudarme con esto?",
            "Gracias por tu ayuda.",
            "Hasta luego."
        ]
        
        for text in test_texts:
            print(f"\nTraduciendo: '{text}'")
            
            result = translator.translate_sync(text)
            
            if result['success']:
                print(f"Resultado: '{result['translated_text']}'")
                print(f"Traductor usado: {result['translator_used']}")
                print(f"Tiempo: {result['processing_time']:.2f}s")
            else:
                print(f"Error: {result['error']}")
        
        # Mostrar estadísticas
        stats = translator.get_stats()
        print(f"\nEstadísticas: {stats}")
    
    else:
        print("Traductor no disponible")
    
    # Limpiar
    translator.cleanup()
    print("Prueba completada")