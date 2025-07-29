#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de depuración para TTS Manager
"""

import logging
import time
from tts_manager import TTSManager

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_callback_received(result):
    """Callback para cuando se recibe audio"""
    logger.info(f"CALLBACK RECIBIDO: {result}")
    logger.info(f"Success: {result.get('success', 'N/A')}")
    logger.info(f"Audio file: {result.get('audio_file', 'N/A')}")
    logger.info(f"Audio data type: {type(result.get('audio_data', None))}")
    if result.get('audio_data') is not None:
        logger.info(f"Audio data length: {len(result.get('audio_data', []))}")
    logger.info(f"Sample rate: {result.get('sample_rate', 'N/A')}")
    logger.info(f"Engine: {result.get('engine', 'N/A')}")

def test_error_callback(error):
    """Callback para errores"""
    logger.error(f"ERROR CALLBACK: {error}")

def main():
    logger.info("=== INICIANDO TEST DE TTS MANAGER ===")
    
    # Crear TTS Manager
    tts_manager = TTSManager()
    
    # Configurar callbacks
    tts_manager.set_callbacks(
        on_audio_ready=test_callback_received,
        on_error=test_error_callback
    )
    
    # Obtener motores disponibles
    engines = tts_manager.get_available_engines()
    logger.info(f"Motores disponibles: {engines}")
    
    # Configurar motor actual
    current_engine = tts_manager.get_current_engine()
    logger.info(f"Motor actual: {current_engine}")
    
    # Test de síntesis
    test_text = "Hola, esta es una prueba de síntesis de voz."
    logger.info(f"Sintetizando: '{test_text}'")
    
    # Síntesis asíncrona
    tts_manager.synthesize_async(test_text)
    
    # Esperar un poco para que se complete
    logger.info("Esperando resultado...")
    time.sleep(5)
    
    # Mostrar estadísticas
    stats = tts_manager.get_stats()
    logger.info(f"Estadísticas: {stats}")
    
    # Limpiar
    tts_manager.cleanup()
    logger.info("=== TEST COMPLETADO ===")

if __name__ == "__main__":
    main()