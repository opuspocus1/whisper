#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar específicamente el callback de la GUI
"""

import logging
import time
import sys
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tts_manager import TTSManager

def test_gui_callback():
    """Simular el callback de la GUI"""
    logger = logging.getLogger(__name__)
    
    # Variables para capturar el resultado
    callback_received = False
    synthesis_result = None
    error_received = None
    
    def on_synthesis_ready(result):
        nonlocal callback_received, synthesis_result
        callback_received = True
        synthesis_result = result
        logger.info("=== CALLBACK RECIBIDO EN GUI ===")
        logger.info(f"Resultado completo: {result}")
        logger.info(f"Tipo de audio_data: {type(result.get('audio_data'))}")
        logger.info(f"Longitud de audio_data: {len(result.get('audio_data', []))}")
        logger.info(f"audio_file: {result.get('audio_file')}")
        logger.info(f"sample_rate: {result.get('sample_rate')}")
        logger.info(f"success: {result.get('success')}")
        logger.info(f"engine: {result.get('engine')}")
        
        # Verificar si el archivo existe
        audio_file = result.get('audio_file')
        if audio_file and os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file)
            logger.info(f"Archivo de audio existe: {audio_file} (tamaño: {file_size} bytes)")
        else:
            logger.warning(f"Archivo de audio NO existe: {audio_file}")
    
    def on_synthesis_error(error):
        nonlocal error_received
        error_received = error
        logger.error(f"Error en síntesis: {error}")
    
    # Crear TTSManager y configurar callbacks
    logger.info("Creando TTSManager...")
    tts_manager = TTSManager()
    tts_manager.set_callbacks(
        on_audio_ready=on_synthesis_ready,
        on_error=on_synthesis_error
    )
    
    logger.info(f"Motores disponibles: {tts_manager.get_available_engines()}")
    logger.info(f"Motor actual: {tts_manager.get_current_engine()}")
    
    # Realizar síntesis asíncrona
    test_text = "Hola, esta es una prueba del callback de la GUI."
    logger.info(f"Iniciando síntesis asíncrona: '{test_text}'")
    
    tts_manager.synthesize_async(test_text)
    
    # Esperar a que se complete
    max_wait = 10  # segundos
    wait_time = 0
    while not callback_received and not error_received and wait_time < max_wait:
        time.sleep(0.1)
        wait_time += 0.1
    
    # Resultados
    logger.info("=== RESULTADOS DE LA PRUEBA ===")
    logger.info(f"Callback recibido: {callback_received}")
    logger.info(f"Error recibido: {error_received}")
    logger.info(f"Tiempo de espera: {wait_time:.1f}s")
    
    if callback_received:
        logger.info("✅ ÉXITO: El callback funciona correctamente")
        return True
    elif error_received:
        logger.error(f"❌ ERROR: {error_received}")
        return False
    else:
        logger.error("❌ TIMEOUT: No se recibió callback ni error")
        return False
    
    # Limpiar
    tts_manager.cleanup()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=== INICIANDO PRUEBA DE CALLBACK GUI ===")
    
    success = test_gui_callback()
    
    if success:
        logger.info("🎉 PRUEBA EXITOSA: El problema está solucionado")
    else:
        logger.error("💥 PRUEBA FALLIDA: El problema persiste")
    
    logger.info("=== PRUEBA COMPLETADA ===")