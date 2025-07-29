#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo de las funcionalidades de TTS
Muestra cómo usar ElevenLabs y pyttsx3
"""

import sys
import logging
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from tts_manager import TTSManager
from config import config

def main():
    """Función principal de demostración"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== DEMO TTS - Alternativas Gratuitas a ElevenLabs ===")
    
    # Crear TTS Manager
    tts_manager = TTSManager()
    
    # Mostrar motores disponibles
    logger.info(f"Motores TTS disponibles: {tts_manager.available_engines}")
    logger.info(f"Motor principal: {tts_manager.current_engine}")
    logger.info(f"Motor de respaldo: {tts_manager.fallback_engine}")
    
    # Texto de prueba
    test_texts = [
        "Hola, este es un test del sistema de síntesis de voz.",
        "Hello, this is a test of the text-to-speech system.",
        "Bonjour, ceci est un test du système de synthèse vocale."
    ]
    
    # Probar cada motor disponible
    for engine in tts_manager.available_engines:
        logger.info(f"\n=== Probando motor: {engine.upper()} ===")
        
        # Cambiar motor temporalmente
        original_engine = tts_manager.current_engine
        tts_manager.set_engine(engine)
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"Sintetizando texto {i}: '{text[:50]}...'")
            
            try:
                result = tts_manager.synthesize_sync(text)
                
                if result['success']:
                    logger.info(f"✅ Síntesis exitosa - {len(result['audio_data'])} muestras")
                    logger.info(f"   Tiempo: {result['processing_time']:.2f}s")
                    logger.info(f"   Sample rate: {result['sample_rate']}Hz")
                else:
                    logger.error(f"❌ Error: {result['error']}")
                    
            except Exception as e:
                logger.error(f"❌ Excepción: {e}")
        
        # Restaurar motor original
        tts_manager.set_engine(original_engine)
    
    # Mostrar estadísticas
    logger.info("\n=== ESTADÍSTICAS FINALES ===")
    stats = tts_manager.get_stats()
    for key, value in stats.items():
        if not key.endswith('_stats'):
            logger.info(f"{key}: {value}")
    
    # Limpiar recursos
    tts_manager.cleanup()
    logger.info("\n=== DEMO COMPLETADA ===")

if __name__ == "__main__":
    main()