#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar la configuración auto_fallback
"""

import logging
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
from config import config

def test_auto_fallback_disabled():
    """
    Probar que cuando auto_fallback está desactivado, 
    no se usa el motor de respaldo
    """
    logger = logging.getLogger(__name__)
    logger.info("=== PRUEBA AUTO_FALLBACK DESACTIVADO ===")
    
    # Verificar configuración actual
    tts_config = config.get_tts_config()
    logger.info(f"Configuración TTS actual:")
    logger.info(f"  Motor principal: {tts_config.get('engine', 'N/A')}")
    logger.info(f"  Motor de respaldo: {tts_config.get('fallback_engine', 'N/A')}")
    logger.info(f"  Auto fallback: {tts_config.get('auto_fallback', 'N/A')}")
    
    # Crear TTSManager
    tts_manager = TTSManager()
    
    logger.info(f"TTSManager configurado:")
    logger.info(f"  Motor actual: {tts_manager.current_engine}")
    logger.info(f"  Motor de respaldo: {tts_manager.fallback_engine}")
    logger.info(f"  Auto fallback: {tts_manager.auto_fallback}")
    logger.info(f"  Motores disponibles: {tts_manager.available_engines}")
    
    # Texto de prueba que debería funcionar
    test_text_1 = "Hello, this is a test."
    logger.info(f"\nPrueba 1 - Texto normal: '{test_text_1}'")
    
    result_1 = tts_manager.synthesize_sync(test_text_1)
    logger.info(f"Resultado 1:")
    logger.info(f"  Éxito: {result_1['success']}")
    logger.info(f"  Motor usado: {result_1.get('engine', 'N/A')}")
    if not result_1['success']:
        logger.info(f"  Error: {result_1.get('error', 'N/A')}")
    
    # Segundo texto de prueba (puede fallar con pyttsx3)
    test_text_2 = "This is another test to see if pyttsx3 works consistently."
    logger.info(f"\nPrueba 2 - Segundo texto: '{test_text_2}'")
    
    result_2 = tts_manager.synthesize_sync(test_text_2)
    logger.info(f"Resultado 2:")
    logger.info(f"  Éxito: {result_2['success']}")
    logger.info(f"  Motor usado: {result_2.get('engine', 'N/A')}")
    if not result_2['success']:
        logger.info(f"  Error: {result_2.get('error', 'N/A')}")
    
    # Mostrar estadísticas
    stats = tts_manager.get_stats()
    logger.info(f"\nEstadísticas finales:")
    logger.info(f"  Total de solicitudes: {stats['total_requests']}")
    logger.info(f"  Solicitudes exitosas: {stats['successful_requests']}")
    logger.info(f"  Solicitudes fallidas: {stats['failed_requests']}")
    logger.info(f"  Usos de fallback: {stats['fallback_uses']}")
    
    # Verificar que no se usó fallback si auto_fallback está desactivado
    if not tts_manager.auto_fallback:
        if stats['fallback_uses'] > 0:
            logger.error("❌ ERROR: Se usó fallback cuando auto_fallback está desactivado")
            return False
        else:
            logger.info("✅ CORRECTO: No se usó fallback cuando auto_fallback está desactivado")
    
    # Limpiar
    tts_manager.cleanup()
    
    return True

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=== INICIANDO PRUEBA AUTO_FALLBACK ===")
    
    success = test_auto_fallback_disabled()
    
    if success:
        logger.info("🎉 PRUEBA EXITOSA: auto_fallback funciona correctamente")
    else:
        logger.error("💥 PRUEBA FALLIDA: auto_fallback no funciona correctamente")
    
    logger.info("=== PRUEBA COMPLETADA ===")