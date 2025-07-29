#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba final del flujo completo de la GUI
Simula exactamente el proceso de síntesis de voz como lo hace la GUI
"""

import os
import sys
import time
import logging
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tts_manager import TTSManager

def test_gui_flow():
    """Probar el flujo completo de síntesis como en la GUI"""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 INICIANDO PRUEBA FINAL DEL FLUJO GUI")
    
    # Variables para almacenar resultado
    synthesis_result = None
    synthesis_error = None
    
    def on_synthesis_ready(result):
        """Simular callback de la GUI"""
        nonlocal synthesis_result
        synthesis_result = result
        
        logger.info(f"✅ Callback on_synthesis_ready recibido")
        logger.info(f"📊 Resultado: success={result.get('success', False)}")
        
        if result.get('success'):
            audio_file = result.get('audio_file')
            audio_data = result.get('audio_data')
            sample_rate = result.get('sample_rate')
            
            logger.info(f"📁 audio_file: {audio_file}")
            logger.info(f"🎵 audio_data: {type(audio_data)} (length: {len(audio_data) if audio_data is not None else 'None'})")
            logger.info(f"🔊 sample_rate: {sample_rate}")
            
            # Simular lógica de play_audio de la GUI
            if audio_file and os.path.exists(audio_file):
                logger.info(f"✅ Archivo de audio existe y se puede reproducir: {audio_file}")
                logger.info(f"📊 Tamaño del archivo: {os.path.getsize(audio_file)} bytes")
                return True
            elif audio_data is not None and len(audio_data) > 0:
                logger.info(f"✅ Datos de audio en memoria disponibles para reproducir")
                return True
            else:
                logger.warning("⚠️ No hay datos de audio ni archivo para reproducir")
                return False
        else:
            logger.error(f"❌ Síntesis falló: {result.get('error')}")
            return False
    
    def on_synthesis_error(error):
        """Simular callback de error de la GUI"""
        nonlocal synthesis_error
        synthesis_error = error
        logger.error(f"❌ Error de síntesis: {error}")
    
    # Crear TTSManager como en la GUI
    logger.info("🔧 Inicializando TTSManager...")
    tts_manager = TTSManager()
    
    # Configurar callbacks como en la GUI
    tts_manager.on_audio_ready = on_synthesis_ready
    tts_manager.on_error = on_synthesis_error
    
    # Texto de prueba
    test_text = "Hola, esta es una prueba final del sistema de síntesis de voz."
    logger.info(f"🎤 Sintetizando texto: '{test_text}'")
    
    # Iniciar síntesis asíncrona como en la GUI
    start_time = time.time()
    tts_manager.synthesize_async(test_text)
    
    # Esperar resultado con timeout
    timeout = 15
    while synthesis_result is None and synthesis_error is None and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    synthesis_time = time.time() - start_time
    logger.info(f"⏱️ Tiempo total de síntesis: {synthesis_time:.2f} segundos")
    
    # Verificar resultado
    if synthesis_error:
        logger.error(f"❌ PRUEBA FALLIDA: Error de síntesis: {synthesis_error}")
        return False
    
    if synthesis_result is None:
        logger.error(f"❌ PRUEBA FALLIDA: Timeout esperando resultado")
        return False
    
    if not synthesis_result.get('success'):
        logger.error(f"❌ PRUEBA FALLIDA: Síntesis no exitosa: {synthesis_result.get('error')}")
        return False
    
    # Verificar que se puede reproducir
    audio_file = synthesis_result.get('audio_file')
    audio_data = synthesis_result.get('audio_data')
    
    can_play = False
    
    if audio_file and os.path.exists(audio_file):
        logger.info(f"✅ Archivo de audio disponible: {audio_file}")
        logger.info(f"📊 Tamaño: {os.path.getsize(audio_file)} bytes")
        can_play = True
    
    if audio_data is not None and len(audio_data) > 0:
        logger.info(f"✅ Datos de audio en memoria disponibles: {len(audio_data)} elementos")
        can_play = True
    
    if not can_play:
        logger.error(f"❌ PRUEBA FALLIDA: No hay audio disponible para reproducir")
        return False
    
    # Simular tiempo de procesamiento de la GUI
    logger.info("⏳ Simulando procesamiento de GUI...")
    time.sleep(1)
    
    # Verificar que el archivo sigue disponible
    if audio_file and not os.path.exists(audio_file):
        logger.error(f"❌ PRUEBA FALLIDA: Archivo de audio fue eliminado prematuramente")
        return False
    
    # Mostrar estadísticas
    current_engine = tts_manager.get_current_engine()
    logger.info(f"🔧 Motor TTS actual: {current_engine}")
    
    stats = tts_manager.get_stats()
    logger.info(f"📈 Estadísticas del TTS Manager: {stats}")
    
    logger.info("🎉 PRUEBA FINAL EXITOSA - El sistema de síntesis funciona correctamente")
    return True

def main():
    """Función principal"""
    try:
        success = test_gui_flow()
        
        print("\n" + "="*60)
        if success:
            print("✅ PRUEBA FINAL EXITOSA")
            print("🎉 El problema de síntesis de voz está RESUELTO")
            print("📱 La aplicación GUI debería funcionar correctamente")
        else:
            print("❌ PRUEBA FINAL FALLIDA")
            print("🔧 Revisar logs para identificar problemas restantes")
        print("="*60)
            
    except Exception as e:
        print(f"\n💥 ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()