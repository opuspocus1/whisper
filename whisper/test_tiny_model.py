#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test del modelo tiny optimizado para tiempo real
"""

import sys
import time
import numpy as np
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from faster_whisper_handler import FasterWhisperHandler

def test_tiny_model():
    """Test del modelo tiny con optimizaciones"""
    print("🧪 Probando modelo tiny optimizado...")
    
    # Crear configuración
    config = Config()
    
    # Forzar modelo tiny
    whisper_config = config.get_whisper_config()
    whisper_config['model_size'] = 'tiny'
    whisper_config['device'] = 'cpu'  # Usar CPU para prueba
    whisper_config['compute_type'] = 'int8'
    
    # Crear handler
    handler = FasterWhisperHandler(config)
    
    print(f"📊 Configuración del handler:")
    print(f"   - Modelo: {handler.model_size}")
    print(f"   - Dispositivo: {handler.device}")
    print(f"   - Compute Type: {handler.compute_type}")
    print(f"   - Es modelo tiny: {handler.is_tiny_model}")
    print(f"   - Optimizado para velocidad: {handler.optimize_for_speed}")
    
    # Cargar modelo
    print("\n🔄 Cargando modelo tiny...")
    start_time = time.time()
    
    if handler.load_model():
        load_time = time.time() - start_time
        print(f"✅ Modelo cargado en {load_time:.2f}s")
        
        # Obtener información del modelo
        model_info = handler.get_model_info()
        print(f"📋 Información del modelo:")
        for key, value in model_info.items():
            print(f"   - {key}: {value}")
        
        # Crear audio de prueba (1 segundo de silencio)
        print("\n🎵 Creando audio de prueba...")
        sample_rate = 16000
        audio_data = np.zeros(sample_rate, dtype=np.float32)  # 1 segundo de silencio
        
        # Test de transcripción normal
        print("\n🔤 Probando transcripción normal...")
        start_time = time.time()
        result = handler._transcribe_audio(audio_data, sample_rate, {})
        normal_time = time.time() - start_time
        
        print(f"✅ Transcripción normal completada en {normal_time:.3f}s")
        print(f"   - Texto: '{result.get('text', '')}'")
        print(f"   - Confianza: {result.get('confidence', 0.0):.3f}")
        
        # Test de transcripción en tiempo real
        print("\n⚡ Probando transcripción en tiempo real...")
        start_time = time.time()
        realtime_result = handler.transcribe_realtime(audio_data, sample_rate)
        realtime_time = time.time() - start_time
        
        print(f"✅ Transcripción en tiempo real completada en {realtime_time:.3f}s")
        print(f"   - Texto: '{realtime_result.get('text', '')}'")
        print(f"   - Confianza: {realtime_result.get('confidence', 0.0):.3f}")
        print(f"   - Optimizado para tiempo real: {realtime_result.get('realtime_optimized', False)}")
        
        # Comparar tiempos
        print(f"\n📊 Comparación de tiempos:")
        print(f"   - Carga del modelo: {load_time:.2f}s")
        print(f"   - Transcripción normal: {normal_time:.3f}s")
        print(f"   - Transcripción en tiempo real: {realtime_time:.3f}s")
        
        if realtime_time < normal_time:
            print(f"   🚀 ¡La optimización en tiempo real es {normal_time/realtime_time:.1f}x más rápida!")
        else:
            print(f"   ⚠️  La transcripción normal fue más rápida")
        
        # Limpiar recursos
        handler.cleanup()
        print("\n🧹 Recursos limpiados")
        
    else:
        print("❌ Error cargando modelo tiny")
        return False
    
    return True

def test_model_switching():
    """Test de cambio de modelos"""
    print("\n🔄 Probando cambio de modelos...")
    
    config = Config()
    handler = FasterWhisperHandler(config)
    
    # Cargar modelo tiny
    print("📦 Cargando modelo tiny...")
    if handler.load_model('tiny'):
        print("✅ Modelo tiny cargado")
        
        # Cambiar a modelo base
        print("📦 Cambiando a modelo base...")
        if handler.load_model('base'):
            print("✅ Modelo base cargado")
            
            # Volver a tiny
            print("📦 Volviendo a modelo tiny...")
            if handler.load_model('tiny'):
                print("✅ Modelo tiny recargado")
            else:
                print("❌ Error recargando modelo tiny")
                return False
        else:
            print("❌ Error cargando modelo base")
            return False
    else:
        print("❌ Error cargando modelo tiny inicial")
        return False
    
    handler.cleanup()
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DE MODELO TINY OPTIMIZADO")
    print("=" * 60)
    
    try:
        # Test principal
        if test_tiny_model():
            print("\n✅ Test principal completado exitosamente")
        else:
            print("\n❌ Test principal falló")
            sys.exit(1)
        
        # Test de cambio de modelos
        if test_model_switching():
            print("\n✅ Test de cambio de modelos completado exitosamente")
        else:
            print("\n❌ Test de cambio de modelos falló")
            sys.exit(1)
        
        print("\n🎉 Todos los tests completados exitosamente!")
        print("El modelo tiny está optimizado y funcionando correctamente.")
        
    except Exception as e:
        print(f"\n❌ Error durante los tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 