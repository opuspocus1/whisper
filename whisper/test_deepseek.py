#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script para Qwen Handler
Prueba la funcionalidad de corrección de texto usando Qwen 3 via OpenRouter

Para usar este script:
1. Ve a https://openrouter.ai/ y regístrate gratis
2. Ve a 'Keys' y crea una nueva API key
3. Copia tu API key (empieza con 'sk-or-v1-')
4. Reemplaza 'sk-or-v1-your-openrouter-api-key-here' abajo con tu API key real
5. Ejecuta: python test_deepseek.py

El script usa el modelo GRATUITO 'qwen/qwen3-235b-a22b-2507:free'
con fallback automático a 'qwen/qwen3-235b-a22b-2507' (de pago)
"""

import asyncio
from deepseek_handler import DeepSeekHandler

def test_qwen_correction():
    """Función de prueba para Qwen Handler"""
    print("=== Test Qwen Handler ===")
    print("Modelo primario: qwen/qwen3-235b-a22b-2507:free (GRATUITO)")
    print("Modelo fallback: qwen/qwen3-235b-a22b-2507 (DE PAGO)")
    print("Proveedor: OpenRouter")
    print()
    
    # Usar configuración desde config.py
    from config import config as app_config
    config = app_config.get_deepseek_config()
    
    # Crear handler
    print("Inicializando Qwen Handler...")
    handler = DeepSeekHandler(config)
    
    if not handler.is_available:
        print("❌ Error: Qwen Handler no está disponible")
        print("   Verifica tu API key de OpenRouter")
        return
    
    print("✅ Qwen Handler inicializado correctamente")
    print()
    
    # Textos de prueba con errores comunes (enfoque en estructura y gramática)
    test_texts = [
        "hello how are you today i am fine",
        "i go to store yesterday and buy some food",
        "the weather is very good today isnt it",
        "can you help me with this problem please",
        "thank you very much for you help",
        "i dont know what to do about this situation",
        "she have been working here for five years",
        "we was planning to visit the museum tomorrow",
        "its a beautiful day outside today",
        "could you please tell me where is the library"
    ]
    
    print("Probando corrección de textos...")
    
    successful_corrections = 0
    failed_corrections = 0
    improvements = 0
    total_time = 0
    total_chars = 0
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Prueba {i+1}/{len(test_texts)} ---")
        print(f"Original: '{text}'")
        
        # Corregir texto
        result = handler.correct_text_sync(text)
        
        if result['success']:
            print(f"Corregido: '{result['corrected_text']}'")
            print(f"Tiempo: {result['processing_time']:.2f}s")
            if 'model_used' in result:
                print(f"Modelo usado: {result['model_used']}")
            
            successful_corrections += 1
            total_time += result['processing_time']
            total_chars += len(text)
            
            # Verificar si hubo cambios
            if result['original_text'] != result['corrected_text']:
                improvements += 1
        else:
            print(f"❌ Error: {result['error']}")
            failed_corrections += 1
    
    print("\n=== Resumen de Pruebas ===")
    print(f"Total de textos procesados: {len(test_texts)}")
    print(f"Correcciones exitosas: {successful_corrections}")
    print(f"Correcciones fallidas: {failed_corrections}")
    print(f"Textos mejorados: {improvements}")
    print(f"Tiempo total: {total_time:.2f}s")
    print(f"Tiempo promedio: {total_time/max(successful_corrections, 1):.2f}s")
    print(f"Caracteres procesados: {total_chars}")
    print(f"Velocidad: {total_chars/max(total_time, 0.001):.0f} chars/s")
    
    # Estadísticas del handler
    stats = handler.get_stats()
    print("\n=== Estadísticas del Handler ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Limpiar
    handler.cleanup()
    print("\n✅ Qwen Handler limpiado")
    print("Prueba completada")

if __name__ == "__main__":
    test_qwen_correction()