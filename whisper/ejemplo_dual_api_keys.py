#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de configuración y uso del sistema de doble API keys para Qwen

Este script muestra cómo:
1. Configurar dos API keys separadas (gratuita y de pago)
2. Habilitar el sistema de fallback automático
3. Probar el funcionamiento del fallback
"""

import logging
from config import config
from deepseek_handler import DeepSeekHandler

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configurar_dual_api_keys():
    """
    Configurar el sistema de doble API keys
    """
    print("=== Configuración de Doble API Keys para Qwen ===")
    print()
    
    # Solicitar API keys al usuario
    print("Para obtener tus API keys de OpenRouter:")
    print("1. Ve a https://openrouter.ai/")
    print("2. Regístrate o inicia sesión")
    print("3. Ve a la sección 'Keys'")
    print("4. Crea una nueva API key")
    print("5. Copia la key que empieza con 'sk-or-v1-'")
    print()
    
    # API key gratuita
    api_key_free = input("Ingresa tu API key GRATUITA (Enter para mantener actual): ").strip()
    if api_key_free:
        config.setup_deepseek_free_key(api_key_free)
        print("✓ API key gratuita configurada")
    
    # API key de pago
    api_key_paid = input("Ingresa tu API key DE PAGO (Enter para omitir): ").strip()
    if api_key_paid:
        config.setup_deepseek_paid_key(api_key_paid)
        print("✓ API key de pago configurada")
    
    # Habilitar sistema de doble API key
    if api_key_free or api_key_paid:
        config.enable_dual_keys(True)
        print("✓ Sistema de doble API key habilitado")
        print()
        return True
    else:
        print("⚠ No se configuraron API keys")
        return False

def probar_fallback():
    """
    Probar el sistema de fallback
    """
    print("=== Prueba del Sistema de Fallback ===")
    print()
    
    # Crear handler con la nueva configuración
    handler = DeepSeekHandler()
    
    if not handler.is_available:
        print("❌ Handler no disponible - verifica tu configuración")
        return
    
    # Mostrar configuración actual
    print(f"API key gratuita configurada: {'✓' if handler.api_key_free else '❌'}")
    print(f"API key de pago configurada: {'✓' if handler.api_key_paid else '❌'}")
    print(f"Sistema dual habilitado: {'✓' if handler.use_dual_keys else '❌'}")
    print(f"Usando API key de pago: {'✓' if handler.using_paid_key else '❌'}")
    print()
    
    # Textos de prueba
    textos_prueba = [
        "this text have some error in grammar",
        "i want to goes to the store yesterday",
        "the weather are very good today",
        "she don't like to eat vegetables",
        "we was very happy with the results"
    ]
    
    print("Probando correcciones (esto puede activar el fallback si se agotan los límites gratuitos):")
    print()
    
    for i, texto in enumerate(textos_prueba, 1):
        print(f"Prueba {i}/5:")
        print(f"Original: {texto}")
        
        # Realizar corrección
        resultado = handler.correct_text_sync(texto)
        
        if resultado['success']:
            print(f"Corregido: {resultado['corrected_text']}")
            print(f"Tiempo: {resultado['processing_time']:.2f}s")
            if 'model_used' in resultado:
                print(f"Modelo usado: {resultado['model_used']}")
            print(f"Usando API de pago: {'✓' if handler.using_paid_key else '❌'}")
        else:
            print(f"❌ Error: {resultado['error']}")
        
        print("-" * 50)
    
    # Mostrar estadísticas
    stats = handler.get_stats()
    print("\n=== Estadísticas ===")
    print(f"Total de correcciones: {stats['total_corrections']}")
    print(f"Exitosas: {stats['successful_corrections']}")
    print(f"Fallidas: {stats['failed_corrections']}")
    print(f"Tiempo promedio: {stats['average_response_time']:.2f}s")
    print(f"Caracteres procesados: {stats['total_characters_processed']}")

def mostrar_configuracion_actual():
    """
    Mostrar la configuración actual
    """
    print("=== Configuración Actual ===")
    
    deepseek_config = config.get_deepseek_config()
    
    print(f"Sistema dual habilitado: {'✓' if deepseek_config.get('use_dual_keys', False) else '❌'}")
    
    api_key_free = deepseek_config.get('api_key_free', '')
    if api_key_free and api_key_free != 'sk-or-v1-your-openrouter-api-key-here':
        print(f"API key gratuita: {api_key_free[:20]}...")
    else:
        print("API key gratuita: No configurada")
    
    api_key_paid = deepseek_config.get('api_key_paid', '')
    if api_key_paid and api_key_paid != 'sk-or-v1-your-openrouter-api-key-here':
        print(f"API key de pago: {api_key_paid[:20]}...")
    else:
        print("API key de pago: No configurada")
    
    print(f"Modelo gratuito: {deepseek_config.get('model', 'No configurado')}")
    print(f"Modelo de pago: {deepseek_config.get('fallback_model', 'No configurado')}")
    print(f"Habilitado: {'✓' if deepseek_config.get('enabled', False) else '❌'}")
    print()

def main():
    """
    Función principal
    """
    print("🤖 Sistema de Doble API Keys para Qwen 3")
    print("========================================")
    print()
    
    while True:
        print("Opciones:")
        print("1. Mostrar configuración actual")
        print("2. Configurar API keys")
        print("3. Probar sistema de fallback")
        print("4. Salir")
        print()
        
        opcion = input("Selecciona una opción (1-4): ").strip()
        print()
        
        if opcion == '1':
            mostrar_configuracion_actual()
        elif opcion == '2':
            configurar_dual_api_keys()
        elif opcion == '3':
            probar_fallback()
        elif opcion == '4':
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida")
        
        print()
        input("Presiona Enter para continuar...")
        print()

if __name__ == "__main__":
    main()