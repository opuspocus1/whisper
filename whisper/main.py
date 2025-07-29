#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traductor de Voz en Tiempo Real - Español a Inglés
Usando OpenAI Whisper + ElevenLabs

Ejecución principal de la aplicación
"""

import sys
import os
import logging
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Verificar dependencias críticas"""
    missing_deps = []
    
    # Verificar dependencias básicas
    try:
        import tkinter
    except ImportError:
        missing_deps.append('tkinter')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import sounddevice
    except ImportError:
        missing_deps.append('sounddevice')
    
    try:
        import requests
    except ImportError:
        missing_deps.append('requests')
    
    # Verificar Whisper (faster-whisper)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        missing_deps.append('faster-whisper')
    
    if missing_deps:
        print("❌ Dependencias faltantes:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Ejecuta: pip install -r requirements.txt")
        return False
    
    return True

def setup_logging():
    """Configurar sistema de logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'voice_translator.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Reducir verbosidad de algunas librerías
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('faster_whisper').setLevel(logging.WARNING)

def show_welcome():
    """Mostrar mensaje de bienvenida"""
    print("\n" + "="*60)
    print("🎤 TRADUCTOR DE VOZ EN TIEMPO REAL 🔊")
    print("   Español → Inglés")
    print("   Powered by OpenAI Whisper + ElevenLabs")
    print("="*60)
    print("\n📋 Características:")
    print("   • Reconocimiento de voz en español (Whisper)")
    print("   • Traducción automática")
    print("   • Síntesis de voz en inglés (ElevenLabs)")
    print("   • Interfaz gráfica intuitiva")
    print("\n🎯 Atajos de teclado:")
    print("   • F1: Iniciar/Detener grabación")
    print("   • F2: Reproducir traducción")
    print("   • F3: Limpiar textos")
    print("   • Ctrl+S: Guardar traducción")
    print("   • Ctrl+Q: Salir")
    print("\n🚀 Iniciando aplicación...\n")

def main():
    """Función principal"""
    try:
        # Mostrar bienvenida
        show_welcome()
        
        # Verificar dependencias
        print("🔍 Verificando dependencias...")
        if not check_dependencies():
            print("\n❌ No se puede continuar sin las dependencias requeridas.")
            input("\nPresiona Enter para salir...")
            return 1
        
        print("✅ Todas las dependencias están disponibles")
        
        # Configurar logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 50)
        logger.info("Iniciando Traductor de Voz en Tiempo Real")
        logger.info("=" * 50)
        
        # Importar y ejecutar GUI
        print("🎨 Cargando interfaz gráfica...")
        
        try:
            from main_gui import VoiceTranslatorGUI
            
            # Crear y ejecutar aplicación
            app = VoiceTranslatorGUI()
            logger.info("GUI inicializada correctamente")
            
            print("✅ Aplicación lista. ¡Disfruta traduciendo!")
            app.run()
            
        except ImportError as e:
            logger.error(f"Error importando GUI: {e}")
            print(f"❌ Error cargando interfaz: {e}")
            return 1
        
        except Exception as e:
            logger.error(f"Error en GUI: {e}")
            print(f"❌ Error en la aplicación: {e}")
            return 1
        
        logger.info("Aplicación cerrada correctamente")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Aplicación interrumpida por el usuario")
        return 0
    
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        logging.error(f"Error fatal: {e}", exc_info=True)
        input("\nPresiona Enter para salir...")
        return 1

if __name__ == "__main__":
    sys.exit(main())