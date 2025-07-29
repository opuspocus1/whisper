# 🎤 Traductor de Voz en Tiempo Real

**Español → Inglés con IA avanzada**

Un traductor de voz en tiempo real que combina las mejores tecnologías de IA para ofrecer traducciones precisas y naturales.

## ✨ Características Principales

- **🎙️ Reconocimiento de Voz**: OpenAI Whisper para transcripción precisa
- **🔄 Traducción Inteligente**: Múltiples servicios de traducción con fallback automático
- **🧠 Corrección con IA**: Qwen para mejorar gramática y fluidez
- **🔊 Síntesis de Voz**: ElevenLabs para audio natural en inglés
- **🖥️ Interfaz Intuitiva**: GUI moderna con controles fáciles de usar

## 🚀 Tecnologías Integradas

### Core Components
- **OpenAI Whisper**: Reconocimiento de voz de última generación
- **ElevenLabs**: Síntesis de voz con calidad profesional
- **Qwen**: Corrección y mejora de texto con IA
- **Deep Translator**: Múltiples servicios de traducción

### Servicios de Traducción
- Google Translator (Deep Translator)
- Google Translate API
- Microsoft Translator
- LibreTranslate
- Traductor simple de fallback

## 📋 Requisitos del Sistema

- **Python**: 3.11+ (recomendado)
- **Sistema Operativo**: Windows 10/11
- **Memoria RAM**: 4GB mínimo, 8GB recomendado
- **Espacio en Disco**: 2GB para modelos de IA
- **Micrófono**: Para captura de audio
- **Altavoces/Auriculares**: Para reproducción

## 🛠️ Instalación

### Instalación Automática
```bash
# Ejecutar el instalador automático
.\install.bat
```

### Instalación Manual
```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar FFmpeg (incluido en el proyecto)
# El instalador automático ya configura FFmpeg
```

## ⚙️ Configuración

### APIs Requeridas

1. **ElevenLabs API Key**:
   - Registrarse en [ElevenLabs](https://elevenlabs.io)
   - Obtener API key desde el dashboard
   - Configurar en la aplicación

2. **OpenRouter API Key** (Para Qwen - Recomendado):
   - Registrarse GRATIS en [OpenRouter](https://openrouter.ai/)
   - Ir a la sección "Keys" y hacer clic en "Create Key"
   - Asignar un nombre a tu API key (ej: "Transcriptor-App")
   - Copiar la API key generada (¡guárdala de forma segura!)
   - La aplicación usa el modelo GRATUITO `qwen/qwen3-235b-a22b-2507:free`
   - **Fallback automático**: Si se agotan los requests gratuitos, usa `qwen/qwen3-235b-a22b-2507` (de pago)
   - **Nota**: La aplicación prioriza el modelo gratuito y solo usa el de pago como fallback

### 🔑 Sistema de Doble API Keys (Avanzado)

Para usuarios que quieren máxima disponibilidad, puedes configurar **dos API keys separadas**:

**Configuración Recomendada**:
- **API Key Gratuita**: Para uso diario con límites gratuitos
- **API Key de Pago**: Como respaldo automático cuando se agoten los límites

**Ventajas**:
- ✅ **Continuidad garantizada**: Nunca te quedas sin servicio
- ✅ **Optimización de costos**: Usa gratis primero, pago solo cuando sea necesario
- ✅ **Fallback automático**: Cambio transparente entre APIs
- ✅ **Configuración flexible**: Habilitar/deshabilitar según necesidades

**Cómo configurar**:
1. Crear dos cuentas en OpenRouter (o usar la misma con diferentes keys)
2. Una cuenta sin créditos (gratuita) y otra con créditos (pago)
3. Ejecutar `python ejemplo_dual_api_keys.py` para configuración guiada
4. O configurar manualmente en `settings.json`:
   ```json
   "deepseek": {
       "api_key_free": "sk-or-v1-tu-key-gratuita",
       "api_key_paid": "sk-or-v1-tu-key-de-pago",
       "use_dual_keys": true
   }
   ```

**Funcionamiento**:
- Inicia siempre con la API key gratuita
- Al detectar límites agotados, cambia automáticamente a la de pago
- Logs detallados muestran qué API key está en uso
- Estadísticas de uso disponibles en tiempo real

### Configuración Inicial
1. Ejecutar la aplicación
2. Ir a **⚙️ Configuración**
3. Introducir las API keys
4. Seleccionar dispositivos de audio
5. Ajustar configuraciones según preferencias

## 🎯 Uso

### Controles Principales
- **F1**: Iniciar/Detener grabación
- **F2**: Reproducir traducción
- **F3**: Limpiar textos
- **Ctrl+S**: Guardar traducción
- **Ctrl+Q**: Salir

### Flujo de Trabajo
1. **Configurar**: Establecer APIs y dispositivos de audio
2. **Grabar**: Presionar F1 o el botón de grabación
3. **Hablar**: Hablar en español claramente
4. **Procesar**: El sistema transcribe, traduce y corrige automáticamente
5. **Escuchar**: La traducción se reproduce en inglés
6. **Guardar**: Opcional - guardar la traducción

## 🔧 Características Avanzadas

### Qwen Integration
- **Corrección Enfocada**: Solo estructura y gramática, sin información extra
- **Preservación del Mensaje**: Mantiene el contenido original exacto
- **Fallback Automático**: Modelo gratuito con respaldo de pago
- **Configuración Flexible**: Habilitar/deshabilitar según necesidades

### Múltiples Traductores
- **Fallback Automático**: Si un servicio falla, usa otro automáticamente
- **Priorización Inteligente**: Usa el mejor traductor disponible
- **Configuración Personalizada**: Seleccionar servicios preferidos

### Audio Avanzado
- **Detección Automática**: Encuentra dispositivos de audio disponibles
- **Control de Volumen**: Ajuste en tiempo real
- **Calidad Optimizada**: Configuración automática para mejor calidad

## 📁 Estructura del Proyecto

```
whisper/
├── main.py                 # Punto de entrada principal
├── main_gui.py            # Interfaz gráfica
├── audio_manager.py       # Gestión de audio
├── whisper_handler.py     # Integración con Whisper
├── translator.py          # Servicios de traducción
├── qwen_handler.py        # Integración con Qwen
├── elevenlabs_handler.py  # Síntesis de voz
├── config.py              # Gestión de configuración
├── httpcore_compat.py     # Parches de compatibilidad
├── cgi.py                 # Compatibilidad Python 3.13
├── requirements.txt       # Dependencias
├── settings.json          # Configuración de usuario
└── ffmpeg-7.1.1-essentials_build/  # FFmpeg incluido
```

## 🐛 Solución de Problemas

### Problemas Comunes

**Error de módulo 'cgi'**:
- Solucionado automáticamente con `cgi.py` incluido
- Compatible con Python 3.13+

**Error 'BaseTransport' o 'SyncHTTPTransport'**:
- Solucionado con `httpcore_compat.py`
- Parches automáticos de compatibilidad

**Audio no funciona**:
- Verificar permisos de micrófono
- Comprobar dispositivos en configuración
- Reiniciar la aplicación

**Traducción lenta**:
- Verificar conexión a internet
- Comprobar APIs configuradas
- Usar modelo Whisper más pequeño

### Logs y Depuración
- Los logs se guardan en `logs/voice_translator.log`
- Nivel de detalle configurable
- Información de rendimiento incluida

## 🔄 Actualizaciones Recientes

### v2.2 - Sistema de Doble API Keys
- ✅ **Nuevo**: Sistema de doble API keys para máxima disponibilidad
- ✅ **Fallback automático**: Cambio transparente entre API gratuita y de pago
- ✅ **Optimización de costos**: Prioriza uso gratuito, pago solo como respaldo
- ✅ **Script de configuración**: `ejemplo_dual_api_keys.py` para setup guiado
- ✅ **Compatibilidad**: Mantiene funcionamiento con configuración antigua
- ✅ **Logs mejorados**: Información detallada sobre qué API key está en uso
- ✅ **Estadísticas avanzadas**: Monitoreo de uso y rendimiento en tiempo real

### v2.1 - Migración a Qwen
- ✅ Migración de DeepSeek a Qwen 3 para mejor control de corrección
- ✅ Fallback automático de modelo gratuito a modelo de pago
- ✅ Corrección enfocada - solo estructura y gramática, sin información extra
- ✅ Parches de compatibilidad para Python 3.13
- ✅ Resolución de conflictos de dependencias
- ✅ Mejoras en la estabilidad del sistema
- ✅ Interfaz de configuración actualizada

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

## 📞 Soporte

Para soporte técnico:
- Revisar la documentación
- Consultar los logs de la aplicación
- Verificar configuración de APIs
- Comprobar requisitos del sistema

---

**¡Disfruta traduciendo con IA de última generación!** 🚀