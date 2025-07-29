@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ================================================================
echo INSTALADOR - TRADUCTOR DE VOZ EN TIEMPO REAL
echo    Espanol -^> Ingles
echo    Powered by OpenAI Whisper + ElevenLabs
echo ================================================================
echo.

:: Verificar si Python esta instalado
echo [INFO] Verificando Python...
py --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python no esta instalado o no esta en el PATH
        echo.
        echo [TIP] Por favor instala Python desde: https://python.org
        echo    Asegurate de marcar "Add Python to PATH" durante la instalacion
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python
    )
) else (
    set PYTHON_CMD=py
)

for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% encontrado

:: Verificar pip
echo [INFO] Verificando pip...
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip no esta disponible
    echo [INFO] Instalando pip...
    %PYTHON_CMD% -m ensurepip --upgrade
    if errorlevel 1 (
        echo [ERROR] No se pudo instalar pip
        pause
        exit /b 1
    )
)
echo [OK] pip disponible

:: Actualizar pip
echo [INFO] Actualizando pip...
%PYTHON_CMD% -m pip install --upgrade pip

:: Verificar si ffmpeg esta disponible
echo [INFO] Verificando ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg no encontrado
    echo.
    echo [TIP] ffmpeg es requerido para Whisper. Opciones:
    echo    1. Instalar desde: https://ffmpeg.org/download.html
    echo    2. Usar chocolatey: choco install ffmpeg
    echo    3. Usar winget: winget install ffmpeg
    echo.
    echo [INFO] La aplicacion intentara funcionar sin ffmpeg, pero puede tener limitaciones
    echo.
    set /p continue="Continuar sin ffmpeg? (s/n): "
    if /i "!continue!" neq "s" (
        echo Instalacion cancelada
        pause
        exit /b 1
    )
) else (
    echo [OK] ffmpeg disponible
)

:: Crear entorno virtual (opcional)
echo.
set /p create_venv="Crear entorno virtual? (recomendado) (s/n): "
if /i "!create_venv!" equ "s" (
    echo [INFO] Creando entorno virtual...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Error creando entorno virtual
        pause
        exit /b 1
    )
    
    echo [INFO] Activando entorno virtual...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo [ERROR] Error activando entorno virtual
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual activado
)

:: Instalar dependencias
echo.
echo [INFO] Instalando dependencias...
echo    Esto puede tomar varios minutos...
echo.

:: Instalar dependencias basicas primero
echo [INFO] Instalando dependencias basicas...
%PYTHON_CMD% -m pip install numpy requests sounddevice soundfile
if errorlevel 1 (
    echo [ERROR] Error instalando dependencias basicas
    pause
    exit /b 1
)

:: Instalar Whisper
echo [INFO] Instalando Faster Whisper...
%PYTHON_CMD% -m pip install faster-whisper
if errorlevel 1 (
    echo [ERROR] Error instalando Faster Whisper
    pause
    exit /b 1
)

:: Instalar dependencias de traduccion
echo [INFO] Instalando traductores...
%PYTHON_CMD% -m pip install deep-translator googletrans==4.0.0rc1
if errorlevel 1 (
    echo [ERROR] Error instalando traductores
    pause
    exit /b 1
)

:: Instalar dependencias de audio
echo [INFO] Instalando procesamiento de audio...
%PYTHON_CMD% -m pip install noisereduce pyaudio
if errorlevel 1 (
    echo [ERROR] Error instalando procesamiento de audio
    pause
    exit /b 1
)

:: Verificar instalacion
echo.
echo [INFO] Verificando instalacion...
%PYTHON_CMD% -c "import whisper; import numpy; import sounddevice; import requests; print('[OK] Dependencias principales verificadas')"
if errorlevel 1 (
    echo [ERROR] Error en la verificacion
    pause
    exit /b 1
)

:: Descargar modelo base de Whisper
echo.
set /p download_model="Descargar modelo base de Whisper ahora? (recomendado) (s/n): "
if /i "!download_model!" equ "s" (
    echo [INFO] Descargando modelo base de Whisper...
    echo    Esto puede tomar varios minutos dependiendo de tu conexion...
    %PYTHON_CMD% -c "import whisper; whisper.load_model('base')"
    if errorlevel 1 (
        echo [WARNING] Error descargando modelo (se descargara automaticamente al usar la app)
    ) else (
        echo [OK] Modelo base descargado
    )
)

:: Crear directorio de logs
echo.
echo [INFO] Creando estructura de directorios...
if not exist "logs" mkdir logs
echo [OK] Directorio logs creado

:: Crear archivo de configuracion inicial
echo [INFO] Creando configuracion inicial...
if not exist "settings.json" (
    echo {
    echo   "elevenlabs": {
    echo     "api_key": "",
    echo     "voice_id": "21m00Tcm4TlvDq8ikWAM",
    echo     "model_id": "eleven_multilingual_v2"
    echo   },
    echo   "whisper": {
    echo     "model_size": "base",
    echo     "language": "es"
    echo   },
    echo   "audio": {
    echo     "sample_rate": 16000,
    echo     "chunk_size": 1024
    echo   }
    echo } > settings.json
    echo [OK] Archivo settings.json creado
)

:: Crear script de ejecucion
echo [INFO] Creando script de ejecucion...
echo @echo off > ejecutar.bat
echo chcp 65001 ^>nul >> ejecutar.bat
if /i "!create_venv!" equ "s" (
    echo call venv\Scripts\activate.bat >> ejecutar.bat
)
echo %PYTHON_CMD% main.py >> ejecutar.bat
echo pause >> ejecutar.bat
echo [OK] Script ejecutar.bat creado

echo.
echo ================================================================
echo INSTALACION COMPLETADA!
echo ================================================================
echo.
echo [OK] Todas las dependencias han sido instaladas
echo [OK] Configuracion inicial creada
echo [OK] Script de ejecucion creado
echo.
echo [INFO] Para ejecutar la aplicacion:
echo    1. Haz doble clic en "ejecutar.bat"
echo    2. O ejecuta: python main.py
echo.
echo [TIP] Configuracion:
echo    - Configura tu API key de ElevenLabs en la aplicacion
echo    - Ajusta configuraciones en settings.json si es necesario
echo.
echo [INFO] Consulta README.md para mas informacion
echo ================================================================
echo.
pause