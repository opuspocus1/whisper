# 🎙️ Complete Kokoro TTS API

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

> **A lightning-fast, production-grade text-to-speech server with OpenAI-style quality, robust text processing, and accessibility-first design. Processing time: ~1 second with nearly instant output.**

*Created by [nodeblackbox](https://github.com/nodeblackbox) - Making accessibility available for everyone.*

## ✨ Overview

The Complete Kokoro TTS API delivers **OpenAI-style text-to-speech quality** with exceptional performance and accessibility features. Designed with a commitment that **accessibility should be for everyone**, this API provides crystal-clear voices especially suitable for dyslexic users and assistive technology integration.

**🚀 Performance Highlights:**
- **~1 second total processing time**
- **Nearly instant audio output**
- **GPU acceleration available**
- **Real-time streaming capabilities**

## 🎯 Accessibility & Integration

### 🔗 Read Aloud Chrome Extension Integration

This API seamlessly integrates with the **[Read Aloud](https://chromewebstore.google.com/detail/read-aloud-a-text-to-spee/hdhinadidafjejdhmfkjgnolgimiaplp)** Chrome extension, providing an excellent solution for dyslexic users and anyone who benefits from text-to-speech technology.

**Setup Instructions:**

1. **Install the Extension**: Add [Read Aloud](https://chromewebstore.google.com/detail/read-aloud-a-text-to-spee/hdhinadidafjejdhmfkjgnolgimiaplp) to Chrome
2. **Configure API Endpoint**: `http://127.0.0.1:5000/v1`
3. **API Key**: `your-secret-key`
4. **Select from 28 High-Quality Voices** (see voice configuration below)

### 🎤 Available Voices for Read Aloud

```json
[
  { "lang": "en-US", "model": "tts-1", "voice": "af_heart" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_bella" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_nicole" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_aoede" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_kore" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_sarah" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_nova" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_sky" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_alloy" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_jessica" },
  { "lang": "en-US", "model": "tts-1", "voice": "af_river" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_michael" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_fenrir" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_puck" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_echo" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_eric" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_liam" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_onyx" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_santa" },
  { "lang": "en-US", "model": "tts-1", "voice": "am_adam" },
  { "lang": "en-GB", "model": "tts-1", "voice": "bf_emma" },
  { "lang": "en-GB", "model": "tts-1", "voice": "bf_isabella" },
  { "lang": "en-GB", "model": "tts-1", "voice": "bf_alice" },
  { "lang": "en-GB", "model": "tts-1", "voice": "bf_lily" },
  { "lang": "en-GB", "model": "tts-1", "voice": "bm_george" },
  { "lang": "en-GB", "model": "tts-1", "voice": "bm_fable" },
  { "lang": "en-GB", "model": "tts-1", "voice": "bm_lewis" },
  { "lang": "en-GB", "model": "tts-1", "voice": "bm_daniel" }
]
```

## 🚀 Features

### Core Features
- **🔧 Robust Text Processing**: Intelligent handling of markdown, Unicode characters, numbers, abbreviations, and special formatting
- **⚡ Ultra-Fast Performance**: ~1 second total processing with nearly instant output
- **🎚️ Zero-Default Effects**: Clean audio output with effects only when explicitly configured
- **🎵 Local Playback Control**: Built-in audio playback with interrupt capability and session management
- **📡 Real-time Streaming**: Live audio streaming support for compatible clients
- **🎼 Advanced Audio Effects**: FIXED and robust pitch shifting with librosa compatibility
- **🌐 Browser Integration**: Full CORS support for browser extensions and web applications
- **♿ Accessibility First**: Crystal-clear voices optimized for dyslexic users and assistive technology

### Audio Processing
- **28 High-quality voices** (20 US English, 8 British English)
- **OpenAI-style TTS quality** with superior clarity
- Various audio format outputs
- Pitch shifting and formant modification
- Dynamic range compression
- Professional-grade audio processing

### Developer Experience
- RESTful API design
- OpenAPI specification
- Comprehensive error handling
- Session-based playback management
- Easy integration with existing applications
- Chrome extension compatibility

## 📋 Table of Contents

- [🚀 Features](#-features)
- [⚡ Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔧 Configuration](#-configuration)
- [📡 API Endpoints](#-api-endpoints)
- [💡 Usage Examples](#-usage-examples)
- [🎨 Audio Effects](#-audio-effects)
- [📚 API Documentation](#-api-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/Kokoro-Voice-Api.git
cd Kokoro-Voice-Api

# Install dependencies
pip install -r requirements.txt

# Run the server
python kokoro_api.py

# Test the API
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "af_heart"}' \
  http://localhost:5000/v1/audio/speech
```

## 📦 Installation

### Prerequisites
- **Python 3.8+** (recommended: Python 3.10+)
- **PyTorch** with CUDA support (optional, for GPU acceleration)
- **Git** for cloning the repository

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Kokoro-Voice-Api.git
   cd Kokoro-Voice-Api
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch, librosa, flask; print('✅ All dependencies installed successfully!')"
   ```

## 🔧 Configuration

### Environment Variables
```bash
# Server Configuration
export TTS_HOST=0.0.0.0
export TTS_PORT=5000
export TTS_DEBUG=false

# Audio Configuration
export TTS_SAMPLE_RATE=22050
export TTS_AUDIO_FORMAT=wav

# Performance
export TTS_MAX_TEXT_LENGTH=1000
export TTS_CACHE_SIZE=100
```

### Configuration File
Create a `config.yaml` file in the project root:
```yaml
server:
  host: "0.0.0.0"
  port: 5000
  debug: false

audio:
  sample_rate: 22050
  format: "wav"
  quality: "high"

processing:
  max_text_length: 1000
  cache_enabled: true
  cache_size: 100
```

## 📡 API Endpoints

### Speech Generation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Standard speech generation with clean zero-default effects |
| `/v1/audio/speech/robust` | POST | Enhanced speech generation with advanced text processing |
| `/v1/audio/speech/stream` | POST | Real-time streaming speech generation |

### Playback Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech/play` | POST | Local playback with session control |
| `/v1/audio/speech/stop` | POST | Stop/interrupt current playback |
| `/v1/audio/speech/status` | GET | Get current playback status |

### System Information

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/voices` | GET | List available voices |
| `/version` | GET | API version information |

## 💡 Usage Examples

### Basic Text-to-Speech
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world! This is a test of the Kokoro TTS API.",
    "voice": "af_heart"
  }' \
  http://localhost:5000/v1/audio/speech \
  --output hello.wav
```

### Squeaky Voice Effect
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I sound like a chipmunk!",
    "voice": "af_heart",
    "effects": {
      "pitch": {
        "semitone_shift": 8.0
      }
    }
  }' \
  http://localhost:5000/v1/audio/speech \
  --output squeaky.wav
```

### Deep Voice Effect
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I have a very deep voice now.",
    "voice": "af_heart",
    "effects": {
      "pitch": {
        "semitone_shift": -6.0
      }
    }
  }' \
  http://localhost:5000/v1/audio/speech \
  --output deep.wav
```

### Robust Text Processing
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Process this: **bold text**, _italic_, numbers: 123, $50.99, and 50% off!",
    "voice": "af_heart",
    "robust_processing": true
  }' \
  http://localhost:5000/v1/audio/speech/robust \
  --output processed.wav
```

### Streaming Audio
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This will be streamed in real-time as it is generated.",
    "voice": "af_heart",
    "stream": true
  }' \
  http://localhost:5000/v1/audio/speech/stream \
  --output stream.wav
```

### Local Playback Control
```bash
# Start playback
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This will play locally on the server.",
    "voice": "af_heart",
    "session_id": "my-session"
  }' \
  http://localhost:5000/v1/audio/speech/play

# Check status
curl http://localhost:5000/v1/audio/speech/status

# Stop playback
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session"}' \
  http://localhost:5000/v1/audio/speech/stop
```

## 🎨 Audio Effects

### Pitch Modification
```json
{
  "effects": {
    "pitch": {
      "semitone_shift": 4.0,     // Shift by semitones (-12 to +12)
      "preserve_formants": true   // Maintain voice character
    }
  }
}
```

### Dynamic Range Compression
```json
{
  "effects": {
    "compression": {
      "ratio": 4.0,              // Compression ratio
      "threshold": -20.0,        // Threshold in dB
      "attack": 0.003,           // Attack time in seconds
      "release": 0.1             // Release time in seconds
    }
  }
}
```

### Multiple Effects
```json
{
  "effects": {
    "pitch": {
      "semitone_shift": 2.0
    },
    "compression": {
      "ratio": 2.0,
      "threshold": -18.0
    },
    "reverb": {
      "room_size": 0.3,
      "damping": 0.5,
      "wet_level": 0.2
    }
  }
}
```

## 🛠️ Development

### Running in Development Mode
```bash
# Enable debug mode
export FLASK_ENV=development
export TTS_DEBUG=true

# Run with auto-reload
python kokoro_api.py
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=kokoro_api
```

### Docker Support
```dockerfile
# Build Docker image
docker build -t kokoro-tts-api .

# Run container
docker run -p 5000:5000 kokoro-tts-api
```

## 📚 API Documentation

### OpenAPI Specification
The complete API documentation is available in OpenAPI format:
- **Specification File**: [`openapi.yaml`](openapi.yaml)
- **Interactive Documentation**: Visit `/docs` when the server is running
- **Redoc Documentation**: Visit `/redoc` when the server is running

### Response Formats
All endpoints return standardized responses:

**Success Response:**
```json
{
  "success": true,
  "data": {
    "audio_url": "/generated/audio.wav",
    "duration": 2.5,
    "sample_rate": 22050
  },
  "metadata": {
    "voice": "af_heart",
    "effects_applied": ["pitch_shift"],
    "processing_time": 0.85
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_VOICE",
    "message": "The specified voice 'invalid_voice' is not available",
    "details": {
      "available_voices": ["af_heart", "af_bella", "af_sarah"]
    }
  }
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
   ```bash
   git fork https://github.com/your-username/Kokoro-Voice-Api.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

3. **Make Your Changes**
   - Follow PEP 8 style guidelines
   - Add tests for new functionality
   - Update documentation as needed

4. **Run Tests**
   ```bash
   pytest tests/ -v
   black kokoro_api.py
   flake8 kokoro_api.py
   ```

5. **Submit a Pull Request**
   - Provide a clear description of your changes
   - Reference any related issues
   - Ensure all tests pass

### Development Guidelines
- **Code Style**: Follow PEP 8 and use `black` for formatting
- **Testing**: Maintain >90% test coverage
- **Documentation**: Update docstrings and README for new features
- **Performance**: Profile code for optimization opportunities

## 🔒 Security

- **API Keys**: Never hardcode API keys; use environment variables
- **Input Validation**: All inputs are sanitized and validated
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **CORS**: Configurable CORS settings for web integration

## 📊 Performance

### Benchmarks
- **Average Response Time**: ~1 second for 50-word text
- **Output Latency**: Nearly instant audio delivery
- **Concurrent Requests**: Supports up to 10 simultaneous requests
- **Memory Usage**: ~200MB baseline + ~50MB per active session
- **GPU Acceleration**: 3x faster processing with CUDA-enabled PyTorch
- **Voice Quality**: OpenAI-comparable clarity and naturalness

### Optimization Tips
- Use GPU acceleration when available for fastest processing
- Enable caching for repeated requests
- Batch multiple requests when possible
- Use streaming for long-form content
- Perfect for real-time applications and accessibility tools

## 🌟 Accessibility Statement

**We believe accessibility should be for everyone.** This API is specifically designed with dyslexic users and assistive technology in mind, providing:

- **Crystal-clear voice quality** optimized for comprehension
- **Multiple accent options** (US and British English)
- **Fast processing** for responsive user experience
- **Browser extension compatibility** for seamless web integration
- **Professional-grade audio** without distortion or artifacts

### Perfect for:
- 📚 **Dyslexic students and professionals**
- 👩‍🦯 **Users with visual impairments**
- 🧠 **People with learning differences**
- 👥 **Anyone who benefits from audio content**
- 🌐 **Web accessibility implementations**

## 🙏 Acknowledgments

- **[nodeblackbox](https://github.com/nodeblackbox)** - Project creator and maintainer
- **Kokoro TTS Team** for the underlying neural TTS technology
- **PyTorch Team** for the deep learning framework
- **Librosa Contributors** for audio processing capabilities
- **Flask Community** for the web framework
- **Accessibility advocates** who inspire inclusive technology

---

<div align="center">

**[⬆ Back to Top](#-complete-kokoro-tts-api)**

Made with ❤️ for accessibility by [nodeblackbox](https://github.com/nodeblackbox)

*"Accessibility should be for everyone"*

</div>
# Updated Kokoro-Voice-Api as regular folder
