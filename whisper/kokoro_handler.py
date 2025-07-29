import requests
import json
import logging
import tempfile
import os
import subprocess
import time
import threading
import asyncio
import queue
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np

@dataclass
class PerformanceMetrics:
    """Performance metrics for latency monitoring"""
    first_byte_latency: float = 0.0
    total_latency: float = 0.0
    synthesis_time: float = 0.0
    network_time: float = 0.0
    buffer_time: float = 0.0
    timestamp: float = 0.0

class OptimizedKokoroHandler:
    """Optimized Kokoro Voice API Handler for sub-100ms latency"""
    
    def __init__(self, 
                 api_url: str = "http://localhost:5000", 
                 voice: str = "af_heart",
                 max_workers: int = 4,
                 connection_pool_size: int = 10,
                 chunk_size: int = 512,
                 pre_warm_models: bool = True):
        
        self.api_url = api_url
        self.voice = voice
        self.server_process = None
        self.logger = logging.getLogger(__name__)
        self.kokoro_path = os.path.join(os.path.dirname(__file__), "Kokoro-Voice-Api")
        
        # Callbacks
        self.on_audio_ready: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Performance optimization settings
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.pre_warm_models = pre_warm_models
        
        # Connection pooling with optimized settings
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=connection_pool_size,
            pool_maxsize=connection_pool_size,
            max_retries=1,
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="KokoroWorker"
        )
        
        # Performance monitoring
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.performance_history: List[PerformanceMetrics] = []
        
        # Model pre-warming
        self.model_warmed = False
        self.warmup_lock = threading.Lock()
        
        # Buffer management
        self.audio_buffer = queue.Queue(maxsize=50)
        self.buffer_thread = None
        self.buffer_running = False
        
        # Health check settings
        self.health_check_timeout = 0.5
        self.synthesis_timeout = 10.0
        self.streaming_timeout = 15.0
        
        # Start buffer management thread
        self._start_buffer_manager()
        
        # Pre-warm models if enabled
        if pre_warm_models:
            self._pre_warm_models_async()
    
    def _start_buffer_manager(self):
        """Start the buffer management thread for immediate flush/fsync"""
        self.buffer_running = True
        self.buffer_thread = threading.Thread(
            target=self._buffer_manager_worker,
            daemon=True,
            name="KokoroBufferManager"
        )
        self.buffer_thread.start()
    
    def _buffer_manager_worker(self):
        """Worker thread for optimized buffer management"""
        while self.buffer_running:
            try:
                # Process buffered audio data with immediate flush
                audio_data = self.audio_buffer.get(timeout=0.1)
                if audio_data:
                    self._process_audio_buffer(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Buffer manager error: {e}")
    
    def _process_audio_buffer(self, audio_data: bytes):
        """Process audio buffer with immediate flush/fsync"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force sync to disk
                return temp_file.name
        except Exception as e:
            self.logger.error(f"Buffer processing error: {e}")
            return None
    
    def _pre_warm_models_async(self):
        """Pre-warm models asynchronously to eliminate cold start latency"""
        def warmup_worker():
            try:
                with self.warmup_lock:
                    if self.model_warmed:
                        return
                    
                    self.logger.info("🔥 Pre-warming Kokoro models...")
                    start_time = time.time()
                    
                    # Warm up with a short text
                    warmup_text = "Hello world"
                    warmup_data = {
                        "input": warmup_text,
                        "voice": self.voice,
                        "response_format": "mp3",
                        "use_gpu": True,
                        "speed": 1.0,
                        "optimize_for_latency": True
                    }
                    
                    try:
                        response = self.session.post(
                            f"{self.api_url}/v1/audio/speech",
                            json=warmup_data,
                            timeout=5.0,
                            headers={'Connection': 'keep-alive'}
                        )
                        
                        if response.status_code == 200:
                            warmup_time = time.time() - start_time
                            self.model_warmed = True
                            self.logger.info(f"✅ Models pre-warmed in {warmup_time:.3f}s")
                        else:
                            self.logger.warning(f"⚠️ Model warmup failed: {response.status_code}")
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Model warmup error: {e}")
                        
            except Exception as e:
                self.logger.error(f"❌ Model warmup failed: {e}")
        
        # Run warmup in background
        self.executor.submit(warmup_worker)
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        try:
            self.performance_history.append(metrics)
            # Keep only last 1000 metrics
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Log if latency is high
            if metrics.first_byte_latency > 0.1:  # > 100ms
                self.logger.warning(f"⚠️ High first-byte latency: {metrics.first_byte_latency:.3f}s")
                
        except Exception as e:
            self.logger.error(f"Metrics recording error: {e}")
    
    def _get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.performance_history:
            return {}
        
        latencies = [m.first_byte_latency for m in self.performance_history]
        return {
            'avg_first_byte_latency': np.mean(latencies),
            'min_first_byte_latency': np.min(latencies),
            'max_first_byte_latency': np.max(latencies),
            'p95_first_byte_latency': np.percentile(latencies, 95),
            'p99_first_byte_latency': np.percentile(latencies, 99),
            'total_requests': len(self.performance_history)
        }
    
    def start_server(self) -> bool:
        """Start Kokoro API server with optimized settings"""
        try:
            if self.is_server_running():
                self.logger.info("✅ Kokoro server already running")
                return True
                
            kokoro_script = os.path.join(self.kokoro_path, "kokoro_api.py")
            if not os.path.exists(kokoro_script):
                self.logger.error(f"❌ kokoro_api.py not found: {kokoro_script}")
                return False
                
            self.logger.info("🚀 Starting optimized Kokoro server...")
            self.server_process = subprocess.Popen(
                ["python", kokoro_script],
                cwd=self.kokoro_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            
            # Wait for server with reduced timeout
            for i in range(15):  # 15 seconds max
                if self.is_server_running():
                    self.logger.info("✅ Kokoro server started successfully")
                    # Pre-warm models after server start
                    if self.pre_warm_models:
                        self._pre_warm_models_async()
                    return True
                time.sleep(1)
                
            self.logger.error("❌ Timeout waiting for Kokoro server")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error starting Kokoro server: {e}")
            return False
    
    def stop_server(self):
        """Stop Kokoro server with proper cleanup"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=3)
                self.logger.info("✅ Kokoro server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.logger.warning("⚠️ Forced Kokoro server shutdown")
            except Exception as e:
                self.logger.error(f"❌ Error stopping server: {e}")
            finally:
                self.server_process = None
    
    def is_server_running(self) -> bool:
        """Check server status with optimized timeout"""
        try:
            response = self.session.get(
                f"{self.api_url}/ping",
                timeout=self.health_check_timeout,
                headers={'Connection': 'keep-alive'}
            )
            return response.status_code == 200
        except:
            return False
    
    def get_voices(self) -> Dict[str, Any]:
        """Get available voices with connection reuse"""
        try:
            if not self.is_server_running():
                if not self.start_server():
                    return {}
                    
            response = self.session.get(
                f"{self.api_url}/v1/voices",
                timeout=2.0,
                headers={'Connection': 'keep-alive'}
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            self.logger.error(f"❌ Error getting voices: {e}")
            return {}
    
    def synthesize(self, 
                   text: str, 
                   voice: Optional[str] = None, 
                   use_gpu: bool = True, 
                   use_streaming: bool = None,
                   optimize_for_latency: bool = True) -> Optional[str]:
        """Optimized text-to-speech synthesis with sub-100ms latency"""
        
        # Auto-determine streaming based on text length
        if use_streaming is None:
            use_streaming = len(text) > 50
        
        try:
            if not text.strip():
                return None
                
            # Ensure server is running
            if not self.is_server_running():
                if not self.start_server():
                    self.logger.error("❌ Failed to start Kokoro server")
                    return None
            
            selected_voice = voice or self.voice
            self.logger.info(f"🎤 SYNTHESIS: {selected_voice} - {text[:50]}... (GPU: {use_gpu}, Stream: {use_streaming})")
            
            # Prepare optimized request data
            data = {
                "input": text,
                "voice": selected_voice,
                "response_format": "mp3",
                "use_gpu": use_gpu,
                "speed": 1.0,
                "optimize_for_latency": optimize_for_latency
            }
            
            # Choose endpoint based on streaming
            if use_streaming:
                endpoint = f"{self.api_url}/v1/audio/speech/stream"
                data["stream"] = True
            else:
                endpoint = f"{self.api_url}/v1/audio/speech"
            
            # Performance monitoring
            metrics = PerformanceMetrics()
            metrics.timestamp = time.time()
            
            # Make optimized request
            start_time = time.time()
            response = self.session.post(
                endpoint,
                json=data,
                timeout=self.streaming_timeout if use_streaming else self.synthesis_timeout,
                stream=use_streaming,
                headers={
                    'Connection': 'keep-alive',
                    'Accept': 'audio/mpeg',
                    'Cache-Control': 'no-cache'
                }
            )
            
            network_time = time.time() - start_time
            metrics.network_time = network_time
            
            if response.status_code == 200:
                # Process response with optimized buffering
                suffix = f".{data['response_format']}"
                audio_data = b""
                first_byte_received = False
                
                if use_streaming:
                    # Process streaming response with smaller chunks
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            if not first_byte_received:
                                first_byte_time = time.time() - start_time
                                metrics.first_byte_latency = first_byte_time
                                first_byte_received = True
                            audio_data += chunk
                else:
                    # Process regular response
                    audio_data = response.content
                    if audio_data:
                        first_byte_time = time.time() - start_time
                        metrics.first_byte_latency = first_byte_time
                
                # Save audio with immediate flush
                synthesis_start = time.time()
                temp_path = self._process_audio_buffer(audio_data)
                buffer_time = time.time() - synthesis_start
                metrics.buffer_time = buffer_time
                
                # Calculate total metrics
                total_time = time.time() - start_time
                metrics.total_latency = total_time
                metrics.synthesis_time = total_time - network_time - buffer_time
                
                # Record performance metrics
                self._record_metrics(metrics)
                
                # Log performance
                self.logger.info(f"⚡ PERFORMANCE: First byte: {metrics.first_byte_latency:.3f}s, Total: {total_time:.3f}s")
                
                if temp_path:
                    self.logger.info(f"✅ Audio synthesized: {temp_path}")
                    return temp_path
                else:
                    self.logger.error("❌ Failed to save audio buffer")
                    return None
            else:
                self.logger.error(f"❌ Synthesis error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Synthesis error: {e}")
            return None
    
    def synthesize_streaming(self, text: str, voice: Optional[str] = None, 
                           chunk_callback: Optional[Callable[[bytes], None]] = None) -> Dict[str, Any]:
        """
        Synthesize text with true streaming for ultra-low latency
        Calls chunk_callback for each audio chunk as it arrives
        """
        try:
            if not self.is_server_running():
                raise Exception("Kokoro server is not running")
            
            # Start timing
            start_time = time.time()
            first_chunk_time = None
            
            # Use provided voice or default
            selected_voice = voice or self.voice
            
            # Prepare request data
            data = {
                "input": text,
                "voice": selected_voice,
                "response_format": "mp3",
                "use_gpu": True,
                "speed": 1.0,
                "optimize_for_latency": True,
                "stream": True  # Enable streaming on server side
            }
            
            # Make streaming request
            response = self.session.post(
                f"{self.api_url}/v1/audio/speech/stream",
                json=data,
                timeout=self.streaming_timeout,
                stream=True,
                headers={'Connection': 'keep-alive'}
            )
            
            if response.status_code != 200:
                error_msg = f"Streaming synthesis failed: {response.status_code}"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'latency': time.time() - start_time
                }
            
            # Process streaming response
            audio_chunks = []
            total_bytes = 0
            
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    # Record first chunk time
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        first_byte_latency = first_chunk_time - start_time
                        self.logger.info(f"⚡ First audio chunk received in {first_byte_latency*1000:.1f}ms")
                    
                    # Call chunk callback if provided
                    if chunk_callback:
                        chunk_callback(chunk)
                    
                    audio_chunks.append(chunk)
                    total_bytes += len(chunk)
            
            # Calculate metrics
            total_time = time.time() - start_time
            metrics = PerformanceMetrics(
                first_byte_latency=first_byte_latency if first_chunk_time else total_time,
                total_latency=total_time,
                synthesis_time=total_time,
                network_time=0.0,
                buffer_time=0.0,
                timestamp=time.time()
            )
            
            self._record_metrics(metrics)
            
            # Combine chunks
            audio_data = b''.join(audio_chunks)
            
            self.logger.info(f"✅ Streaming synthesis completed in {total_time:.3f}s ({total_bytes} bytes)")
            
            return {
                'success': True,
                'audio_data': audio_data,
                'latency': total_time,
                'first_byte_latency': first_byte_latency if first_chunk_time else total_time,
                'size': total_bytes
            }
            
        except Exception as e:
            self.logger.error(f"Streaming synthesis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'latency': time.time() - start_time
            }
    
    def synthesize_sync(self, 
                       text: str, 
                       voice: Optional[str] = None, 
                       use_gpu: bool = True, 
                       use_streaming: bool = None,
                       optimize_for_latency: bool = True,
                       **kwargs) -> Dict[str, Any]:
        """Synchronous synthesis with performance metrics"""
        try:
            start_time = time.time()
            audio_path = self.synthesize(text, voice, use_gpu, use_streaming, optimize_for_latency)
            total_time = time.time() - start_time
            
            if audio_path:
                return {
                    'success': True,
                    'audio_data': audio_path,
                    'error': None,
                    'format': 'mp3',
                    'latency': total_time,
                    'performance_stats': self._get_performance_stats()
                }
            else:
                return {
                    'success': False,
                    'audio_data': None,
                    'error': 'Synthesis failed',
                    'format': None,
                    'latency': total_time
                }
        except Exception as e:
            self.logger.error(f"❌ Sync synthesis error: {e}")
            return {
                'success': False,
                'audio_data': None,
                'error': str(e),
                'format': None
            }
    
    def synthesize_async(self, 
                        text: str, 
                        voice: Optional[str] = None, 
                        callback: Optional[Callable] = None,
                        optimize_for_latency: bool = True,
                        use_streaming: bool = True):
        """Asynchronous synthesis with ThreadPoolExecutor and streaming support"""
        
        def _synthesize_worker():
            try:
                start_time = time.time()
                
                if use_streaming and optimize_for_latency:
                    # Use streaming for ultra-low latency
                    self.logger.info("🚀 Using streaming synthesis for real-time performance")
                    
                    # Create a temporary file for streamed audio
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    
                    def chunk_callback(chunk):
                        temp_file.write(chunk)
                        temp_file.flush()
                    
                    result = self.synthesize_streaming(text, voice, chunk_callback)
                    temp_file.close()
                    
                    if result['success']:
                        audio_path = temp_file.name
                        total_time = result['latency']
                        
                        result = {
                            'success': True,
                            'audio_path': audio_path,
                            'audio_data': audio_path,
                            'error': None,
                            'latency': total_time,
                            'first_byte_latency': result.get('first_byte_latency', total_time)
                        }
                        
                        # Call callbacks
                        if self.on_audio_ready:
                            self.on_audio_ready(result)
                        if callback:
                            callback(audio_path, None)
                    else:
                        error_msg = result.get('error', 'Streaming synthesis failed')
                        if self.on_error:
                            self.on_error(error_msg)
                        if callback:
                            callback(None, error_msg)
                else:
                    # Use regular synthesis
                    audio_path = self.synthesize(text, voice, use_gpu=True, optimize_for_latency=optimize_for_latency)
                    total_time = time.time() - start_time
                    
                    if audio_path:
                        result = {
                            'success': True,
                            'audio_path': audio_path,
                            'audio_data': audio_path,
                            'error': None,
                            'latency': total_time
                        }
                        
                        # Call callbacks
                        if self.on_audio_ready:
                            self.on_audio_ready(result)
                        if callback:
                            callback(audio_path, None)
                    else:
                        error_msg = "Synthesis failed"
                        if self.on_error:
                            self.on_error(error_msg)
                        if callback:
                            callback(None, error_msg)
                        
            except Exception as e:
                error_msg = str(e)
                if self.on_error:
                    self.on_error(error_msg)
                if callback:
                    callback(None, error_msg)
        
        # Submit to thread pool
        self.executor.submit(_synthesize_worker)
    
    def synthesize_batch(self, 
                        texts: List[str], 
                        voice: Optional[str] = None,
                        optimize_for_latency: bool = True) -> List[Dict[str, Any]]:
        """Batch synthesis for multiple texts"""
        try:
            futures = []
            for text in texts:
                future = self.executor.submit(
                    self.synthesize_sync, 
                    text, 
                    voice, 
                    use_gpu=True, 
                    optimize_for_latency=optimize_for_latency
                )
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'success': False,
                        'error': str(e),
                        'audio_data': None
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Batch synthesis error: {e}")
            return []
    
    def set_voice(self, voice_id: str):
        """Set voice for synthesis"""
        old_voice = self.voice
        self.voice = voice_id
        self.logger.info(f"🎤 VOICE CHANGE: {old_voice} -> {voice_id}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        return self._get_performance_stats()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_history.clear()
        self.logger.info("📊 Performance stats reset")
    
    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(f"⏱️ {operation_name}: {duration:.3f}s")
    
    def cleanup(self):
        """Comprehensive resource cleanup"""
        try:
            # Stop buffer manager
            self.buffer_running = False
            if self.buffer_thread and self.buffer_thread.is_alive():
                self.buffer_thread.join(timeout=2)
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Close session
            self.session.close()
            
            # Stop server
            self.stop_server()
            
            # Clear buffers
            while not self.audio_buffer.empty():
                try:
                    self.audio_buffer.get_nowait()
                except queue.Empty:
                    break
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("🧹 Kokoro handler cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")
    
    def __del__(self):
        """Destructor for cleanup"""
        self.cleanup()

# Backward compatibility alias
KokoroHandler = OptimizedKokoroHandler