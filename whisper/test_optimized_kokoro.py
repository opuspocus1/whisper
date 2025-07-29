#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for optimized KokoroHandler with sub-100ms latency
"""

import time
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kokoro_handler import OptimizedKokoroHandler

def setup_logging():
    """Setup logging for performance monitoring"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('kokoro_performance_test.log', encoding='utf-8')
        ]
    )

def test_latency_optimization():
    """Test the optimized KokoroHandler for sub-100ms latency"""
    
    print("🚀 Testing Optimized KokoroHandler for Sub-100ms Latency")
    print("=" * 60)
    
    # Initialize optimized handler
    handler = OptimizedKokoroHandler(
        api_url="http://localhost:5000",
        voice="af_heart",
        max_workers=4,
        connection_pool_size=10,
        chunk_size=512,
        pre_warm_models=True
    )
    
    # Set up callbacks
    def on_audio_ready(result):
        print(f"✅ Audio ready: {result.get('latency', 0):.3f}s")
    
    def on_error(error):
        print(f"❌ Error: {error}")
    
    handler.on_audio_ready = on_audio_ready
    handler.on_error = on_error
    
    try:
        # Test 1: Short text (should use regular endpoint)
        print("\n📝 Test 1: Short text synthesis")
        short_text = "Hello world"
        
        start_time = time.time()
        result = handler.synthesize_sync(short_text, optimize_for_latency=True)
        total_time = time.time() - start_time
        
        print(f"   Text: '{short_text}'")
        print(f"   Success: {result['success']}")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Latency: {result.get('latency', 0):.3f}s")
        
        # Test 2: Medium text (should use streaming)
        print("\n📝 Test 2: Medium text synthesis (streaming)")
        medium_text = "This is a medium length text that should trigger streaming mode for better performance and lower latency."
        
        start_time = time.time()
        result = handler.synthesize_sync(medium_text, optimize_for_latency=True)
        total_time = time.time() - start_time
        
        print(f"   Text: '{medium_text[:50]}...'")
        print(f"   Success: {result['success']}")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Latency: {result.get('latency', 0):.3f}s")
        
        # Test 3: Async synthesis
        print("\n📝 Test 3: Async synthesis")
        async_text = "This is an async test for real-time applications."
        
        start_time = time.time()
        handler.synthesize_async(async_text, optimize_for_latency=True)
        
        # Wait a bit for async completion
        time.sleep(2)
        total_time = time.time() - start_time
        print(f"   Async synthesis initiated in {total_time:.3f}s")
        
        # Test 4: Batch synthesis
        print("\n📝 Test 4: Batch synthesis")
        batch_texts = [
            "First batch item",
            "Second batch item", 
            "Third batch item",
            "Fourth batch item"
        ]
        
        start_time = time.time()
        results = handler.synthesize_batch(batch_texts, optimize_for_latency=True)
        total_time = time.time() - start_time
        
        print(f"   Batch size: {len(batch_texts)}")
        print(f"   Success count: {sum(1 for r in results if r['success'])}")
        print(f"   Total batch time: {total_time:.3f}s")
        print(f"   Average per item: {total_time/len(batch_texts):.3f}s")
        
        # Test 5: Performance statistics
        print("\n📊 Test 5: Performance Statistics")
        stats = handler.get_performance_stats()
        
        if stats:
            print(f"   Total requests: {stats.get('total_requests', 0)}")
            print(f"   Avg first-byte latency: {stats.get('avg_first_byte_latency', 0):.3f}s")
            print(f"   Min first-byte latency: {stats.get('min_first_byte_latency', 0):.3f}s")
            print(f"   Max first-byte latency: {stats.get('max_first_byte_latency', 0):.3f}s")
            print(f"   95th percentile: {stats.get('p95_first_byte_latency', 0):.3f}s")
            print(f"   99th percentile: {stats.get('p99_first_byte_latency', 0):.3f}s")
            
            # Check if we achieved sub-100ms latency
            avg_latency = stats.get('avg_first_byte_latency', 0)
            if avg_latency < 0.1:
                print("   ✅ ACHIEVED: Sub-100ms average latency!")
            else:
                print(f"   ⚠️ Target not met: {avg_latency*1000:.1f}ms average")
        else:
            print("   No performance data available")
        
        # Test 6: Stress test
        print("\n🔥 Test 6: Stress test (10 rapid requests)")
        stress_texts = [f"Stress test {i}" for i in range(10)]
        
        start_time = time.time()
        stress_results = []
        
        for i, text in enumerate(stress_texts):
            req_start = time.time()
            result = handler.synthesize_sync(text, optimize_for_latency=True)
            req_time = time.time() - req_start
            stress_results.append(req_time)
            print(f"   Request {i+1}: {req_time:.3f}s")
        
        total_stress_time = time.time() - start_time
        avg_stress_time = sum(stress_results) / len(stress_results)
        
        print(f"   Total stress time: {total_stress_time:.3f}s")
        print(f"   Average per request: {avg_stress_time:.3f}s")
        print(f"   Requests per second: {len(stress_texts)/total_stress_time:.1f}")
        
        # Final performance summary
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        final_stats = handler.get_performance_stats()
        if final_stats:
            print(f"Total API calls: {final_stats.get('total_requests', 0)}")
            print(f"Average first-byte latency: {final_stats.get('avg_first_byte_latency', 0)*1000:.1f}ms")
            print(f"95th percentile latency: {final_stats.get('p95_first_byte_latency', 0)*1000:.1f}ms")
            print(f"99th percentile latency: {final_stats.get('p99_first_byte_latency', 0)*1000:.1f}ms")
            
            # Performance grade
            avg_latency = final_stats.get('avg_first_byte_latency', 0)
            if avg_latency < 0.05:
                grade = "A+ (Excellent)"
            elif avg_latency < 0.1:
                grade = "A (Great)"
            elif avg_latency < 0.2:
                grade = "B (Good)"
            elif avg_latency < 0.5:
                grade = "C (Acceptable)"
            else:
                grade = "D (Needs improvement)"
            
            print(f"Performance Grade: {grade}")
        
        print("\n✅ Optimization test completed!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        logging.error(f"Test error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up resources...")
        handler.cleanup()
        print("✅ Cleanup completed")

def test_connection_pooling():
    """Test connection pooling performance"""
    print("\n🔗 Testing Connection Pooling")
    print("-" * 40)
    
    handler = OptimizedKokoroHandler(
        connection_pool_size=5,
        pre_warm_models=False  # Disable for this test
    )
    
    try:
        # Test multiple rapid requests to see connection reuse
        test_texts = [f"Connection test {i}" for i in range(5)]
        
        start_time = time.time()
        for i, text in enumerate(test_texts):
            req_start = time.time()
            result = handler.synthesize_sync(text, optimize_for_latency=True)
            req_time = time.time() - req_start
            print(f"   Request {i+1}: {req_time:.3f}s")
        
        total_time = time.time() - start_time
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average: {total_time/len(test_texts):.3f}s")
        
    except Exception as e:
        print(f"❌ Connection pooling test error: {e}")
    finally:
        handler.cleanup()

if __name__ == "__main__":
    setup_logging()
    
    print("🎤 Optimized KokoroHandler Performance Test")
    print("=" * 60)
    
    # Check if server is running
    test_handler = OptimizedKokoroHandler(pre_warm_models=False)
    if not test_handler.is_server_running():
        print("⚠️ Kokoro server not running. Starting server...")
        if not test_handler.start_server():
            print("❌ Failed to start Kokoro server. Please ensure the server is running.")
            sys.exit(1)
        test_handler.cleanup()
    
    # Run tests
    test_latency_optimization()
    test_connection_pooling()
    
    print("\n🎉 All tests completed!") 