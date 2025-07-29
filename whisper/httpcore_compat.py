# -*- coding: utf-8 -*-
"""
Compatibilidad temporal para httpcore
Soluciona problemas de atributos faltantes en versiones incompatibles
"""

import httpcore

# Agregar atributos faltantes si no existen
if not hasattr(httpcore, 'SyncHTTPTransport'):
    class SyncHTTPTransport:
        """Clase de compatibilidad para SyncHTTPTransport"""
        def __init__(self, *args, **kwargs):
            pass
        
        def handle_request(self, request):
            # Implementación básica de compatibilidad
            return None
    
    httpcore.SyncHTTPTransport = SyncHTTPTransport

if not hasattr(httpcore, 'AsyncHTTPTransport'):
    class AsyncHTTPTransport:
        """Clase de compatibilidad para AsyncHTTPTransport"""
        def __init__(self, *args, **kwargs):
            pass
        
        async def handle_async_request(self, request):
            # Implementación básica de compatibilidad
            return None
    
    httpcore.AsyncHTTPTransport = AsyncHTTPTransport

if not hasattr(httpcore, 'BaseTransport'):
    class BaseTransport:
        """Clase base de compatibilidad para Transport"""
        def __init__(self, *args, **kwargs):
            pass
    
    httpcore.BaseTransport = BaseTransport

print("httpcore_compat: Parches de compatibilidad aplicados")