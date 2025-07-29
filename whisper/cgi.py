#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parche temporal para el módulo 'cgi' removido en Python 3.13
Contiene solo las funciones básicas necesarias para compatibilidad
"""

import urllib.parse
from typing import Dict, List, Union, Optional

def parse_qs(qs: str, keep_blank_values: bool = False, strict_parsing: bool = False, 
             encoding: str = 'utf-8', errors: str = 'replace', 
             max_num_fields: Optional[int] = None, separator: str = '&') -> Dict[str, List[str]]:
    """Parse a query string given as a string argument."""
    return urllib.parse.parse_qs(qs, keep_blank_values, strict_parsing, 
                                 encoding, errors, max_num_fields, separator)

def parse_qsl(qs: str, keep_blank_values: bool = False, strict_parsing: bool = False,
              encoding: str = 'utf-8', errors: str = 'replace',
              max_num_fields: Optional[int] = None, separator: str = '&') -> List[tuple]:
    """Parse a query string given as a string argument."""
    return urllib.parse.parse_qsl(qs, keep_blank_values, strict_parsing,
                                  encoding, errors, max_num_fields, separator)

def escape(s: str, quote: bool = True) -> str:
    """Replace special characters with HTML entities."""
    import html
    return html.escape(s, quote)

def unescape(s: str) -> str:
    """Convert HTML entities back to their corresponding characters."""
    import html
    return html.unescape(s)

# Clase FieldStorage básica para compatibilidad
class FieldStorage:
    """Clase básica para compatibilidad con FieldStorage de cgi"""
    
    def __init__(self, fp=None, headers=None, outerboundary=b'',
                 environ=None, keep_blank_values=0, strict_parsing=0,
                 limit=None, encoding='utf-8', errors='replace',
                 max_num_fields=None, separator='&'):
        self.name = None
        self.filename = None
        self.value = None
        self.file = None
        self.type = None
        self.type_options = {}
        self.disposition = None
        self.disposition_options = {}
        self.headers = headers or {}
        
    def __getitem__(self, key):
        return self.value
        
    def getvalue(self, key, default=None):
        return self.value if self.value is not None else default
        
    def getfirst(self, key, default=None):
        return self.getvalue(key, default)
        
    def getlist(self, key):
        value = self.getvalue(key)
        return [value] if value is not None else []

# Funciones adicionales para compatibilidad
def print_exception(type=None, value=None, tb=None, limit=None, file=None):
    """Print exception information to file."""
    import traceback
    traceback.print_exception(type, value, tb, limit, file)

def print_environ(environ=None):
    """Print environment variables."""
    import os
    if environ is None:
        environ = os.environ
    for key, value in environ.items():
        print(f"{key}={value}")

def print_form(form):
    """Print form data."""
    for key in form:
        print(f"{key}: {form[key]}")

def print_directory():
    """Print current directory listing."""
    import os
    for item in os.listdir('.'):
        print(item)

def print_arguments():
    """Print command line arguments."""
    import sys
    for arg in sys.argv:
        print(arg)

# Constantes para compatibilidad
maxlen = 0