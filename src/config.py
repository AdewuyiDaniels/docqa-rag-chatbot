"""
Configuration settings and constants for the application.
"""

import os

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retriever
TOP_K_RESULTS = 4

# File Handling
UPLOAD_DIR = "data/uploads"
VECTOR_STORE_DIR = "data/vector_store"
ALLOWED_EXTENSIONS = [".pdf", ".docx"]

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)