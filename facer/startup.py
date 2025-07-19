"""
Startup script to preload models and reduce memory usage during deployment.
Run this before starting the main server to warm up the models.
"""

# Fix OpenMP conflict - MUST BE FIRST
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import functions as f
import skin_model as m
import gc
import torch

print("🔄 Preloading models for memory optimization...")

try:
    # Preload all models
    f.get_models()
    m.get_model()
    f.get_face_mesh()
    
    # Cleanup memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✅ Models preloaded successfully")
    print("✅ Memory optimized and ready for deployment")
    
except Exception as e:
    print(f"⚠️ Warning: Could not preload all models: {e}")
    print("⚠️ Application will continue but may be slower on first request") 