#!/usr/bin/env python3
"""
Minimal startup script for Render deployment
Only loads essential components to stay under 512MB
"""

import os
import sys
import gc

# Set environment variables for memory optimization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    
    # Clear module cache for heavy libraries
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if any(x in module_name for x in ['torch', 'cv2', 'sklearn', 'matplotlib', 'numpy']):
            modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

if __name__ == "__main__":
    print(f"🧠 Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Cleanup before starting
    cleanup_memory()
    
    print(f"🧠 Memory after cleanup: {get_memory_usage():.1f} MB")
    print("✅ Ready to start server with minimal memory footprint")
    
    # Start the server
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 