"""
Startup script to preload models and reduce memory usage during deployment.
Run this before starting the main server to warm up the models.
"""

import os
import sys
import torch
import gc

def preload_models():
    """Preload all models to avoid memory spikes during first request"""
    print("🔄 Preloading models...")
    
    try:
        # Import and preload skin model
        import skin_model
        print("✅ Skin model loaded")
        
        # Import and preload face detection models
        import functions
        print("✅ Face detection models loaded")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("✅ All models preloaded successfully!")
        print(f"📊 Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB" if torch.cuda.is_available() else "📊 Using CPU")
        
    except Exception as e:
        print(f"❌ Error preloading models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    preload_models() 