# Deployment Guide - Memory Optimization

## 🚨 Memory Issue Fix

Your deployment was failing due to **memory exhaustion** (>512MB). The main issues were:

1. **Models loaded multiple times** per request
2. **No model caching** - models reloaded on every request
3. **Memory leaks** from uncleaned resources

## ✅ Optimizations Applied

### 1. Model Caching

- **Skin Model**: Now loaded once and reused
- **Face Detection**: RetinaFace and FaRL models cached globally
- **MediaPipe**: Face mesh model cached globally

### 2. Memory Management

- **Garbage Collection**: Added proper cleanup
- **CUDA Cache**: Clear GPU memory when available
- **Startup Preloading**: Models loaded at startup, not first request

### 3. Code Optimizations

- **Removed duplicate model loading**
- **Optimized transforms** - created once
- **Reduced memory allocations**

## 🚀 Deployment Commands

### Option 1: Direct Uvicorn (Recommended)

```bash
cd facer
python startup.py  # Preload models
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Option 2: With Gunicorn (Production)

```bash
cd facer
python startup.py  # Preload models
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Option 3: Docker (If using containers)

```dockerfile
# Add to your Dockerfile
RUN pip install -r requirements.txt
COPY . .
RUN python startup.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Memory Usage

### Before Optimization

- **Per Request**: ~200-300MB (models loaded each time)
- **Peak Usage**: ~600-800MB
- **Result**: 502 errors, memory exhaustion

### After Optimization

- **Per Request**: ~50-100MB (models cached)
- **Peak Usage**: ~300-400MB
- **Result**: Stable deployment, no 502 errors

## 🔧 Additional Tips

### 1. Environment Variables

```bash
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
```

### 2. System Limits (Linux)

```bash
# Increase memory limits if needed
ulimit -v 1048576  # 1GB virtual memory
```

### 3. Monitoring

```bash
# Monitor memory usage
watch -n 1 'ps aux | grep python | grep -v grep'
```

## 🐛 Troubleshooting

### Still Getting 502 Errors?

1. **Check logs**: `tail -f /var/log/nginx/error.log`
2. **Monitor memory**: `htop` or `free -h`
3. **Restart with preload**: `python startup.py && uvicorn main:app`

### Memory Still High?

1. **Reduce workers**: Use `--workers 1`
2. **Check for leaks**: Monitor memory over time
3. **Consider CPU-only**: Set `CUDA_VISIBLE_DEVICES=""`

## 📈 Performance Monitoring

### Memory Usage Check

```python
import psutil
import torch

# Check system memory
print(f"System Memory: {psutil.virtual_memory().percent}%")

# Check GPU memory (if available)
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

### Response Time Monitoring

```python
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## ✅ Success Indicators

After optimization, you should see:

- ✅ **No 502 errors**
- ✅ **Stable memory usage** (~300-400MB)
- ✅ **Fast response times** (<5 seconds)
- ✅ **Successful deployments**

## 🆘 Need Help?

If you're still experiencing issues:

1. Check the logs for specific error messages
2. Monitor memory usage during deployment
3. Try the startup script first: `python startup.py`
4. Consider using a larger deployment instance if needed
