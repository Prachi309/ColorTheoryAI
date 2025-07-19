# Fix OpenMP conflict - MUST BE FIRST
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Suppress protobuf deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import fastapi
import functions as f
import cv2
from PIL import Image
from collections import Counter
import numpy as np
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
import base64
import skin_model as m
import requests
import re
from fastapi import Query
from fastapi import Form
from fastapi import Body
import logging
import gc
import torch
import tempfile
import io
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Color Insight API", version="1.0.0")

# Get frontend URL from environment variable
frontend_url = os.getenv("VITE_FRONTEND_URL")
logger.info(f"Frontend URL for CORS: {frontend_url}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Response cache for repeated requests with memory limits
_response_cache = {}
_cache_max_size = 50  # Maximum number of cached responses
_cache_memory_limit = 100  # Maximum memory usage for cache in MB

def get_cache_memory_usage():
    """Get memory usage of cache in MB"""
    import sys
    total_size = 0
    for key, value in _response_cache.items():
        total_size += sys.getsizeof(key) + sys.getsizeof(value)
    return total_size / 1024 / 1024

def cleanup_cache():
    """Remove oldest cache entries if memory limit exceeded"""
    cache_memory = get_cache_memory_usage()
    if cache_memory > _cache_memory_limit or len(_response_cache) > _cache_max_size:
        logger.info(f"🧹 Cleaning cache (current: {cache_memory:.1f}MB, {len(_response_cache)} items)")
        
        # Remove oldest entries (simple FIFO)
        items_to_remove = len(_response_cache) - _cache_max_size + 10
        if items_to_remove > 0:
            keys_to_remove = list(_response_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del _response_cache[key]
        
        logger.info(f"✅ Cache cleaned (now: {get_cache_memory_usage():.1f}MB, {len(_response_cache)} items)")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def log_memory_usage(stage=""):
    """Log current memory usage with stage information"""
    memory_mb = get_memory_usage()
    cache_memory = get_cache_memory_usage()
    logger.info(f"🧠 Memory Usage {stage}: {memory_mb:.1f} MB (Cache: {cache_memory:.1f} MB)")
    return memory_mb

def compress_image(image_path: str, max_size: int = 1024, quality: int = 85) -> str:
    """
    Compress and resize image to reduce memory usage
    Args:
        image_path: Path to original image
        max_size: Maximum width/height (default 1024)
        quality: JPEG quality (1-100, default 85)
    Returns:
        Path to compressed image
    """
    try:
        # Open image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get original size
            original_width, original_height = img.size
            logger.info(f"Original image size: {original_width}x{original_height}")
            
            # Calculate new size (maintain aspect ratio)
            if original_width > max_size or original_height > max_size:
                if original_width > original_height:
                    new_width = max_size
                    new_height = int(original_height * max_size / original_width)
                else:
                    new_height = max_size
                    new_width = int(original_width * max_size / original_height)
                
                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized to: {new_width}x{new_height}")
            else:
                logger.info("Image already within size limits")
            
            # Create temporary file for compressed image
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            
            # Save compressed image
            img.save(temp_path, 'JPEG', quality=quality, optimize=True)
            
            # Get file sizes
            original_size = os.path.getsize(image_path)
            compressed_size = os.path.getsize(temp_path)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            logger.info(f"Compression: {original_size/1024:.1f}KB → {compressed_size/1024:.1f}KB ({compression_ratio:.1f}% reduction)")
            
            return temp_path
            
    except Exception as e:
        logger.error(f"Error compressing image: {e}")
        # Return original path if compression fails
        return image_path

def cleanup_temp_file(file_path: str) -> None:
    """Safely cleanup temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Could not remove temp file {file_path}: {e}")

@app.on_event("startup")
async def startup_event():
    """Startup without preloading models to avoid memory spikes"""
    try:
        initial_memory = log_memory_usage("at startup")
        
        logger.info("🚀 Starting with lazy model loading...")
        logger.info("📝 Models will be loaded on first request to save memory")
        
        # Don't preload models - let them load lazily
        # This prevents memory spikes during deployment
        
        final_memory = log_memory_usage("after startup")
        
        logger.info("✅ Application started successfully")
        logger.info(f"📊 Startup memory usage: {final_memory:.1f} MB")
        
        # Check if we're under 512MB limit
        if final_memory < 512:
            logger.info(f"✅ Memory usage ({final_memory:.1f} MB) is under 512MB limit!")
        else:
            logger.warning(f"⚠️ Memory usage ({final_memory:.1f} MB) is above 512MB limit")
            
    except Exception as e:
        logger.error(f"⚠️ Warning: Startup error: {e}")

@app.get("/")
async def root():
    memory_mb = get_memory_usage()
    return {
        "message": "Colorinsight Personal Color Analysis API", 
        "version": "1.0.0",
        "memory_usage_mb": round(memory_mb, 1),
        "memory_status": "✅ Under 512MB" if memory_mb < 512 else "⚠️ Above 512MB",
        "loading_strategy": "Lazy Loading (models load on first request)",
        "endpoints": ["/image", "/lip", "/skin", "/hair", "/eye", "/analyze_features", "/palette_llm", "/quiz_palette_llm"], 
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint that doesn't load models"""
    memory_mb = get_memory_usage()
    return {
        "status": "healthy",
        "memory_usage_mb": round(memory_mb, 1),
        "memory_status": "✅ Under 512MB" if memory_mb < 512 else "⚠️ Above 512MB",
        "models_loaded": "No (lazy loading enabled)"
    }

@app.get("/memory")
async def memory_status():
    """Get current memory usage"""
    memory_mb = get_memory_usage()
    return {
        "memory_usage_mb": round(memory_mb, 1),
        "memory_status": "✅ Under 512MB" if memory_mb < 512 else "⚠️ Above 512MB",
        "timestamp": "Current"
    }

@app.post("/image")
async def image(file: UploadFile = File(None)):
    try:
        initial_memory = log_memory_usage("before image processing")
        
        if file and file.filename:
            # File upload handling
            logger.info(f"Received file: {file.filename}")
            
            # Save original file
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
            
            # COMPRESS IMAGE TO REDUCE MEMORY USAGE
            logger.info("🔄 Compressing image for memory optimization...")
            compressed_path = compress_image("saved.jpg", max_size=1024, quality=85)
            log_memory_usage("after compression")
            
            # Process compressed image
            eye_color = f.get_eye_color(compressed_path)
            log_memory_usage("after eye color analysis")
            
            f.save_skin_mask(compressed_path)
            log_memory_usage("after skin mask")
            
            # Get season prediction
            ans = m.get_season("temp.jpg")
            log_memory_usage("after season prediction")
            
            # Cleanup files
            cleanup_temp_file("temp.jpg")
            cleanup_temp_file("saved.jpg")
            if compressed_path != "saved.jpg":  # Only delete if it's a different file
                cleanup_temp_file(compressed_path)
            
            # Map season numbers to names
            if ans == 3:
                ans += 1
            elif ans == 0:
                ans = 3

            season_names = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
            
            final_memory = log_memory_usage("after cleanup")
            memory_used = final_memory - initial_memory
            
            logger.info(f"📊 Memory used for this request: {memory_used:.1f} MB")
            
            return JSONResponse(content={
                "message": "complete",
                "result": ans,
                "season": season_names.get(ans, "Unknown"),
                "eye_color": eye_color,
                "memory_usage_mb": round(final_memory, 1),
                "memory_used_mb": round(memory_used, 1)
            })
            
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")
            
    except Exception as e:
        # Cleanup on error
        cleanup_temp_file("temp.jpg")
        cleanup_temp_file("saved.jpg")
        logger.error(f"Error in image analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/lip")
async def lip(file: UploadFile = File(None)):
    try:
        if file and file.filename:
            logger.info(f"Received file: {file.filename}")
            
            # Save original file
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
            
            # COMPRESS IMAGE TO REDUCE MEMORY USAGE
            logger.info("🔄 Compressing image for lip analysis...")
            compressed_path = compress_image("saved.jpg", max_size=1024, quality=85)
            
            # Process compressed image
            result = f.analyze_lip_color(compressed_path)
            
            # Cleanup files
            cleanup_temp_file("saved.jpg")
            if compressed_path != "saved.jpg":
                cleanup_temp_file(compressed_path)

            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])

            return JSONResponse(content={
                "message": "complete",
                "dominant_color_rgb": result["dominant_color_rgb"],
                "dominant_color_hex": result["dominant_color_hex"],
                "season": result["season"]
            })
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")
            
    except Exception as e:
        # Cleanup on error
        cleanup_temp_file("saved.jpg")
        logger.error(f"Error in lip analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/skin")
async def skin(file: UploadFile = File(None)):
    try:
        if file and file.filename:
            print(f"Received file: {file.filename}")
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")

        result = f.analyze_skin_color("saved.jpg")
        os.remove("saved.jpg")

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return JSONResponse(content={
            "message": "complete",
            "dominant_skin_color_rgb": result["dominant_color_rgb"],
            "dominant_skin_color_hex": result["dominant_color_hex"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/hair")
async def hair(file: UploadFile = File(None)):
    try:
        if file and file.filename:
            print(f"Received file: {file.filename}")
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")

        result = f.analyze_hair_color("saved.jpg")
        os.remove("saved.jpg")

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return JSONResponse(content={
            "message": "complete",
            "dominant_hair_color_rgb": result["dominant_color_rgb"],
            "dominant_hair_color_hex": result["dominant_color_hex"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/eye")
async def eye(file: UploadFile = File(None)):
    try:
        if file and file.filename:
            print(f"Received file: {file.filename}")
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")

        result = f.get_eye_color("saved.jpg")
        os.remove("saved.jpg")

        if result is None:
            raise HTTPException(status_code=400, detail="No eye region detected")

        rgb = result["rgb"]
        color_name = result["color"]
        hex_color = '#%02x%02x%02x' % rgb

        return JSONResponse(content={
            "message": "complete",
            "dominant_eye_color_rgb": rgb,
            "dominant_eye_color_hex": hex_color,
            "dominant_eye_color_name": color_name
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/analyze_features")
async def analyze_features(file: UploadFile = File(None)):
    try:
        if file and file.filename:
            print(f"Received file: {file.filename}")
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")

        # feature extraction functions for all features
        skin = f.analyze_skin_color("saved.jpg")
        hair = f.analyze_hair_color("saved.jpg")
        lips = f.analyze_lip_color("saved.jpg")
        eyes = f.get_eye_color("saved.jpg")
        os.remove("saved.jpg")

        
        return JSONResponse(content={
            "message": "complete",
            "skin": skin,
            "hair": hair,
            "lips": lips,
            "eyes": eyes
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def build_palette_prompt(features):
    return (
        f"Suggest a 4-color palette (in hex codes) for a person with:\n"
        f"- Skin color: {features['skin']['dominant_color_hex']}\n"
        f"- Hair color: {features['hair']['dominant_color_hex']}\n"
        f"- Lip color: {features['lips']['dominant_color_hex']}\n"
        f"- Eye color: {features['eyes']['dominant_color_hex']}\n"
        f"Output format: HEX codes only, in array."
    )

def get_palette_from_llm(prompt, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-small-3.2-24b-instruct:free",  #mistral model AI
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

@app.post("/palette_llm")
async def palette_llm(
    file: UploadFile = File(...),
    openrouter_api_key: str = Query(...),
    prompt: str = Form(None),
    season: str = Form(None)  
):
    try:
        if not openrouter_api_key:
            raise HTTPException(status_code=400, detail="API key required as query parameter.")

        if file and file.filename:
            print(f"Received file: {file.filename}")
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")

        # Extract features
        skin = f.analyze_skin_color("saved.jpg")
        hair = f.analyze_hair_color("saved.jpg")
        lips = f.analyze_lip_color("saved.jpg")
        eyes = f.get_eye_color("saved.jpg")
        os.remove("saved.jpg")

        features = {}
        if skin and isinstance(skin, dict) and "dominant_color_hex" in skin:
            features["skin"] = skin
        if hair and isinstance(hair, dict) and "dominant_color_hex" in hair:
            features["hair"] = hair
        if lips and isinstance(lips, dict) and "dominant_color_hex" in lips:
            features["lips"] = lips
        if eyes and isinstance(eyes, dict) and "dominant_color_hex" in eyes:
            features["eyes"] = eyes

        season_text = ""
        if season:
            season_text = f"\nThe best-matched season for this person (from our model) is: {season}. Consider this as a strong prior when generating the palette and analysis.\n"

        undertone_text = ""
        if prompt:
            undertone_text = f"\nUser undertone information: {prompt}\n"

        if not features:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract any valid features (skin, hair, lips, eyes) from the image."}
            )

        prompt_text = (
            "You are a professional color consultant. "
            "Given the following personal color analysis data, generate a personalized color palette and recommendations.\n\n"
            f"Detected season: {season}\n"
            f"Skin color (HEX): {features.get('skin', {}).get('dominant_color_hex', 'N/A')}\n"
            f"Hair color (HEX): {features.get('hair', {}).get('dominant_color_hex', 'N/A')}\n"
            f"Lip color (HEX): {features.get('lips', {}).get('dominant_color_hex', 'N/A')}\n"
            f"Eye color (HEX): {features.get('eyes', {}).get('dominant_color_hex', 'N/A')}\n"
            f"User undertone (if provided): {prompt if prompt else 'N/A'}\n\n"
            "Based on these determine the user's personal color season that and give a diverse range of palette colors that suits them and"
            "After that you have to give primary palette colors , warm tones , cool tones , neutral or black tones , clothing suggestions , makeup tips the format is specifies below"
            "Very Important:  Include shades from light to dark for each major color family suitable for that season \n"
            "(e.g., off-white to ivory, blush pink to rose, peach to burnt orange, mint to forest green, sky blue to navy, etc.)\n"
            "Analyze the features, determine the best season, and return your analysis in the following JSON format:\n"
            "Include primary color in primary and secondary colors in primary and seocndary  palettes, warm colrs in warm tones, cool colrs in cool tones, neutral or black colrs in neutral or black tones . very important use variety of shades for a diverse palette\n"
            "{\n"
            '  "season": "...",\n'
            '  "why": "...(Do not use Hex codes and asterics, Give very short and precise answer)",\n'
            '  "palettes": {\n'
            '    "Primary and Secondary Colors": ["- #HEX (Color Name)",  "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)",],\n'
            '    "Warm Tones": ["- #HEX (Color Name)",  "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)",],\n'
            '    "Cool Tones": ["- #HEX (Color Name)",  "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)",\n'
            '    "Neutral or Black Tones": ["- #HEX (Color Name)",  "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)", "- #HEX (Color Name)",\n'
            "  },\n"
            '  "More Colors": ["- #HEX (Color Name)", ...],\n'
            '  "makeup": [\n'
            '    { "part": "Lip", "hex": "...", "name": "..." },\n'
            '    { "part": "Eyes", "hex": "...", "name": "..." },\n'
            '    { "part": "Cheeks", "hex": "...", "name": "..." }\n'
            "  ]\n"
            "}\n"
            "Do NOT include markdown, code blocks, or any text outside the JSON.And Do not use terms like user looks good , use you when explaining hwy this season, use different shades od colors everywhere and not the shades that just match the user"
        )

        llm_response = get_palette_from_llm(prompt_text, openrouter_api_key)
        try:
            content = llm_response['choices'][0]['message']['content']
        except Exception as e:
            content = str(llm_response)

        return JSONResponse(content={
            "message": "complete",
            "llm_response": content,
            "features": features,
            "prompt": prompt_text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def build_quiz_palette_prompt(quiz_answers):
    return f"""
You are a color theory expert analyzing personal color palettes.

User's self-reported features:
- Skin undertone: {quiz_answers.get('undertone')}
- Natural hair color: {quiz_answers.get('hairColor')}
- Natural eye color: {quiz_answers.get('eyeColor')}
- Skin reaction to sun: {quiz_answers.get('sunReaction')}
- Skin depth/tone: {quiz_answers.get('skinDepth')}
- Vein color: {quiz_answers.get('veinColor')}

Based on these features, analyze the user's likely personal color season (Spring, Summer, Autumn, Winter) using the same logic as if you had hex codes. Match their traits to the seasonal color theory below.
User variety of shades for a diverse palette
...

Return ONLY a valid JSON object in this format:
{{
  "season": "...",
  "why": "...",
  "palettes": {{
    "Primary and Secondary Colors": ["- #HEX (Color Name)", ...],
    "Warm Tones": ["- #HEX (Color Name)", ...],
    "Cool Tones": ["- #HEX (Color Name)", ...],
    "Neutral or Black Tones": ["- #HEX (Color Name)", ...]
  }},
  "clothing": ["- #HEX (Color Name)", ...],
  "makeup": [
    {{ "part": "Lip", "hex": "...", "name": "..." }},
    {{ "part": "Eyes", "hex": "...", "name": "..." }},
    {{ "part": "Cheeks", "hex": "...", "name": "..." }}
  ]
}}
Do NOT include markdown, code blocks, or any text outside the JSON. Do not use hex codes and asterics in the why section. Do not use words like user, use you in statements
"""





@app.post("/quiz_palette_llm")
async def quiz_palette_llm(
    quiz_answers: dict = Body(...),
    openrouter_api_key: str = Query(...)
):
    try:
        if not openrouter_api_key:
            raise HTTPException(status_code=400, detail="API key required as query parameter.")

        
        prompt_text = build_quiz_palette_prompt(quiz_answers)

        llm_response = get_palette_from_llm(prompt_text, openrouter_api_key)
        try:
            content = llm_response['choices'][0]['message']['content']
        except Exception as e:
            content = str(llm_response)

        return JSONResponse(content={
            "message": "complete",
            "llm_response": content,
            "quiz_answers": quiz_answers,
            "prompt": prompt_text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/style-recommendation")
async def style_recommendation(request: Request, openrouter_api_key: str = Query(...)):
    data = await request.json()
    
    suggestion = f.get_style_recommendation_from_llm(data, openrouter_api_key)
    return JSONResponse({"suggestion": suggestion})



@app.get("/api/serpapi-proxy")
def serpapi_proxy(q: str):
    url = f"https://serpapi.com/search.json?q={q}&tbm=isch&api_key={SERPAPI_KEY}"
    print(f"[SerpAPI Proxy] Requesting: {url}")
    try:
        resp = requests.get(url)
        print(f"[SerpAPI Proxy] Response status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[SerpAPI Proxy] Error response: {resp.text}")
        return JSONResponse(resp.json())
    except Exception as e:
        print(f"[SerpAPI Proxy] Exception: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
