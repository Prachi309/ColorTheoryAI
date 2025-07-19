# Fix OpenMP conflict - MUST BE FIRST
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Suppress protobuf deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import argparse
import glob
import os
import os.path as osp
import random
from collections import Counter
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.filters import gaussian
import facer
import mediapipe as mp
from dotenv import load_dotenv
import logging
from functools import lru_cache
import gc
import psutil

load_dotenv() 

# Configure logging
logger = logging.getLogger(__name__)

api_key = os.getenv("API_KEY")

# Global instances to avoid reloading models
_face_detector = None
_face_parser = None
_device = None
_face_mesh = None

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def log_memory_usage(stage=""):
    """Log current memory usage with stage information"""
    memory_mb = get_memory_usage()
    logger.info(f"🧠 Memory Usage {stage}: {memory_mb:.1f} MB")
    return memory_mb

def get_models():
    """Get cached face detection and parsing models with quantization"""
    global _face_detector, _face_parser, _device
    if _face_detector is None:
        try:
            log_memory_usage("before face models load")
            
            _device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading face models on device: {_device}")
            
            # Load face detector
            _face_detector = facer.face_detector('retinaface/mobilenet', device=_device)
            
            # QUANTIZE FACE DETECTOR TO REDUCE MEMORY USAGE
            if hasattr(_face_detector, 'half'):
                logger.info("🔄 Quantizing face detector to FP16...")
                _face_detector = _face_detector.half()
            
            log_memory_usage("after face detector")
            
            # Load face parser
            _face_parser = facer.face_parser('farl/lapa/448', device=_device)
            
            # QUANTIZE FACE PARSER TO REDUCE MEMORY USAGE
            if hasattr(_face_parser, 'half'):
                logger.info("🔄 Quantizing face parser to FP16...")
                _face_parser = _face_parser.half()
            
            log_memory_usage("after face parser")
            
            logger.info("Face models loaded and quantized successfully")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            log_memory_usage("after face models cleanup")
            
        except Exception as e:
            logger.error(f"Error loading face models: {e}")
            raise
    return _face_detector, _face_parser, _device

def get_face_mesh():
    """Get cached MediaPipe face mesh model"""
    global _face_mesh
    if _face_mesh is None:
        try:
            log_memory_usage("before MediaPipe load")
            
            mp_face_mesh = mp.solutions.face_mesh
            _face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True, 
                max_num_faces=1, 
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            log_memory_usage("after MediaPipe load")
            
            logger.info("MediaPipe face mesh loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MediaPipe face mesh: {e}")
            raise
    return _face_mesh

def validate_image_path(image_path: str) -> None:
    """Validate image file exists and is readable"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")

def get_rgb_codes(path: str) -> np.ndarray:
    """Extract RGB codes from lip region with error handling and memory optimization"""
    try:
        validate_image_path(path)
        log_memory_usage("before RGB extraction")
        
        face_detector, face_parser, device = get_models()
        
        # Load image with memory optimization
        image = facer.hwc2bchw(facer.read_hwc(path)).to(device=device)
        
        # Convert to half precision if using GPU
        if device == 'cuda' and image.dtype == torch.float32:
            image = image.half()
        
        log_memory_usage("after image loading")
        
        with torch.inference_mode():
            faces = face_detector(image)
            if len(faces) == 0:
                raise ValueError("No faces detected in image")
            
            faces = face_parser(image, faces)

        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)
        seg_probs = seg_probs.cpu() 

        tensor = seg_probs.permute(0, 2, 3, 1)
        tensor = tensor.squeeze().numpy()

        # Extract lip regions
        llip = tensor[:, :, 7]
        ulip = tensor[:,:,9]
        lips = llip + ulip
        binary_mask = (lips >= 0.5).astype(int)

        # Read and process image
        sample = cv2.imread(path)
        if sample is None:
            raise ValueError(f"Could not read image: {path}")
            
        img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        # Extract RGB codes from lip region
        indices = np.argwhere(binary_mask)
        if len(indices) == 0:
            raise ValueError("No lip region detected")
            
        rgb_codes = img[indices[:, 0], indices[:, 1], :]
        
        # Cleanup GPU memory
        del image, faces, seg_logits, seg_probs, tensor, sample, img
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        log_memory_usage("after RGB extraction cleanup")
            
        return rgb_codes
        
    except Exception as e:
        logger.error(f"Error in get_rgb_codes: {e}")
        raise

def filter_lip_random(rgb_codes,randomNum=40):
    blue_condition = (rgb_codes[:, 2] <= 227)
    red_condition = (rgb_codes[:, 0] >= 97)
    filtered_rgb_codes = rgb_codes[blue_condition & red_condition]
    # Deterministic sampling
    step = max(1, filtered_rgb_codes.shape[0] // randomNum)
    indices = np.arange(0, filtered_rgb_codes.shape[0], step)[:randomNum]
    random_rgb_codes = filtered_rgb_codes[indices]
    return random_rgb_codes


def calc_dis(rgb_codes):
    spring = [[253,183,169],[247,98,77],[186,33,33]]
    summer = [[243,184,202],[211,118,155],[147,70,105]]
    autum = [[210,124,110],[155,70,60],[97,16,28]]
    winter = [[237,223,227],[177,47,57],[98,14,37]]
  
    res = []
    for i in range(len(rgb_codes)):
      sp = np.inf
      su = np.inf
      au = np.inf
      win = np.inf
      for j in range(3):
        sp = min(sp, np.linalg.norm(rgb_codes[i] - spring[j]))
        su = min(su, np.linalg.norm(rgb_codes[i]- summer[j]))
        au = min(au, np.linalg.norm(rgb_codes[i] - autum[j]))
        win = min(win, np.linalg.norm(rgb_codes[i] - winter[j]))
    
      min_type = min(sp, su, au, win)
      if min_type == sp:
        ctype = "sp"
      elif min_type == su:
        ctype = "su"
      elif min_type == au:
        ctype = "au"
      elif min_type == win:
        ctype = "win"
    
      res.append(ctype)
    return res


def save_skin_mask(img_path: str) -> None:
    """Save skin mask with improved error handling and memory optimization"""
    try:
        validate_image_path(img_path)
        log_memory_usage("before skin mask processing")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load image with memory optimization
        image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)
        
        # Convert to half precision if using GPU
        if device == 'cuda' and image.dtype == torch.float32:
            image = image.half()
        
        log_memory_usage("after skin image loading")
        
        face_detector = facer.face_detector('retinaface/mobilenet', device=device)

        with torch.inference_mode():
            faces = face_detector(image)
            if len(faces) == 0:
                raise ValueError("No faces detected for skin analysis")

        log_memory_usage("after face detection")
        
        face_parser = facer.face_parser('farl/lapa/448', device=device)
        with torch.inference_mode():
            faces = face_parser(image, faces)

        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)
        seg_probs = seg_probs.cpu() 
        tensor = seg_probs.permute(0, 2, 3, 1)
        tensor = tensor.squeeze().numpy()

        # Extract skin region
        face_skin = tensor[:, :, 1]
        binary_mask = (face_skin >= 0.5).astype(int)

        # Process and save masked image
        sample = cv2.imread(img_path)
        if sample is None:
            raise ValueError(f"Could not read image: {img_path}")
            
        img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        masked_image = np.zeros_like(img) 
        
        try: 
            masked_image[binary_mask == 1] = img[binary_mask == 1] 
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("temp.jpg", masked_image)
        except Exception as e:
            logger.error(f"Error saving skin mask: {e}")
            raise
            
        # Cleanup GPU memory
        del image, faces, seg_logits, seg_probs, tensor, sample, img, masked_image
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        log_memory_usage("after skin mask cleanup")
            
    except Exception as e:
        logger.error(f"Error in save_skin_mask: {e}")
        raise


def get_eye_color(image_path):
    import cv2
    import numpy as np
    from scipy import stats

    face_mesh = get_face_mesh()
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    LEFT_IRIS = [468, 469, 470, 471]

    def isolate_region(image, landmarks, indices):
        h, w = image.shape[:2]
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        region = cv2.bitwise_and(image, image, mask=mask)
        return region, mask

    def classify_eye_color(hue, saturation, value, rgb=None):
        if rgb is not None:
            r, g, b = rgb
            if max(r, g, b) < 90 and abs(r - g) < 15 and abs(g - b) < 15 and abs(b - r) < 15:
                return "Black"
        if value < 40:
            return "Black"
        if (hue <= 15 or hue >= 165) and value < 65:
            return "Black"
        if value > 200 and saturation < 30:
            return "Gray"
        if 10 < hue <= 25 and saturation > 50:
            return "Amber"
        if 25 < hue <= 45:
            return "Hazel"
        if 45 < hue <= 85:
            return "Green"
        if 85 < hue <= 130:
            return "Blue"
        if hue <= 10 or hue >= 160:
            return "Brown"
        if 140 <= hue <= 160 and saturation < 50:
            return "Violet"
        return "Unknown"

    def get_filtered_dominant_color(image, mask):
        pixels = image[mask == 255]
        if len(pixels) == 0:
            return "Unknown", (0, 0, 0)
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        filtered = hsv[(hsv[:, 1] > 30) & (hsv[:, 2] > 50) & (hsv[:, 2] < 230)]
        if len(filtered) == 0:
            return "Unknown", (0, 0, 0)
        hue_mode = int(stats.mode(filtered[:, 0], keepdims=False)[0])
        sat_avg = int(np.mean(filtered[:, 1]))
        val_avg = int(np.mean(filtered[:, 2]))
        rgb = cv2.cvtColor(np.uint8([[[hue_mode, sat_avg, val_avg]]]), cv2.COLOR_HSV2BGR)[0][0]
        rgb_tuple = tuple(int(c) for c in rgb)
        color_name = classify_eye_color(hue_mode, sat_avg, val_avg, rgb=rgb_tuple)
        return color_name, rgb_tuple

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        _, left_iris_mask = isolate_region(image, face_landmarks.landmark, LEFT_IRIS)
        color_name, rgb_tuple = get_filtered_dominant_color(image, left_iris_mask)
        return {"rgb": rgb_tuple, "color": color_name}
    else:
        return None


def analyze_lip_color(image_path):
    rgb_codes = get_rgb_codes(image_path)
    if rgb_codes is None or len(rgb_codes) == 0:
        return {"error": "No lip region detected"}

    # k-means for dominant color
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for k-means clustering. Please add it to requirements.txt.")
    kmeans = KMeans(n_clusters=1, random_state=42).fit(rgb_codes)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    # Converting to Python int for JSON serialization
    dominant_color_rgb = tuple(int(x) for x in dominant_color)
    dominant_color_hex = '#%02x%02x%02x' % dominant_color_rgb

    # Seasonal classification
    filtered = filter_lip_random(rgb_codes, 40)
    types = Counter(calc_dis(filtered))
    season = max(types, key=types.get)

    return {
        "dominant_color_rgb": dominant_color_rgb,
        "dominant_color_hex": dominant_color_hex,
        "season": season
    }


def analyze_skin_color(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu()
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()
    face_skin = tensor[:, :, 1]
    binary_mask = (face_skin >= 0.5).astype(int)
    sample = cv2.imread(image_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    skin_pixels = img[binary_mask == 1]
    if skin_pixels is None or len(skin_pixels) == 0:
        return {"error": "No skin region detected"}
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for k-means clustering. Please add it to requirements.txt.")
    kmeans = KMeans(n_clusters=1, random_state=42).fit(skin_pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    dominant_color_rgb = tuple(int(x) for x in dominant_color)
    dominant_color_hex = '#%02x%02x%02x' % dominant_color_rgb
    return {
        "dominant_color_rgb": dominant_color_rgb,
        "dominant_color_hex": dominant_color_hex
    }


def analyze_hair_color(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu()
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()
    print('Segmentation tensor shape:', tensor.shape)
    num_classes = tensor.shape[2] if tensor.ndim == 3 else 0
    print('Number of available classes:', num_classes)
    # Accessing all classes and printing their mean mask value
    for idx in range(num_classes):
        mask = tensor[:, :, idx]
        print(f'Class {idx} mean mask value:', mask.mean())
    # trying the last class as a guess for hair
    hair_mask = tensor[:, :, -1]
    binary_mask = (hair_mask >= 0.5).astype(int)
    sample = cv2.imread(image_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    hair_pixels = img[binary_mask == 1]
    if hair_pixels is None or len(hair_pixels) == 0:
        return {"error": "No hair region detected"}
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for k-means clustering. Please add it to requirements.txt.")
    kmeans = KMeans(n_clusters=1, random_state=42).fit(hair_pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    dominant_color_rgb = tuple(int(x) for x in dominant_color)
    dominant_color_hex = '#%02x%02x%02x' % dominant_color_rgb
    return {
        "dominant_color_rgb": dominant_color_rgb,
        "dominant_color_hex": dominant_color_hex
    }

    import requests

    def get_palette_from_llm(prompt, api_key):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistralai/mistral-small-3.2-24b-instruct:free",  
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data)
        print("OpenRouter response:", response.status_code, response.text)  
        return response.json()


def get_style_recommendation_from_llm(answers, api_key):
    """
    Given a dict of answers (with keys: dressing_focus, gender, body_type, context_answer),
    build a prompt, call OpenRouter, and return the suggestion string.
    """
    dressing_focus = answers.get("dressing_focus", "")
    gender = answers.get("gender", "")
    body_type = answers.get("body_type", "")
    context_answer = answers.get("context_answer", "")
    prompt = f"""
Recommend 3 items each of:
- Clothing
- Footwear
- Accessories (if suitable)
- Makeup products (foundation, lipstick, blush)
- Fragrance

For a user with:
Dressing focus: {dressing_focus}
Gender: {gender}
Body type: {body_type}
Context: {context_answer}

Return as JSON array only:
[
  {{
    "category": "...",
    "product": "...",
    "description": "...",
    "query": "search keywords only for finding the product"
  }},
  ...
]
"""
    llm_response = get_palette_from_llm(prompt, api_key)
    try:
        suggestion = llm_response['choices'][0]['message']['content']
    except Exception as e:
        suggestion = str(llm_response)
    return suggestion