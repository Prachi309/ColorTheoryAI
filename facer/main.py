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


app = FastAPI()


# Get frontend URL from environment variable
frontend_url = os.getenv("VITE_FRONTEND_URL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_url,
        "https://color-theory-ai-prachis-projects-c52565cb.vercel.app",
        "https://color-theory-ai-git-main-prachis-projects-c52565cb.vercel.app"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
@app.get("/")
async def root():
    return {"message": "Colorinsight Personal Color Analysis API", "endpoints": ["/image", "/lip"], "docs": "/docs"}



@app.post("/image")
async def image(file: UploadFile = File(None)):

    try:
        if file and file.filename:
            # File upload handling
            print(f"Received file: {file.filename}")
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")
      
       
        eye_color = f.get_eye_color("saved.jpg")
      
        f.save_skin_mask("saved.jpg")
   
        ans = m.get_season("temp.jpg")
        os.remove("temp.jpg")
        os.remove("saved.jpg")
   
        if ans == 3:
            ans += 1
        elif ans == 0:
            ans = 3

        # Returning result directly 
        season_names = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
        return JSONResponse(content={
            "message": "complete",
            "result": ans,
            "season": season_names.get(ans, "Unknown"),
            "eye_color": eye_color  
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/lip")
async def lip(file: UploadFile = File(None)):
    try:
        if file and file.filename:
            print(f"Received file: {file.filename}")
            with open("saved.jpg", "wb") as fi:
                content = await file.read()
                fi.write(content)
        else:
            raise HTTPException(status_code=400, detail="No image file provided. Please upload an image file.")

        # lip color analysis function
        result = f.analyze_lip_color("saved.jpg")
        os.remove("saved.jpg")

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return JSONResponse(content={
            "message": "complete",
            "dominant_color_rgb": result["dominant_color_rgb"],
            "dominant_color_hex": result["dominant_color_hex"],
            "season": result["season"]
        })
    except Exception as e:
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
