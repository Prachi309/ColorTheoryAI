pip install git+https://github.com/serpapi/serpapi-python.git

# ColorAI: Personal Color Analysis App

ColorAI is a modern web application that uses AI-powered color analysis to help you discover your most flattering colors. Upload a photo or take a personalized quiz, and get a custom color palette, clothing, and makeup recommendations tailored to your unique features.

---

## 🚀 Features
- **AI Color Analysis:** Upload your photo for instant, AI-driven color and undertone analysis.
- **Interactive Quiz:** Not ready to upload a photo? Take a quick quiz to get your color season and palette.
- **Personalized Palettes:** Receive curated color palettes, clothing, and makeup suggestions based on your results.
- **Modern UI/UX:** Clean, responsive, and visually appealing interface.

---

## 🛠 Frameworks & Tools
- **Frontend:** React (Vite), modern CSS
- **Backend:** FastAPI (Python)
- **AI/ML:**
  - Face & skin detection: [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface), custom skin tone models
  - Color analysis: Custom Python logic
  - LLM (Language Model): For generating recommendations and explanations
- **Other:**
  - Python libraries: numpy, torch, PIL, etc.
  - Node.js for frontend tooling

---

## 🧠 Models Used
- **Face Detection:** RetinaFace (PyTorch)
- **Skin Tone & Color Analysis:** Custom-trained models (see `facer/` directory)
- **LLM:** Used for generating natural language recommendations and explanations

---

## 🌈 How It Works
1. **Upload a Photo or Take the Quiz**
   - Upload a clear photo of your face, or answer a few quick questions about your features.
   - ![Upload Photo](docs/screenshots/upload_photo.jpg)
   - ![Take Quiz](docs/screenshots/take_quiz.jpg)
2. **AI Analysis**
   - The backend detects your face, analyzes your skin tone, undertone, and features.
   - An LLM generates a personalized color palette and recommendations.
3. **View Your Results**
   - Instantly see your color season, palette, and tailored clothing/makeup suggestions.
   - ![Palette Display](docs/screenshots/palette_display.jpg)
   - ![Why Color Analysis](docs/screenshots/why_color_analysis.jpg)

---

## 📸 Example Flow
1. **Upload your photo or take the quiz:**
   - ![Upload Photo](docs/screenshots/upload_photo.jpg)
   - ![Take Quiz](docs/screenshots/take_quiz.jpg)
2. **Get your personalized palette and recommendations:**
   - ![Palette Display](docs/screenshots/palette_display.jpg)
   - ![Why Color Analysis](docs/screenshots/why_color_analysis.jpg)

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
MIT
  