import React, { useState } from 'react';
import axios from 'axios';
import './index.css';
import './Landing.css';
import AIResponseDisplay from './AIResponseDisplay';
import SeasonTypes from './SeasonTypes';
import WhyChooseColorAI from './WhyChooseColorAI';
import ColorQuiz from './ColorQuiz';
import UndertoneInfo from './UndertoneInfo';
import StyleAssistant from "./StyleAssistant";

const API_KEY = import.meta.env.VITE_OPENROUTER_API_KEY;
const HERO_IMAGE = 'https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=facearea&w=400&h=400&facepad=2';

// Helper to safely parse LLM response (strip markdown code block if present)
function safeParseLLMResponse(llm_response) {
  if (typeof llm_response !== 'string') return llm_response;
  let cleaned = llm_response.trim();
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```[a-zA-Z]*\n?/, '').replace(/```$/, '').trim();
  }
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    return llm_response;
  }
}

function App() {
  const [mode, setMode] = useState(null); // null, 'image', 'quiz'
  const [showWhyModal, setShowWhyModal] = useState(false);
  const [showUndertoneModal, setShowUndertoneModal] = useState(false);
  const [image, setImage] = useState(null);
  const [imageName, setImageName] = useState('');
  const [undertone, setUndertone] = useState('');
  const [quizColors, setQuizColors] = useState(['', '', '', '', '', '']);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [imagePrompt, setImagePrompt] = useState('');
  const [showStyleAssistant, setShowStyleAssistant] = useState(false);

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
    setImageName(e.target.files[0]?.name || '');
    setResult('');
  };

  const handleQuizColorChange = (idx, value) => {
    const newColors = [...quizColors];
    newColors[idx] = value;
    setQuizColors(newColors);
    setResult('');
  };

  const handleImageSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult('');
    try {
      if (!image) {
        setResult('Please select an image.');
        setLoading(false);
        return;
      }
      if (!undertone) {
        setResult('Please select your undertone.');
        setLoading(false);
        return;
      }

      // 1. Get the season from /image
      const imageFormData = new FormData();
      imageFormData.append('file', image);
      const imageResponse = await axios.post(
        'http://localhost:8000/image',
        imageFormData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      const season = imageResponse.data.season; // <-- get the season

      // 2. Now call /palette_llm with the season
      const formData = new FormData();
      formData.append('file', image);
      formData.append('openrouter_api_key', API_KEY);
      const undertonePrompt = `The user selected their undertone as: ${undertone}. Please consider this in your analysis.`;
      setImagePrompt(undertonePrompt);
      formData.append('prompt', undertonePrompt);
      formData.append('season', season); // <-- use the season here

      const response = await axios.post(
        `http://localhost:8000/palette_llm?openrouter_api_key=${API_KEY}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResult(safeParseLLMResponse(response.data.llm_response) || 'No response from LLM.');
    } catch (err) {
      setResult('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleQuizSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult('');
    try {
      const quizPrompt = `The user selected these colors: ${quizColors.filter(Boolean).join(', ')}. Please analyze and provide a seasonal color palette and recommendations in the same format as usual.`;
      const response = await axios.post(
        `http://localhost:8000/palette_llm?openrouter_api_key=${API_KEY}`,
        { prompt: quizPrompt },
        { headers: { 'Content-Type': 'application/json' } }
      );
      setResult(safeParseLLMResponse(response.data.llm_response) || 'No response from LLM.');
    } catch (err) {
      setResult('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Main UI
  return (
    <div className="landing-bg" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      {/* Hero Section two options */}
      {!mode && (
        <section className="hero-section" style={{ flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '80vh' }}>
          <div className="hero-content" style={{ textAlign: 'center', marginBottom: 32 }}>
            <h1>Your Personal Color Theory AI Assistant</h1>
            <p>AI-powered color analysis to unlock your most flattering colors. Upload a photo or take our quiz to get your personalized palette.</p>
            <div className="hero-btns" style={{ justifyContent: 'center', gap: 16, display: 'flex', flexWrap: 'wrap', marginBottom: 16 }}>
              <button className='hero-btn' onClick={() => setMode('image')}>📷 Upload Photo</button>
              <button className='hero-btn' onClick={() => setMode('quiz')}>📝 Take Quiz</button>
            </div>
            <div className="hero-btns" style={{ justifyContent: 'center', gap: 16, display: 'flex', flexWrap: 'wrap', marginBottom: 0 }}>
              <button className='hero-btn' style={{ background: '#fff', color: '#7b7be5', border: '2px solid #a084ee', fontWeight: 700 }} onClick={() => setShowWhyModal(true)}>
                💡 Why You Need Color Analysis
              </button>
              <button className='hero-btn' style={{ background: '#fff', color: '#7b7be5', border: '2px solid #43e97b', fontWeight: 700 }} onClick={() => setShowUndertoneModal(true)}>
                🎨 Know About Your Undertone
              </button>
              <button className='hero-btn' style={{ background: '#fff', color: '#a084ee', border: '2px solid #a084ee', fontWeight: 700 }} onClick={() => setShowStyleAssistant(true)}>
                🤔 What to wear today?
              </button>
            </div>
          </div>
          <SeasonTypes />
        </section>
      )}
      {/* Why You Need Color Analysis */}
      {showWhyModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(123,123,229,0.10)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: 18, boxShadow: '0 4px 32px #a084ee22', padding: '2.5rem 2.5rem 2rem 2.5rem', maxWidth: 1100, width: '95vw', position: 'relative' }}>
            <button onClick={() => setShowWhyModal(false)} style={{ position: 'absolute', top: 70, right: 18, background: 'none', border: 'none', fontSize: 28, color: '#bbb', cursor: 'pointer' }}>&times;</button>
            <WhyChooseColorAI />
          </div>
        </div>
      )}
      {/* Know About Your Undertone */}
      {showUndertoneModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(67,233,123,0.10)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: 18, boxShadow: '0 4px 32px #43e97b22', padding: '2.5rem 2.5rem 2rem 2.5rem', maxWidth: 1100, width: '95vw', position: 'relative' }}>
            <button onClick={() => setShowUndertoneModal(false)} style={{ position: 'absolute', top: 100, right: 18, background: 'none', border: 'none', fontSize: 28, color: '#bbb', cursor: 'pointer' }}>&times;</button>
            <UndertoneInfo />
          </div>
        </div>
      )}
      {showStyleAssistant && (
        <StyleAssistant onClose={() => setShowStyleAssistant(false)} />
      )}
      {/* Upload Image Modal */}
      {mode === 'image' && (
        <div className="upload-modal" style={{ background: 'rgba(255,255,255,0.98)', borderRadius: 18, boxShadow: '0 4px 32px #0002', padding: '2.5rem 2.5rem 2rem 2.5rem', maxWidth: 1200, width: '100%', textAlign: 'center', position: 'relative' }}>
          <button onClick={() => { setMode(null); setImage(null); setResult(''); setUndertone(''); setImagePrompt(''); }} style={{ position: 'absolute', top: 18, right: 18, background: 'none', border: 'none', fontSize: 22, color: '#bbb', cursor: 'pointer' }}>&times;</button>
          {/* Go back to home page button, only when result is shown */}
          {result && (
            <button
              onClick={() => {
                setMode(null);
                setImage(null);
                setResult('');
                setUndertone('');
                setImagePrompt('');
              }}
              style={{
                position: 'absolute',
                top: 18,
                left: 18,
                background: '#7b7be5',
                color: '#fff',
                border: 'none',
                borderRadius: 8,
                padding: '10px 20px',
                fontWeight: 700,
                fontSize: 16,
                cursor: 'pointer',
                zIndex: 2
              }}
            >
              ← Go back to home page
            </button>
          )}
          {/* Improved upload form UI from root app.jsx */}
          {!result && (
            <div
              style={{
                background: "linear-gradient(135deg, #f6f3ff 0%, #eaf6ff 100%)",
                borderRadius: 22,
                boxShadow: "0 4px 24px rgba(123,123,229,0.10)",
                padding: "2.5rem 2.5rem 2rem 2.5rem",
                maxWidth: 440,
                margin: "0 auto 32px auto",
                textAlign: "center",
                border: "1.5px solid #e5e5f7",
                position: "relative"
              }}
            >
              <div style={{ fontSize: 48, marginBottom: 10, color: "#7b7be5" }}>📷</div>
              <h2 style={{
                color: "#7b7be5",
                fontSize: "2.2rem",
                fontWeight: 900,
                marginBottom: 8,
                letterSpacing: "1px",
                fontFamily: "Inter, sans-serif"
              }}>
                Upload Your Photo
              </h2>
              <p style={{
                color: "#888",
                fontSize: "1.15rem",
                marginBottom: 22,
                fontWeight: 500
              }}>
                Get instant color analysis with our <span style={{ color: "#7b7be5", fontWeight: 700 }}>AI technology</span>
              </p>
              <form onSubmit={handleImageSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                <div className="drop-area" style={{ marginBottom: 0 }}>
                  <label htmlFor="file-upload" className="drop-label" style={{ cursor: "pointer" }}>
                    <span className="drop-icon" style={{ fontSize: 32, display: "block", marginBottom: 6 }}>☁</span>
                    <span style={{ fontWeight: 600, color: "#7b7be5" }}>Upload your photo here</span>
                    <input id="file-upload" type="file" accept="image/*" onChange={handleFileChange} required style={{ display: 'none' }} />
                  </label>
                  <button
                    type="button"
                    className="choose-photo-btn"
                    onClick={() => document.getElementById('file-upload').click()}
                    style={{
                      marginTop: 10,
                      background: "#ececff",
                      color: "#7b7be5",
                      border: "none",
                      borderRadius: 8,
                      padding: "8px 18px",
                      fontWeight: 600,
                      cursor: "pointer"
                    }}
                  >
                    Choose Photo
                  </button>
                  <div className="file-info" style={{ color: "#aaa", fontSize: 13, marginTop: 6 }}>
                    Supports JPG Format Only
                  </div>
                  {imageName && (
                    <div style={{ color: '#7b7be5', fontWeight: 600, marginTop: 8 }}>
                      Selected: {imageName}
                    </div>
                  )}
                </div>
                {/* Undertone selection */}
                <div style={{ marginTop: 8, marginBottom: 0, textAlign: 'left' }}>
                  <label htmlFor="undertone-select" style={{
                    fontWeight: 700,
                    color: '#7b7be5',
                    marginRight: 8,
                    fontSize: 15
                  }}>
                    Your Undertone:
                  </label>
                  <select
                    id="undertone-select"
                    value={undertone}
                    onChange={e => setUndertone(e.target.value)}
                    style={{
                      padding: '0.4rem 1rem',
                      borderRadius: 8,
                      border: '1px solid #a084ee',
                      fontWeight: 500,
                      color: '#7b7be5',
                      background: '#f6f3ff',
                      marginLeft: 8
                    }}
                    required
                  >
                    <option value="">Select</option>
                    <option value="warm">Warm</option>
                    <option value="cool">Cool</option>
                    <option value="neutral">Neutral</option>
                  </select>
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="analyze-btn"
                  style={{
                    marginTop: 0,
                    background: "#7b7be5",
                    color: "#fff",
                    border: "none",
                    borderRadius: 8,
                    padding: "12px 0",
                    fontWeight: 700,
                    fontSize: 16,
                    boxShadow: loading ? '0 2px 8px #a084ee33' : 'none',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    opacity: loading ? 0.7 : 1
                  }}
                >
                  {loading ? 'Analyzing...' : 'Analyze'}
                </button>
              </form>
            </div>
          )}
          {imagePrompt && (
            <div style={{ marginTop: 24, marginBottom: 8, color: '#888', fontSize: 13, background: '#f6f3ff', borderRadius: 8, padding: 8, wordBreak: 'break-word' }}>
              {/* <strong>Please wait a few seconds for the analysis results</strong><br /> */}
              {/* {imagePrompt} */}
            </div>
          )}
          {result && (
            <div className="result-area-horizontal" style={{ marginTop: 16, display: 'flex', gap: 32, alignItems: 'flex-start', justifyContent: 'center' }}>
              <AIResponseDisplay response={safeParseLLMResponse(result)} />
            </div>
          )}
        </div>
      )}
      {/* Quiz Modal */}
      {mode === 'quiz' && (
        <div className="upload-modal" style={{ background: 'rgba(255,255,255,0.98)', borderRadius: 18, boxShadow: '0 4px 32px #0002', padding: '2.5rem 2.5rem 2rem 2.5rem',  maxWidth: 1200, width: '100%', textAlign: 'center', position: 'relative' }}>
          <button onClick={() => { setMode(null); setResult(''); }} style={{ position: 'absolute', top: 18, right: 18, background: 'none', border: 'none', fontSize: 22, color: '#bbb', cursor: 'pointer' }}>&times;</button>
          {/* Go back to home page button, only when result is shown */}
          {result && (
            <button
              onClick={() => {
                setMode(null);
                setResult('');
              }}
              style={{
                position: 'absolute',
                top: 18,
                left: 18,
                background: '#7b7be5',
                color: '#fff',
                border: 'none',
                borderRadius: 8,
                padding: '10px 20px',
                fontWeight: 700,
                fontSize: 16,
                cursor: 'pointer',
                zIndex: 2
              }}
            >
              ← Go back to home page
            </button>
          )}
          {/* Use ColorQuiz component */}
          {!result && (
            <ColorQuiz
              loading={loading}
              result={null}
              onSubmit={async (answers) => {
                setLoading(true);
                setResult('');
                try {
                  const response = await axios.post(
                    `http://localhost:8000/quiz_palette_llm?openrouter_api_key=${API_KEY}`,
                    answers,
                    { headers: { 'Content-Type': 'application/json' } }
                  );
                  setResult(safeParseLLMResponse(response.data.llm_response) || 'No response from LLM.');
                } catch (err) {
                  setResult('Error: ' + (err.response?.data?.detail || err.message));
                } finally {
                  setLoading(false);
                }
              }}
            />
          )}
          {result && (
            <div className="result-area-horizontal" style={{ marginTop: 32, display: 'flex', gap: 32, alignItems: 'flex-start', justifyContent: 'center' }}>
              <AIResponseDisplay response={safeParseLLMResponse(result)} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;