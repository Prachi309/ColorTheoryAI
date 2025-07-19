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
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

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
  const [currentFact, setCurrentFact] = useState('');

  const fashionFacts = [
    "India has one of the world's oldest textile traditions, dating back over 5,000 years to the Indus Valley Civilization, where cotton weaving was already practiced.",
    "The saree is considered one of the oldest forms of clothing still in existence, with origins traced back to around 2800–1800 BCE.",
    "Turmeric-dyed clothing was used in ancient India not just for its vibrant color but also for its antibacterial properties.",
    "India was a global exporter of fine muslin and silk during the Mughal period, particularly from Bengal and Kashmir.",
    "Zardozi embroidery, which uses gold and silver threads, flourished during the Mughal era and was traditionally reserved for royal garments.",
    "India is the largest producer of cotton in the world, and cotton remains a staple in traditional Indian wear.",
    "Khadi, a hand-spun cloth, was popularized by Mahatma Gandhi during the freedom movement and symbolizes self-reliance.",
    "Banarasi silk from Varanasi, Chikankari from Lucknow, and Bandhani from Gujarat/Rajasthan are internationally recognized textile crafts.",
    "Pashmina shawls from Kashmir are made from the fine undercoat of Himalayan goats and are among the most luxurious shawls globally.",
    "India has more than 100 unique types of handloom fabrics, many of which have Geographical Indication (GI) tags (e.g., Kanjeevaram silk, Muga silk).",
    "Lakmé Fashion Week, started in 2000, is one of India's biggest fashion events and has helped launch many leading designers.",
    "India's fashion industry is valued at over $100 billion (2024 estimate) and is one of the fastest-growing fashion markets globally.",
    "Bollywood significantly shapes Indian fashion trends, often popularizing traditional styles like lehengas, anarkalis, and sarees with a modern twist.",
    "Designers like Sabyasachi, Manish Malhotra, and Anita Dongre have global recognition and are often worn by international celebrities.",
    "Fusion fashion (mix of Indian and Western styles) is a defining trend in Indian urban wear, especially among Gen Z and millennials.",
    "India's fashion exports rely heavily on handmade and artisan products, contributing to employment in rural areas and preserving craft traditions.",
    "Many Indian designers are shifting toward slow fashion, promoting handloom, eco-friendly dyes, and zero-waste design.",
    "The \"Make in India\" initiative supports the handloom sector, which employs over 4 million people—70% of them women."
  ];

  // Function to get random fact
  const getRandomFact = () => {
    return fashionFacts[Math.floor(Math.random() * fashionFacts.length)];
  };

  // Effect to change fact every 3 seconds when loading
  React.useEffect(() => {
    if (loading) {
      setCurrentFact(getRandomFact());
      const interval = setInterval(() => {
        setCurrentFact(getRandomFact());
      }, 8000);
      return () => clearInterval(interval);
    }
  }, [loading]);

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
        `${BACKEND_URL}/image`,
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
        `${BACKEND_URL}/palette_llm?openrouter_api_key=${API_KEY}`,
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
        `${BACKEND_URL}/palette_llm?openrouter_api_key=${API_KEY}`,
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
          <div className="hero-content fade-in-hero" style={{ textAlign: 'center', marginBottom: 32 }}>
            <h1>Your Personal Color Theory AI Assistant</h1>
            <p>AI-powered color analysis to unlock your most flattering colors. Upload a photo or take our quiz to get your personalized palette.</p>
            <div className="hero-btns fade-in-hero-btns" style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
              gap: '20px', 
              maxWidth: '800px', 
              margin: '0 auto 32px auto',
              justifyContent: 'center'
            }}>
              <button className='btn btn-primary btn-icon' onClick={() => setMode('image')}>
                <span role="img" aria-label="camera">📷</span> Upload Photo
              </button>
              <button className='btn btn-secondary btn-icon' onClick={() => setMode('quiz')}>
                <span role="img" aria-label="quiz">📝</span> Take Quiz
              </button>
              <button className='btn btn-outlined btn-icon' onClick={() => setShowWhyModal(true)}>
                <span role="img" aria-label="lightbulb">💡</span> Why You Need Color Analysis
              </button>
              <button className='btn btn-outlined btn-icon' onClick={() => setShowUndertoneModal(true)}>
                <span role="img" aria-label="palette">🎨</span> Know About Your Undertone
              </button>
            </div>
          </div>
          <SeasonTypes />
        </section>
      )}
      {/* Floating Chat Button - Only on landing page */}
      {!mode && (
        <div style={{
          position: 'fixed',
          bottom: '32px',
          right: '32px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '8px',
          zIndex: 100
        }}>
          {/* Floating Text */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            color: '#7b7be5',
            padding: '8px 16px',
            borderRadius: '20px',
            fontSize: '14px',
            fontWeight: '600',
            boxShadow: '0 2px 12px rgba(123, 123, 229, 0.2)',
            whiteSpace: 'nowrap',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(160, 132, 238, 0.2)'
          }}>
            What to wear today?
          </div>
          {/* Chat Button */}
          <button 
            className="floating-chat-btn" 
            onClick={() => setShowStyleAssistant(true)}
            style={{
              width: '72px',
              height: '72px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #a084ee 0%, #7b7be5 100%)',
              color: '#fff',
              border: '3px solid #fff',
              boxShadow: '0 6px 32px rgba(123, 123, 229, 0.4)',
              cursor: 'pointer',
              fontSize: '28px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s, transform 0.18s cubic-bezier(0.4,0,0.2,1)',
              zIndex: 100
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'scale(1.1)';
              e.target.style.boxShadow = '0 8px 40px rgba(123, 123, 229, 0.5)';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'scale(1)';
              e.target.style.boxShadow = '0 6px 32px rgba(123, 123, 229, 0.4)';
            }}
            aria-label="What to wear today?"
          >
            🤔
          </button>
        </div>
      )}
      {/* Why You Need Color Analysis */}
      {showWhyModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(123,123,229,0.10)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: 18, boxShadow: '0 4px 32px #a084ee22', padding: '2.5rem 2.5rem 2rem 2.5rem', maxWidth: 1100, width: '95vw', position: 'relative' }}>
            <button className="btn btn-outlined" onClick={() => setShowWhyModal(false)} style={{ position: 'absolute', top: 70, right: 18, width: 44, height: 44, borderRadius: '50%', fontSize: 28, color: '#bbb', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }} aria-label="Close">&times;</button>
            <WhyChooseColorAI />
          </div>
        </div>
      )}
      {/* Know About Your Undertone */}
      {showUndertoneModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(67,233,123,0.10)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: 18, boxShadow: '0 4px 32px #43e97b22', padding: '2.5rem 2.5rem 2rem 2.5rem', maxWidth: 1100, width: '95vw', position: 'relative' }}>
            <button className="btn btn-outlined" onClick={() => setShowUndertoneModal(false)} style={{ position: 'absolute', top: 100, right: 18, width: 44, height: 44, borderRadius: '50%', fontSize: 28, color: '#bbb', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }} aria-label="Close">&times;</button>
            <UndertoneInfo />
          </div>
        </div>
      )}
      {showStyleAssistant && (
        <StyleAssistant onClose={() => setShowStyleAssistant(false)} />
      )}
      {/* Upload Image Modal */}
      {mode === 'image' && (
        <div className="upload-modal" style={{ background: 'rgba(255,255,255,0.98)', borderRadius: 18, boxShadow: '0 4px 32px #0002', padding: '2.5rem 2.5rem 2rem 2.5rem',  maxWidth: 1200, width: '100%', textAlign: 'center', position: 'relative' }}>
          <button className="btn btn-outlined" onClick={() => { setMode(null); setImage(null); setResult(''); setUndertone(''); setImagePrompt(''); }} style={{ position: 'absolute', top: 18, right: 18, width: 40, height: 40, borderRadius: '50%', fontSize: 22, color: '#bbb', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }} aria-label="Close">&times;</button>
          {/* Go back to home page button, only when result is shown */}
          {result && (
            <button
              className="btn btn-outlined"
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
                zIndex: 2,
                fontSize: '14px',
                padding: '8px 16px',
                background: 'rgba(255,255,255,0.9)',
                border: '1px solid #e5e5f7',
                color: '#7b7be5',
                fontWeight: 600,
                backdropFilter: 'blur(8px)',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.target.style.background = 'rgba(255,255,255,0.95)';
                e.target.style.borderColor = '#a084ee';
                e.target.style.transform = 'translateY(-1px)';
                e.target.style.boxShadow = '0 2px 8px rgba(123,123,229,0.15)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'rgba(255,255,255,0.9)';
                e.target.style.borderColor = '#e5e5f7';
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = 'none';
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
                      cursor: "pointer",
                      transition: "all 0.2s ease"
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.background = "#d8d8ff";
                      e.target.style.transform = "translateY(-1px)";
                      e.target.style.boxShadow = "0 2px 8px rgba(123,123,229,0.2)";
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.background = "#ececff";
                      e.target.style.transform = "translateY(0)";
                      e.target.style.boxShadow = "none";
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
                  <label style={{
                    fontWeight: 700,
                    color: '#7b7be5',
                    marginBottom: 12,
                    fontSize: 15,
                    display: 'block'
                  }}>
                    Your Undertone:
                  </label>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    {[
                      { value: 'warm', label: 'Warm' },
                      { value: 'cool', label: 'Cool' },
                      { value: 'neutral', label: 'Neutral' }
                    ].map((option) => (
                      <label key={option.value} style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        padding: '8px 16px',
                        borderRadius: 8,
                        border: undertone === option.value ? '2px solid #a084ee' : '2px solid #e5e5f7',
                        background: undertone === option.value ? '#f6f3ff' : '#fff',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        fontWeight: 500,
                        color: undertone === option.value ? '#7b7be5' : '#666'
                      }}>
                        <input
                          type="radio"
                          name="undertone"
                          value={option.value}
                          checked={undertone === option.value}
                          onChange={(e) => setUndertone(e.target.value)}
                          style={{ display: 'none' }}
                        />
                        {option.label}
                      </label>
                    ))}
                  </div>
                </div>
                {loading && (
                  <div className="loading-facts">
                    <h3>Did you know?</h3>
                    <p>{currentFact}</p>
                  </div>
                )}
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
                    boxShadow: loading ? '0 2px 8px #a084ee33' : '0 2px 8px #a084ee22',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    opacity: loading ? 0.7 : 1,
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (!loading) {
                      e.target.style.background = '#a084ee';
                      e.target.style.boxShadow = '0 4px 16px #a084ee33';
                      e.target.style.transform = 'translateY(-1px)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!loading) {
                      e.target.style.background = '#7b7be5';
                      e.target.style.boxShadow = '0 2px 8px #a084ee22';
                      e.target.style.transform = 'translateY(0)';
                    }
                  }}
                >
                  {loading ? (
                    <>
                      Analyzing
                      <span className="bouncing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </span>
                    </>
                  ) : 'Analyze'}
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
                    `${BACKEND_URL}/quiz_palette_llm?openrouter_api_key=${API_KEY}`,
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