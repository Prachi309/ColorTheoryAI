import React, { useState } from "react";
import ClothingImageResults from "./ClothingImageResults";
import RecommendationPage from "./RecommendationPage";

const OPENROUTER_API_KEY = import.meta.env.VITE_OPENROUTER_API_KEY;

const steps = [
  {
    bot: "What would you like to dress for?",
    options: [
      { label: "Dress according to Occasion", value: "Occasion" },
      { label: "Dress according to Weather", value: "Weather" },
      { label: "Dress according to Mood", value: "Mood" },
      { label: "Dress according to Place", value: "Place" },
    ],
    key: "dressing_focus",
  },
  {
    bot: "What is your gender?",
    options: [
      { label: "Male", value: "Male" },
      { label: "Female", value: "Female" },
      { label: "Other", value: "Other" },
    ],
    key: "gender",
  },
  {
    bot: "What is your body type?",
    options: [
      { label: "Rectangle / Straight", value: "Rectangle / Straight" },
      { label: "Triangle / Pear-Shaped", value: "Triangle / Pear-Shaped" },
      { label: "Inverted Triangle / Apple-Shaped", value: "Inverted Triangle / Apple-Shaped" },
      { label: "Hourglass / Balanced", value: "Hourglass / Balanced" },
      { label: "Oval / Round", value: "Oval / Round" },
      { label: "Muscular / Athletic", value: "Muscular / Athletic" },
      { label: "I’m not sure", value: "I’m not sure" },
    ],
    key: "body_type",
  },
];

const contextQuestions = {
  Occasion: {
    question: "Which occasion are you dressing for?",
    options: [
      "Casual outing",
      "Formal event",
      "Wedding or party",
      "Office or meeting",
      "College or school",
      "Date or romantic evening",
    ],
  },
  Weather: {
    question: "What's the current weather like?",
    options: [
      "Sunny",
      "Rainy",
      "Cold / Winter",
      "Hot and humid",
      "Windy",
    ],
  },
  Mood: {
    question: "How are you feeling today?",
    options: [
      "Happy and vibrant",
      "Chill and relaxed",
      "Bold and powerful",
      "Calm and peaceful",
      "Energetic and playful",
    ],
  },
  Place: {
    question: "Where are you heading?",
    options: [
      "Beach",
      "Mountains / Nature",
      "Urban City",
      "Restaurant / Cafe",
      "Home / Stay-in",
      "Gym / Workout",
    ],
  },
};

const optionColors = {
  Occasion: { bg: "#e3f0ff", border: "#2563eb", color: "#2563eb" },
  Weather: { bg: "#e6f9ef", border: "#22c55e", color: "#22c55e" },
  Mood: { bg: "#f3e6fa", border: "#a084ee", color: "#a084ee" },
  Place: { bg: "#fff7e6", border: "#fbbf24", color: "#fbbf24" },
  Gender: { bg: "#f6f7fa", border: "#bbb", color: "#7C83F7" },
  Body: { bg: "#f6f7fa", border: "#bbb", color: "#7C83F7" },
};

function getOptionStyle(step, focus) {
  if (step === 0) return optionColors.Occasion;
  if (step === 1) return optionColors.Gender;
  if (step === 2) return optionColors.Body;
  if (step === 3 && focus && optionColors[focus]) return optionColors[focus];
  return { bg: "#fff", border: "#a084ee", color: "#7C83F7" };
}

const sleep = ms => new Promise(res => setTimeout(res, ms));

const buildPrompt = (answers, season, palette) => {
  let paletteNote = "";
  if (season || (palette && palette.length > 0)) {
    paletteNote = `\nThe user's color season is: ${season || ""}.`;
    if (palette && palette.length > 0) {
      paletteNote += `\nThe user's recommended palette colors are: ${palette.join(", ")}.`;
    }
    paletteNote += "\nTake these into consideration for your recommendations.";
  }
  return `
Recommend 2 items each of:
- Clothing
- Footwear
- Accessories (if suitable)
- Makeup products (foundation, lipstick, blush)
- Fragrance
Very very important => The response should not be more than 10 lines
For a user with:
Dressing focus: ${answers.dressing_focus}
Gender: ${answers.gender}
Body type: ${answers.body_type}
Context: ${answers.context_answer}
${paletteNote}

Return as JSON array only:
[
  {
    "category": "...",
    "product": "...",
    "description": "...",
    "query": "search keywords only for finding the product"
  },
  ...
]
`;
};

const StyleAssistant = ({ onClose, season: propSeason, palette: propPalette }) => {
  const [chat, setChat] = useState([
    { from: "bot", text: steps[0].bot }
  ]);
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [typing, setTyping] = useState(false);
  const [finalReply, setFinalReply] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [showOptions, setShowOptions] = useState(true);

  const handleOption = async (option) => {
    setChat(prev => [...prev, { from: "user", text: option.label || option }]);
    setTyping(true);
    await sleep(700);
    setTyping(false);
    if (step < 3) {
      setAnswers(prev => ({ ...prev, [steps[step].key]: option.value || option }));
      if (step === 2) {
        const focus = answers.dressing_focus || (option.value || option);
        setChat(prev => [...prev, { from: "bot", text: contextQuestions[focus].question }]);
        setStep(3);
      } else {
        setChat(prev => [...prev, { from: "bot", text: steps[step + 1].bot }]);
        setStep(step + 1);
      }
    } else if (step === 3) {
      setShowOptions(false); // Hide options immediately after last question is answered
      const focus = answers.dressing_focus;
      setAnswers(prev => ({ ...prev, context_answer: option }));
      setTyping(true);
      setTimeout(async () => {
        setTyping(false);
        try {
          const prompt = buildPrompt(answers, propSeason, propPalette);
          const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              model: "mistralai/mistral-small-3.2-24b-instruct:free",
              messages: [{ role: "user", content: prompt }]
            })
          });
          const data = await response.json();
          const suggestion = data.choices?.[0]?.message?.content || "No suggestion received.";
          setFinalReply(suggestion);
          setChat(prev => [...prev, { from: "bot", text: suggestion }]);
        } catch {
          setFinalReply("Sorry, something went wrong. Please try again later.");
          setChat(prev => [...prev, { from: "bot", text: "Sorry, something went wrong. Please try again later." }]);
        }
      }, 1200);
    }
  };

  let options = [];
  let optionStyle = { bg: "#fff", border: "#a084ee", color: "#7C83F7" };
  let focus = answers.dressing_focus;
  if (step < 3) {
    options = steps[step].options;
    optionStyle = getOptionStyle(step, focus);
  } else if (step === 3) {
    focus = answers.dressing_focus;
    if (focus && contextQuestions[focus]) {
      options = contextQuestions[focus].options;
      optionStyle = getOptionStyle(step, focus);
    }
  }

  let productList = [];
  let cleaned = finalReply ? finalReply.trim() : "";
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```[a-zA-Z]*\n?/, '').replace(/```$/, '').trim();
  }
  try {
    productList = JSON.parse(cleaned);
  } catch (e) {
    productList = [];
  }

  // Color map for main options
  const optionColorMap = {
    "Dress according to Occasion": { bg: "#e3f0ff", border: "#2563eb", color: "#2563eb" },
    "Dress according to Weather": { bg: "#e6f9ef", border: "#22c55e", color: "#22c55e" },
    "Dress according to Mood": { bg: "#f3e6fa", border: "#a084ee", color: "#a084ee" },
    "Dress according to Place": { bg: "#fff7e6", border: "#fbbf24", color: "#fbbf24" },
    // fallback
    default: { bg: "#fff", border: "#a084ee", color: "#7C83F7" }
  };

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, width: "100vw", height: "100vh", background: "rgba(123,123,229,0.10)", zIndex: 1200,
      display: "flex", alignItems: "center", justifyContent: "center"
    }}>
      <div style={{ background: "#fff", borderRadius: 18, boxShadow: "0 4px 32px #a084ee22", padding: 0, maxWidth: 700, width: "95vw", height: 600, maxHeight: "70vh", display: "flex", flexDirection: "column" }}>
        {/* Header */}
        <div style={{ background: "linear-gradient(90deg,#a084ee,#7C83F7)", borderTopLeftRadius: 18, borderTopRightRadius: 18, padding: "18px 24px 10px 24px", color: "#fff" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ background: "#fff", color: "#a084ee", borderRadius: "50%", width: 36, height: 36, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, fontWeight: 700 }}>🧑‍🎤</span>
            <div>
              <div style={{ fontWeight: 700, fontSize: 20 }}>StyleAI Assistant</div>
              <div style={{ fontSize: 13, opacity: 0.9 }}>What should I wear today? - Personalized Outfit Suggestions</div>
            </div>
            <button onClick={onClose} style={{ marginLeft: "auto", background: "none", border: "none", color: "#fff", fontSize: 26, cursor: "pointer" }}>&times;</button>
          </div>
        </div>
        {/* Chat area and options */}
        <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", background: "transparent", boxSizing: "border-box" }}>
          <div
            style={{
              flex: 1,
              height: "100%",
              maxHeight: "100%",
              overflowY: "auto",
              background: "transparent",
              paddingRight: 8,
              minHeight: 0,
              boxSizing: "border-box"
            }}
          >
            {chat.map((msg, i) => (
              <div key={i} style={{ display: "flex", justifyContent: msg.from === "user" ? "flex-end" : "flex-start", marginBottom: 12 }}>
                <div style={{
                  background: msg.from === "user" ? "#a084ee" : "#fff",
                  color: msg.from === "user" ? "#fff" : "#333",
                  borderRadius: 14,
                  padding: "10px 16px",
                  maxWidth: 540,
                  fontSize: 15,
                  boxShadow: msg.from === "user" ? "0 2px 8px #a084ee33" : "0 2px 8px #eee",
                  textAlign: "left"
                }}>{msg.text}</div>
              </div>
            ))}
            {typing && (
              <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 12 }}>
                <div style={{ background: "#fff", color: "#aaa", borderRadius: 14, padding: "10px 16px", maxWidth: 180, fontSize: 15, fontStyle: "italic" }}>
                  <span className="typing-dots">Typing</span>
                </div>
              </div>
            )}
            {/* Show product recommendations and images after finalReply */}
            {finalReply && !showAll && (
              <div style={{ marginTop: 24, whiteSpace: "pre-wrap", fontSize: 16, color: "#444" }}>
                {/* Show only the summary/first 10 lines */}
                {Array.isArray(productList) && productList.length > 0
                  ? productList
                      .slice(0, 3) // show first 3 products as a summary
                      .map((item, idx) => (
                        <div key={idx}>
                          <strong>{item.category}: {item.product}</strong>
                          <div style={{ color: "#888", fontSize: 14 }}>{item.description}</div>
                        </div>
                      ))
                  : (() => {
                      // fallback: show first 10 lines of raw text
                      const lines = cleaned.split('\n').slice(0, 10).join('\n');
                      return lines.length > 0 ? lines : cleaned.slice(0, 500);
                    })()
                }
              </div>
            )}
            {finalReply && !showAll && (
              <button
                onClick={() => setShowAll(true)}
                style={{
                  margin: "24px auto 0 auto",
                  display: "block",
                  background: "#fff",
                  color: "#a084ee",
                  border: "2px solid #a084ee",
                  borderRadius: 8,
                  fontWeight: 700,
                  fontSize: 16,
                  padding: "10px 24px",
                  cursor: "pointer"
                }}
              >
                See All Recommendations
              </button>
            )}
            {showAll && (
              <RecommendationPage
                productList={productList}
                onClose={() => setShowAll(false)}
              />
            )}
          </div>
          {/* Options only shown if not finalReply */}
          {!finalReply && showOptions && (
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                justifyContent: "center",
                gap: 12,
                padding: "8px 0 24px 0",
                background: "#fafbfc",
                width: "100%",
                boxSizing: "border-box"
              }}
            >
              {options.map((opt, i) => {
                const style = optionColorMap[opt.label || opt] || optionColorMap.default;
                return (
                  <button
                    key={i}
                    onClick={() => !typing && handleOption(opt)}
                    disabled={typing}
                    style={{
                      background: style.bg,
                      border: `1.5px solid ${style.border}`,
                      color: style.color,
                      borderRadius: 8,
                      padding: "12px 18px",
                      fontWeight: 600,
                      fontSize: 15,
                      cursor: typing ? "not-allowed" : "pointer",
                      opacity: typing ? 0.6 : 1,
                      flex: "1 1 calc(50% - 16px)",
                      minWidth: 140,
                      maxWidth: 240,
                      boxSizing: "border-box"
                    }}
                  >
                    {opt.label || opt}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StyleAssistant;