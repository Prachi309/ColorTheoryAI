import React from "react";
import "./Landing.css";

const undertones = [
  {
    type: "Warm Undertone",
    icon: (
      <span style={{ fontSize: 32, color: "#FFA726" }} role="img" aria-label="sun">☀️</span>
    ),
    bg: "#FFF3E0",
    color: "#FFA726",
    button: { text: "Select Warm", color: "#FF9800" },
    features: [
      "Golden or yellow base",
      "Veins appear greenish",
      "Gold jewelry looks better",
      "Tan easily in sun",
    ],
    dots: ["#FFA726", "#FF7043", "#FFD600", "#43A047"],
  },
  {
    type: "Cool Undertone",
    icon: (
      <span style={{ fontSize: 32, color: "#7E57C2" }} role="img" aria-label="snowflake">❄️</span>
    ),
    bg: "#F3EFFF",
    color: "#7E57C2",
    button: { text: "Select Cool", color: "#448AFF" },
    features: [
      "Pink or blue base",
      "Veins appear bluish",
      "Silver jewelry looks better",
      "Burn easily in sun",
    ],
    dots: ["#448AFF", "#AB47BC", "#E040FB", "#00B8D4"],
  },
  {
    type: "Neutral Undertone",
    icon: (
      <span style={{ fontSize: 32, color: "#26C6DA" }} role="img" aria-label="balance">⚖️</span>
    ),
    bg: "#E0F7FA",
    color: "#26C6DA",
    button: { text: "Select Neutral", color: "#00C853" },
    features: [
      "Mix of warm and cool",
      "Veins appear blue-green",
      "Both gold and silver work",
      "Tan moderately",
    ],
    dots: ["#607D8B", "#26C6DA", "#FFD600", "#00C853"],
  },
];

const quickTips = [
  {
    title: "Vein Test",
    desc: "Look at your wrist veins in natural light. Green = warm, blue = cool, blue-green = neutral.",
  },
  {
    title: "Jewelry Test",
    desc: "Compare gold vs silver jewelry against your skin. Which makes you glow?",
  },
  {
    title: "White Paper Test",
    desc: "Hold white paper to your face. Does your skin look pink/red (cool) or yellow/golden (warm)?",
  },
  {
    title: "Sun Reaction",
    desc: "How does your skin react to sun? Tan easily (warm) or burn first (cool)?",
  },
];

const UndertoneInfo = () => (
  <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 0" }}>
    <h1 style={{ textAlign: "center", fontWeight: 700, fontSize: 36 }}>Discover Your Undertone</h1>
    <p style={{ textAlign: "center", color: "#555", fontSize: 18, marginBottom: 40 }}>
      Understanding your skin's undertone is the foundation of finding your perfect colors. Select the option that best describes your skin.
    </p>
    <div style={{ display: "flex", gap: 32, justifyContent: "center", marginBottom: 40, flexWrap: "wrap" }}>
      {undertones.map((u) => (
        <div
          key={u.type}
          style={{
            background: u.bg,
            borderRadius: 16,
            padding: 32,
            minWidth: 280,
            maxWidth: 340,
            flex: 1,
            boxShadow: "0 2px 12px #0001",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            margin: "8px 0",
          }}
        >
          <div style={{ marginBottom: 16 }}>{u.icon}</div>
          <h2 style={{ fontWeight: 700, fontSize: 22, marginBottom: 12 }}>{u.type}</h2>
          <ul style={{ textAlign: "left", color: "#333", fontSize: 16, marginBottom: 18, paddingLeft: 0, listStyle: "none" }}>
            {u.features.map((f, i) => (
              <li key={i} style={{ marginBottom: 6 }}>
                <span style={{ color: u.color, marginRight: 8 }}>✔</span>
                {f}
              </li>
            ))}
          </ul>
          <div style={{ display: "flex", gap: 8, marginBottom: 18 }}>
            {u.dots.map((c, i) => (
              <span key={i} style={{ width: 20, height: 20, borderRadius: "50%", background: c, display: "inline-block" }} />
            ))}
          </div>
          <button
            style={{
              background: u.button.color,
              color: "#fff",
              border: "none",
              borderRadius: 8,
              padding: "10px 0",
              width: "100%",
              fontWeight: 600,
              fontSize: 16,
              cursor: "pointer",
              marginTop: "auto",
              boxShadow: "0 1px 4px #0002",
              transition: "background 0.2s",
            }}
          >
            {u.button.text}
          </button>
        </div>
      ))}
    </div>
    <div
      style={{
        background: "#fafbfc",
        borderRadius: 16,
        padding: 32,
        boxShadow: "0 2px 12px #0001",
        maxWidth: 900,
        margin: "0 auto",
      }}
    >
      <h2 style={{ textAlign: "center", fontWeight: 700, fontSize: 24, marginBottom: 24 }}>Quick Tips to Identify Your Undertone</h2>
      <div style={{ display: "flex", gap: 32, flexWrap: "wrap", justifyContent: "center" }}>
        {quickTips.map((tip, i) => (
          <div key={i} style={{ minWidth: 180, flex: 1, display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 16 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: "50%",
                background: ["#7C4DFF", "#00B8D4", "#AB47BC", "#FF7043"][i],
                color: "#fff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontWeight: 700,
                fontSize: 18,
                marginRight: 8,
                flexShrink: 0,
              }}
            >
              {i + 1}
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 2 }}>{tip.title}</div>
              <div style={{ color: "#555", fontSize: 15 }}>{tip.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

export default UndertoneInfo; 