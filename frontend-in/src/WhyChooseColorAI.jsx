import React from "react";
import "./Landing.css";

const features = [
  {
    icon: <span style={{fontSize: 24, background: '#B388FF', color: '#fff', borderRadius: 8, padding: 6, marginRight: 16}} role="img" aria-label="glow">💜</span>,
    title: "Enhance Your Natural Glow",
    desc: "The right colors make your skin look brighter, clearer, and more radiant.",
  },
  {
    icon: <span style={{fontSize: 24, background: '#40C4FF', color: '#fff', borderRadius: 8, padding: 6, marginRight: 16}} role="img" aria-label="eyes">👁️</span>,
    title: "Make Your Eyes Pop",
    desc: "Colors that complement your undertone make your eyes appear brighter and more vibrant.",
  },
  {
    icon: <span style={{fontSize: 24, background: '#69F0AE', color: '#fff', borderRadius: 8, padding: 6, marginRight: 16}} role="img" aria-label="health">💚</span>,
    title: "Look Younger & Healthier",
    desc: "Harmonious colors reduce signs of fatigue and aging, giving you a youthful appearance.",
  },
];

const benefits = [
  {
    icon: <span style={{fontSize: 28, background: 'linear-gradient(135deg,#FFB300,#FF7043)', color: '#fff', borderRadius: '50%', padding: 8}} role="img" aria-label="shopping">🛍️</span>,
    title: "Smart Shopping",
    desc: "Stop buying clothes that don't work. Invest in pieces that make you look amazing every time.",
    color: '#FF9800',
  },
  {
    icon: <span style={{fontSize: 28, background: 'linear-gradient(135deg,#B388FF,#40C4FF)', color: '#fff', borderRadius: '50%', padding: 8}} role="img" aria-label="clock">⏰</span>,
    title: "Save Time",
    desc: "Get dressed faster with a wardrobe full of colors that work perfectly together.",
    color: '#9575CD',
  },
  {
    icon: <span style={{fontSize: 28, background: 'linear-gradient(135deg,#69F0AE,#00C853)', color: '#fff', borderRadius: '50%', padding: 8}} role="img" aria-label="star">⭐</span>,
    title: "Boost Confidence",
    desc: "Feel confident knowing you always look your best in colors that enhance your natural beauty.",
    color: '#00C853',
  },
];

const WhyChooseColorAI = () => (
  <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 0" }}>
    <h1 style={{ textAlign: "center", fontWeight: 700, fontSize: 36 }}>Why You Need Color Analysis</h1>
    <p style={{ textAlign: "center", color: "#555", fontSize: 18, marginBottom: 40 }}>
      Color analysis transforms how you look and feel by identifying the colors that enhance your natural beauty.
    </p>
    <div style={{ display: "flex", gap: 40, alignItems: "flex-start", flexWrap: "wrap", marginBottom: 48, justifyContent: "center" }}>
      <div style={{ flex: 1, minWidth: 320, maxWidth: 420 }}>
        <h2 style={{ fontWeight: 700, fontSize: 26, marginBottom: 24 }}>Transform Your Appearance</h2>
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {features.map((f, i) => (
            <li key={i} style={{ display: "flex", alignItems: "flex-start", marginBottom: 28 }}>
              {f.icon}
              <div>
                <div style={{ fontWeight: 600, fontSize: 18, marginBottom: 4 }}>{f.title}</div>
                <div style={{ color: "#555", fontSize: 16 }}>{f.desc}</div>
              </div>
            </li>
          ))}
        </ul>
      </div>
      <div style={{ flex: 1, minWidth: 320, maxWidth: 420, display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div style={{ background: "#fff", borderRadius: 20, boxShadow: "0 2px 16px #0001", padding: 16, marginBottom: 8, width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
            {/* Replace these with your real before/after images if available */}
            <img src="/public/before.jpeg" alt="Before" style={{ borderRadius: 12, width: 160, height: 180, objectFit: "cover" }} />
            <img src="/public/after.jpeg" alt="After" style={{ borderRadius: 12, width: 160, height: 180, objectFit: "cover" }} />
          </div>
          <div style={{ color: "#555", fontStyle: "italic", fontSize: 15, marginTop: 10, textAlign: "center" }}>
            See the dramatic difference the right colors can make
          </div>
        </div>
      </div>
    </div>
    <div style={{ display: "flex", gap: 32, justifyContent: "center", flexWrap: "wrap" }}>
      {benefits.map((b, i) => (
        <div
          key={i}
          style={{
            background: "#fff",
            borderRadius: 16,
            padding: 32,
            minWidth: 260,
            maxWidth: 340,
            flex: 1,
            boxShadow: "0 2px 12px #0001",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            margin: "8px 0",
          }}
        >
          <div style={{ marginBottom: 16 }}>{b.icon}</div>
          <h3 style={{ fontWeight: 700, fontSize: 20, marginBottom: 10, color: b.color }}>{b.title}</h3>
          <div style={{ color: "#555", fontSize: 16, textAlign: "center" }}>{b.desc}</div>
        </div>
      ))}
    </div>
  </div>
);

export default WhyChooseColorAI; 