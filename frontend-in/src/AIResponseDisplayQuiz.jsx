import React from "react";
import "./index.css";
// Season brand colors
const seasonColors = {
  Spring: "#F4B400",
  Summer: "#00B0FF",
  Autumn: "#D2691E",
  Winter: "#6A5ACD",
};

// parse "- #HEX (Name)" to {hex, name}
function parsePaletteArray(arr) {
  if (!Array.isArray(arr)) return [];
  return arr.map(str => {
    const match = str.match(/#([A-Fa-f0-9]{6})\s*\(([^)]+)\)/);
    if (match) return { hex: `#${match[1]}`, name: match[2] };
    return { hex: str, name: "" };
  });
}

// parse LLM response
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

function parseFormattedQuizResponse(response) {
  let data = {
    season: "",
    why: [],
    palettes: {},
    clothing: [],
    makeup: [],
  };
  try {
    const parsed = typeof response === 'string' ? JSON.parse(response) : response;
    data.season = parsed.season || "";
    data.why = parsed.why ? [parsed.why] : [];
    const palettes = parsed.palettes || {};
    data.palettes = {
      primary: parsePaletteArray(palettes["Primary and Secondary Colors"]),
      warm: parsePaletteArray(palettes["Warm Tones"]),
      cool: parsePaletteArray(palettes["Cool Tones"]),
      neutral: parsePaletteArray(palettes["Neutral or Black Tones"]),
    };
    data.clothing = parsePaletteArray(parsed.clothing);
    data.makeup = Array.isArray(parsed.makeup) ? parsed.makeup : [];
    return data;
  } catch (e) {
    return data;
  }
}

// Swatches Row
const SwatchRow = ({ colors }) => {
  if (!colors || colors.length === 0) return null;
  return (
    <div className="palette-swatches" style={{ display: 'flex', flexWrap: 'wrap', gap: 35, margin: '18px 0', justifyContent: 'center', width: '100%' }}>
      {colors.map((c, i) => {
        const color = typeof c === 'string' ? parsePaletteArray([c])[0] : c;
        return (
          <div className="palette-swatch" key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 90, marginBottom: 8 }}>
            <div className="swatch-color" style={{ width: 48, height: 48, borderRadius: 12, border: '2px solid #e0d7fa', marginBottom: 8, boxShadow: '0 2px 8px #a084ee22', background: color.hex }} />
            <div className="swatch-label" style={{ fontSize: 15, color: '#222', fontWeight: 600, width:200, textAlign: 'center', wordBreak: 'break-all' }}>{color.name}</div>
            <div style={{ fontSize: 13, color: '#888', textAlign: 'center' }}>{color.hex}</div>
          </div>
        );
      })}
    </div>
  );
};

// Makeup Swatch Row
const MakeupSwatchRow = ({ makeup }) => {
  if (!makeup || makeup.length === 0) return null;
  return (
    <div className="palette-swatches" style={{ display: 'flex', flexWrap: 'wrap', gap: 18, margin: '18px 0', justifyContent: 'center', width: '100%' }}>
      {makeup.map((m, i) => (
        <div className="palette-swatch" key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 90, marginBottom: 8 }}>
          <div className="swatch-color" style={{ width: 48, height: 48, borderRadius: 12, border: '2px solid #e0d7fa', marginBottom: 8, boxShadow: '0 2px 8px #a084ee22', background: m.hex }} />
          <div className="swatch-label" style={{ fontSize: 15, color: '#222', fontWeight: 600, textAlign: 'center', wordBreak: 'break-all' }}>{m.part}: {m.name}</div>
          <div style={{ fontSize: 13, color: '#888', textAlign: 'center' }}>{m.hex}</div>
        </div>
      ))}
    </div>
  );
};

// Section Card
const SectionCard = ({ title, icon, bg, children }) => (
  <div style={{ background: bg, borderRadius: 18, boxShadow: "0 4px 32px #a084ee22", padding: "1.5rem 2rem", margin: "2rem 0" }}>
    <h3 style={{ display: "flex", alignItems: "center", fontWeight: 700, fontSize: 22, color: "#222", marginBottom: 18 }}>
      {icon && <span style={{ fontSize: 24, marginRight: 10 }}>{icon}</span>}{title}
    </h3>
    {children}
  </div>
);

//  Quiz Response Display 
const AIResponseDisplayQuiz = ({ response }) => {
  
  const parsedResponse = safeParseLLMResponse(response && response.llm_response ? response.llm_response : response);
  const { season, why, palettes, clothing, makeup } = parseFormattedQuizResponse(parsedResponse);
  const seasonColor = seasonColors[season] || "#7b7be5";
  const hasContent = season || why.length || Object.values(palettes).some(p => p && p.length > 0) || clothing.length || makeup.length;

  return (
    <div className="palette-container" style={{
      padding: "3rem",
      fontFamily: "Inter, sans-serif",
      maxWidth: 1200,
      margin: "2rem auto",
      display: 'flex',
      flexDirection: 'row',
      gap: 32,
      alignItems: 'flex-start',
      flexWrap: 'wrap',
      background: '#fff',
      borderRadius: 18,
      boxShadow: '0 4px 32px #a084ee22'
    }}>
      {hasContent ? <>
        <div style={{ flex: 1, minWidth: 340 }}>
          /* Season Header */
          <div style={{ textAlign: "center", marginBottom: 32 }}>
            <div style={{ display: "inline-block", background: seasonColor + "22", color: seasonColor, borderRadius: 20, padding: "6px 22px", fontWeight: 700, fontSize: 40, marginBottom: 10 }}>
              {season ? `${season} Season` : "Season"}
            </div>
            /* Why this Season */
          {why.length > 0 && (
            <SectionCard title="Why this season?" bg="#fef9f2">
              {why.map((line, idx) => (
                <div key={idx} style={{ fontSize: 15, color: "#444", marginBottom: 8 }}>{line}</div>
              ))}
            </SectionCard>
          )}
            <h2 style={{ fontSize: 32, fontWeight: 900, color: "#222", margin: "10px 0 0 0" }}>Your Palette</h2>
            <div style={{ color: "#888", fontSize: 17 }}>Bold shades for your bold personality.</div>
          </div>

          

          {clothing.length > 0 && (
            <SectionCard title="More Colors" icon="🎨" bg="#f6f3ff">
              <SwatchRow colors={clothing} />
            </SectionCard>
          )}

          {makeup.length > 0 && (
            <SectionCard title="Colors to go with your Makeup palette" icon="💄" bg="#f6f3ff">
              <MakeupSwatchRow makeup={makeup} />
            </SectionCard>
          )}
        </div>
        <div style={{ flex: 2, minWidth: 400 }}>
          /* Palette Sections */
          {palettes.primary?.length > 0 && (
            <SectionCard title="Primary Colors" icon={<span style={{ color: '#3b82f6' }}>●</span>} bg="#fff">
              <SwatchRow colors={palettes.primary} />
            </SectionCard>
          )}
          {palettes.warm?.length > 0 && (
            <SectionCard title="Warm Tones" icon="🔥" bg="#fff5e6">
              <SwatchRow colors={palettes.warm} />
            </SectionCard>
          )}
          {palettes.cool?.length > 0 && (
            <SectionCard title="Cool Tones" icon="❄️" bg="#e6f0ff">
              <SwatchRow colors={palettes.cool} />
            </SectionCard>
          )}
          {palettes.neutral?.length > 0 && (
            <SectionCard title="Neutral & Black Tones" icon="◑" bg="#f7fafd">
              <SwatchRow colors={palettes.neutral} />
            </SectionCard>
          )}
        </div>
      </> : (
        <pre style={{
          color: '#888',
          background: '#f8f8fa',
          padding: 24,
          borderRadius: 12,
          textAlign: 'center',
          width: '100%'
        }}>{typeof response === 'string' ? response : JSON.stringify(response, null, 2) || "No response available"}</pre>
      )}
    </div>
  );
};

export default AIResponseDisplayQuiz; 