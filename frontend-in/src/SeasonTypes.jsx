import React from "react";

const seasonData = [
  {
    name: "Spring",
    color: "#F4B400",
    description: "Warm, clear and bright colors",
    swatches: ["#FFC107", "#FF9800", "#8BC34A", "#03A9F4"]
  },
  {
    name: "Summer",
    color: "#A084EE",
    description: "Cool, soft and muted colors",
    swatches: ["#A084EE", "#90CAF9", "#B39DDB", "#B0BEC5", "#E1BEE7"]
  },
  {
    name: "Autumn",
    color: "#D2691E",
    description: "Warm, rich and deep colors",
    swatches: ["#F44336", "#FF5722", "#FFEB3B", "#8BC34A", "#D2691E"]
  },
  {
    name: "Winter",
    color: "#222831",
    description: "Cool, clear and intense colors",
    swatches: ["#000", "#1976D2", "#E53935", "#8E24AA", "#00B8D4"]
  }
];

const SeasonTypes = () => (
  <div style={{
    width: "100%",
   
    padding: "48px 0 32px 0"
  }}>
    <h2 style={{ textAlign: "center", fontSize: 32, fontWeight: 900, marginBottom: 36, color: "#222" }}>
      Color Season Types
    </h2>
    <div style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      gap: 36,
      flexWrap: "nowrap",
      maxWidth: 1200,
      margin: "0 auto"
    }}>
      {seasonData.map((season) => (
        <div key={season.name} style={{
          background: "#fff",
          borderRadius: 20,
          boxShadow: "0 4px 24px #0001",
          padding: "32px 36px 28px 36px",
          minWidth: 240,
          maxWidth: 260,
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          alignItems: "center"
        }}>
          <div style={{
            width: 56,
            height: 56,
            borderRadius: "50%",
            background: season.color,
            marginBottom: 18,
            boxShadow: "0 2px 8px #0002"
          }} />
          <div style={{ fontWeight: 700, fontSize: 22, color: "#222", marginBottom: 6 }}>{season.name}</div>
          <div style={{ color: "#666", fontSize: 15, marginBottom: 16 }}>{season.description}</div>
          <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
            {season.swatches.map((swatch, i) => (
              <div key={i} style={{
                width: 22,
                height: 22,
                borderRadius: "50%",
                background: swatch,
                border: "1.5px solid #eee"
              }} />
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
);

export default SeasonTypes; 