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
    padding: "40px 24px 32px 24px",
    background: "linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)",
    borderRadius: "32px 32px 0 0",
    marginTop: "32px"
  }}>
    <div style={{ maxWidth: 1200, margin: "0 auto" }}>
      <h2 style={{ 
        textAlign: "center", 
        fontSize: "2rem", 
        fontWeight: 900, 
        marginBottom: "4px", 
        color: "#fff",
        letterSpacing: "1px",
        opacity: 0,
        animation: "fadeInUp 0.8s 0.2s forwards"
      }}>
        Color Season Types
      </h2>
      <p style={{
        textAlign: "center",
        fontSize: "1rem",
        color: "rgba(255,255,255,0.8)",
        marginBottom: "32px",
        fontWeight: 500,
        opacity: 0,
        animation: "fadeInUp 0.8s 0.4s forwards"
      }}>
        Discover your perfect color palette based on seasonal analysis
      </p>
      
      <div style={{
        display: "flex",
        justifyContent: "center",
        gap: "20px",
        flexWrap: "wrap"
      }}>
        {seasonData.map((season, index) => (
          <div key={season.name} className="season-card" style={{
            opacity: 0,
            animation: `fadeInUp 0.8s ${0.6 + index * 0.1}s forwards`
          }}
          >
            <div style={{
              width: "60px",
              height: "60px",
              borderRadius: "50%",
              background: `linear-gradient(135deg, ${season.color} 0%, ${season.color}dd 100%)`,
              marginBottom: "16px",
              boxShadow: `0 6px 20px ${season.color}33`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "24px",
              color: "#fff",
              fontWeight: "bold"
            }}>
              {season.name.charAt(0)}
            </div>
            
            <h3 style={{ 
              fontWeight: 800, 
              fontSize: "1.2rem", 
              color: "#222", 
              marginBottom: "6px",
              letterSpacing: "0.5px"
            }}>
              {season.name}
            </h3>
            
            <p style={{ 
              color: "#666", 
              fontSize: "0.9rem", 
              marginBottom: "16px",
              fontWeight: 500,
              lineHeight: "1.4"
            }}>
              {season.description}
            </p>
            
            <div style={{ 
              display: "flex", 
              gap: "8px", 
              justifyContent: "center",
              flexWrap: "wrap"
            }}>
              {season.swatches.map((swatch, i) => (
                <div key={i} style={{
                  width: "22px",
                  height: "22px",
                  borderRadius: "50%",
                  background: swatch,
                  border: "2px solid rgba(255,255,255,0.8)",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                  transition: "transform 0.2s ease"
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = "scale(1.2)";
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = "scale(1)";
                }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

export default SeasonTypes;