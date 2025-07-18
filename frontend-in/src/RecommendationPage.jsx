import React from "react";
import ClothingImageResults from "./ClothingImageResults";

const groupByCategory = (products) => {
  const grouped = {};
  products.forEach((item) => {
    const cat = item.category || "Other";
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(item);
  });
  return grouped;
};

const RecommendationPage = ({ productList, onClose }) => {
  const grouped = groupByCategory(productList);

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        background: "rgba(123,123,229,0.10)",
        zIndex: 2000,
        overflowY: "auto",
        padding: "32px 0"
      }}
    >
      <div
        style={{
          background: "#fff",
          borderRadius: 18,
          boxShadow: "0 4px 32px #a084ee22",
          maxWidth: 1100,
          width: "95vw",
          margin: "0 auto",
          padding: "2.5rem 2.5rem 2rem 2.5rem",
          position: "relative"
        }}
      >
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: 24,
            right: 24,
            background: "none",
            border: "none",
            fontSize: 28,
            color: "#bbb",
            cursor: "pointer"
          }}
        >
          &times;
        </button>
        <h2 style={{ fontWeight: 900, fontSize: 32, marginBottom: 24 }}>Recommended for You</h2>
        {Object.keys(grouped).map((cat) => (
          <div key={cat} style={{ marginBottom: 40 }}>
            <h3 style={{ fontWeight: 700, fontSize: 24, marginBottom: 18 }}>{cat}</h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: 24
              }}
            >
              {grouped[cat].map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    background: "#f6f7fa",
                    borderRadius: 12,
                    padding: 16,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center"
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 8 }}>{item.product}</div>
                  <div style={{ color: "#888", fontSize: 14, marginBottom: 8, textAlign: "center" }}>{item.description}</div>
                  <ClothingImageResults query={item.query} />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RecommendationPage;
