import React, { useEffect, useState } from "react";

const SERPAPI_KEY = "22b051b2e9182bcb71d59fb43727c5fcb0374fdffcae2697defebbbe6bba0952"; // Replace with your real key or use env for production

const ClothingImageResults = ({ query }) => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query) return;
    setLoading(true);
    fetch(
      `http://localhost:8000/api/serpapi-proxy?q=${encodeURIComponent(query)}`
    )
      .then((res) => res.json())
      .then((data) => {
        setImages(data.images_results?.slice(0, 5) || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [query]);

  if (!query) return null;
  if (loading) return <div style={{ color: "#aaa", marginTop: 12 }}>Loading outfit images...</div>;

  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 16 }}>
      {images.map((img, idx) => (
        <img
          key={idx}
          src={img.thumbnail}
          alt={img.title || "outfit"}
          style={{
            width: 90,
            height: 90,
            objectFit: "cover",
            borderRadius: 8,
            boxShadow: "0 2px 8px #eee",
            background: "#f6f7fa",
          }}
        />
      ))}
    </div>
  );
};

export default ClothingImageResults; 