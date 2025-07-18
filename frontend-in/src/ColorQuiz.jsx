import React, { useState } from 'react';
import AIResponseDisplayQuiz from './AIResponseDisplayQuiz';

const questions = [
  {
    label: "What is your skin’s natural undertone? (Try looking at the veins on your wrist or the jewelry that flatters you most.)",
    name: "undertone",
    options: [
      "Cool (blue/pink undertones, silver jewelry suits me)",
      "Warm (yellow/golden undertones, gold jewelry suits me)",
      "Neutral (a mix of both or hard to tell)"
    ]
  },
  {
    label: "What is your natural hair color (not dyed)?",
    name: "hairColor",
    options: [
      "Very light blonde",
      "Warm blonde or strawberry blonde",
      "Light brown",
      "Medium brown",
      "Dark brown or black",
      "Red or auburn",
      "Gray or white"
    ]
  },
  {
    label: "What is your natural eye color?",
    name: "eyeColor",
    options: [
      "Light blue or green",
      "Gray or hazel",
      "Amber or golden brown",
      "Deep brown or black"
    ]
  },
  {
    label: "How does your skin react to the sun?",
    name: "sunReaction",
    options: [
      "Burns easily, rarely tans (likely cool undertone)",
      "Burns slightly, then tans (neutral)",
      "Tans easily, rarely burns (warm undertone)"
    ]
  },
  {
    label: "What is your skin depth/tone?",
    name: "skinDepth",
    options: [
      "Fair/light",
      "Medium",
      "Olive",
      "Deep/dark"
    ]
  },
  {
    label: "What color are the veins on your wrist (in natural light)?",
    name: "veinColor",
    options: [
      "Bluish/purplish",
      "Greenish",
      "A mix of both / hard to tell"
    ]
  }
];

const ColorQuiz = ({ onSubmit, loading, result }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [page, setPage] = useState(0);

  const handleChange = (name, value) => {
    setAnswers({ ...answers, [name]: value });
  };

  const handleNext = () => {
    if (answers[questions[page].name]) {
      setPage(page + 1);
      setSubmitted(false);
    } else {
      setSubmitted(true);
    }
  };

  const handleBack = () => {
    setPage(page - 1);
    setSubmitted(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true);
    if (Object.keys(answers).length === questions.length && answers[questions[page].name]) {
      onSubmit(answers);
    }
  };

  const q = questions[page];
  const total = questions.length;

  // Modal style for horizontal layout when result is present
  const modalStyle = result ? {
    background: 'rgba(255,255,255,0.98)',
    borderRadius: 18,
    boxShadow: '0 4px 32px #0002',
    padding: '2.5rem',
    maxWidth: 1200,
    width: '100%',
    textAlign: 'center',
    position: 'relative',
    margin: '2rem auto',
    display: 'flex',
    flexDirection: 'row',
    gap: 32,
    alignItems: 'flex-start',
    justifyContent: 'center',
    flexWrap: 'wrap'
  } : {
    background: 'rgba(255,255,255,0.98)',
    borderRadius: 18,
    boxShadow: '0 4px 32px #0002',
    padding: '2.5rem 2.5rem 2rem 2.5rem',
    maxWidth: 520,
    width: '100%',
    textAlign: 'center',
    position: 'relative',
    margin: '2rem auto'
  };

  return (
    <div className="upload-modal" style={modalStyle}>
      <button onClick={() => window.location.reload()} style={{ position: 'absolute', top: 18, right: 18, background: 'none', border: 'none', fontSize: 22, color: '#bbb', cursor: 'pointer' }}>&times;</button>
      {/* Go back to home page button, only when result is shown */}
      {result && (
        <button
          onClick={() => window.location.reload()}
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
      {/* Form and Result Side by Side if result is present */}
      <div style={{ flex: 1, minWidth: 340 }}>
        {!result && (
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0, maxWidth: 500, margin: '0 auto', background: '#fff', borderRadius: 16, boxShadow: '0 4px 24px #a084ee22', padding: 36, position: 'relative' }}>
            {/* Progress Bar */}
            <div style={{ width: '100%', marginBottom: 32 }}>
              <div style={{ fontWeight: 700, color: '#7b7be5', fontSize: 18, marginBottom: 8 }}>Question {page + 1} of {total}</div>
              <div style={{ height: 8, background: '#f3f0fa', borderRadius: 6, overflow: 'hidden' }}>
                <div style={{ width: `${((page + 1) / total) * 100}%`, height: '100%', background: 'linear-gradient(90deg, #a084ee 0%, #f857a6 100%)', borderRadius: 6, transition: 'width 0.3s' }} />
              </div>
            </div>
            {/* Question */}
            <div style={{ fontWeight: 700, fontSize: 22, color: '#222', marginBottom: 24, textAlign: 'center', minHeight: 60 }}>{q.label}</div>
            {/* Options */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, width: '100%', marginBottom: 24 }}>
              {q.options.map(opt => (
                <label key={opt} style={{
                  fontWeight: 500,
                  color: answers[q.name] === opt ? '#fff' : '#7b7be5',
                  background: answers[q.name] === opt ? 'linear-gradient(90deg, #a084ee 0%, #f857a6 100%)' : '#f6f3ff',
                  border: answers[q.name] === opt ? '2px solid #a084ee' : '2px solid #e0d7fa',
                  borderRadius: 10,
                  padding: '14px 18px',
                  cursor: 'pointer',
                  fontSize: 16,
                  boxShadow: answers[q.name] === opt ? '0 2px 12px #a084ee33' : 'none',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10
                }}>
                  <input
                    type="radio"
                    name={q.name}
                    value={opt}
                    checked={answers[q.name] === opt}
                    onChange={() => handleChange(q.name, opt)}
                    style={{ marginRight: 12, accentColor: '#a084ee', width: 18, height: 18 }}
                    required
                  />
                  {opt}
                </label>
              ))}
            </div>
            {submitted && !answers[q.name] && <div style={{ color: 'red', fontSize: 14, marginBottom: 12 }}>Please select an option to continue.</div>}
            {/* Navigation Buttons */}
            <div style={{ display: 'flex', gap: 16, width: '100%', justifyContent: 'space-between', marginTop: 8 }}>
              <button type="button" onClick={handleBack} disabled={page === 0} style={{ background: '#f6f3ff', color: '#7b7be5', border: 'none', borderRadius: 8, padding: '10px 24px', fontWeight: 700, fontSize: 16, cursor: page === 0 ? 'not-allowed' : 'pointer', opacity: page === 0 ? 0.5 : 1 }}>Back</button>
              {page < total - 1 ? (
                <button type="button" onClick={handleNext} style={{ background: 'linear-gradient(90deg, #a084ee 0%, #f857a6 100%)', color: '#fff', border: 'none', borderRadius: 8, padding: '10px 32px', fontWeight: 700, fontSize: 16, cursor: 'pointer' }}>Next</button>
              ) : (
                <button type="submit" disabled={loading} style={{ background: 'linear-gradient(90deg, #a084ee 0%, #f857a6 100%)', color: '#fff', border: 'none', borderRadius: 8, padding: '10px 32px', fontWeight: 700, fontSize: 16, cursor: 'pointer' }}>{loading ? 'Analyzing...' : 'Submit'}</button>
              )}
            </div>
          </form>
        )}
      </div>
      {result && (
        <div style={{ flex: 2, minWidth: 400 }}>
          <div className="result-area-horizontal" style={{
            marginTop: 0,
            display: 'flex',
            gap: 32,
            alignItems: 'flex-start',
            justifyContent: 'center',
            maxWidth: 1200,
            marginLeft: 'auto',
            marginRight: 'auto',
            width: '100%'
          }}>
            <AIResponseDisplayQuiz response={result} />
          </div>
        </div>
      )}
    </div>
  );
};

export default ColorQuiz; 