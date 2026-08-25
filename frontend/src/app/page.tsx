"use client";

import { useState, useRef, DragEvent, ChangeEvent } from "react";

const SAMPLES = {
  triangle: {
    question: "What is the area of the triangle, in square units?",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" style="background:#fff">
        <polygon points="60,240 340,240 200,40" fill="none" stroke="#111" stroke-width="2.5"/>
        <text x="185" y="270" font-family="monospace" font-size="15" fill="#111">base = 280</text>
        <text x="205" y="140" font-family="monospace" font-size="15" fill="#111">h = 200</text>
      </svg>`
  },
  bars: {
    question: "Which category has the highest value?",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" style="background:#fff">
        <rect x="40" y="180" width="50" height="90" fill="#333"/>
        <rect x="130" y="120" width="50" height="150" fill="#333"/>
        <rect x="220" y="60" width="50" height="210" fill="#c9a24a"/>
        <rect x="310" y="150" width="50" height="120" fill="#333"/>
        <line x1="20" y1="270" x2="380" y2="270" stroke="#111" stroke-width="1.5"/>
        <text x="45" y="290" font-family="monospace" font-size="13" fill="#111">A</text>
        <text x="135" y="290" font-family="monospace" font-size="13" fill="#111">B</text>
        <text x="225" y="290" font-family="monospace" font-size="13" fill="#111">C</text>
        <text x="315" y="290" font-family="monospace" font-size="13" fill="#111">D</text>
      </svg>`
  },
  angles: {
    question: "If the two angles are supplementary and one is 65°, what is the other?",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" style="background:#fff">
        <line x1="20" y1="200" x2="380" y2="200" stroke="#111" stroke-width="2.5"/>
        <line x1="200" y1="200" x2="120" y2="60" stroke="#111" stroke-width="2.5"/>
        <text x="140" y="180" font-family="monospace" font-size="15" fill="#111">65°</text>
        <text x="230" y="180" font-family="monospace" font-size="15" fill="#111">?</text>
      </svg>`
  }
};

function svgToBase64Png(svgString: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    const blob = new Blob([svgString], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    img.onload = function () {
      const canvas = document.createElement("canvas");
      canvas.width = 400;
      canvas.height = 300;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, 400, 300);
        ctx.drawImage(img, 0, 0);
      }
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/png"));
    };
    img.src = url;
  });
}

export default function Home() {
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const backendUrl = process.env.NEXT_PUBLIC_MODAL_BACKEND_URL;
  const notConfigured = !backendUrl;

  const handleImage = (dataUrl: string) => {
    setImagePreview(dataUrl);
    setImageBase64(dataUrl.split(",")[1]);
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      alert("File too large — 10MB max.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => handleImage(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => handleImage(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const loadSample = async (key: keyof typeof SAMPLES) => {
    const s = SAMPLES[key];
    const dataUrl = await svgToBase64Png(s.svg);
    handleImage(dataUrl);
    setQuestion(s.question);
  };

  const handleSolve = async () => {
    if (!imageBase64) { alert("Upload or pick a sample image first."); return; }
    if (!question.trim()) { alert("Enter a question."); return; }
    if (notConfigured) {
      setStatus("error");
      setErrorMsg("No backend endpoint set. Please configure NEXT_PUBLIC_MODAL_BACKEND_URL.");
      return;
    }

    setStatus("running");
    setResult(null);
    setErrorMsg("");

    try {
      const resp = await fetch(backendUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: imageBase64,
          question: question.trim(),
        })
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server responded ${resp.status}: ${text.slice(0, 200)}`);
      }
      
      const data = await resp.json();
      if (data.error) throw new Error(data.error);
      
      setResult(data);
      setStatus("done");
    } catch (err: unknown) {
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  const renderResultContent = () => {
    if (status === "idle") {
      return <div className="placeholder">Results will appear here once you submit an image and a question.</div>;
    }
    
    if (status === "running") {
      return (
        <div className="loading">
          <div className="spinner"></div>
          <span>generating and verifying (may take ~60s on cold start)</span>
        </div>
      );
    }
    
    if (status === "error") {
      return <div className="error">{errorMsg}</div>;
    }
    
    if (result) {
      const answer = result.answer ?? "—";
      const reasoning = result.reasoning ?? "";
      const votes = result.vote_distribution || {};
      const maxVote = Math.max(1, ...Object.values(votes) as number[]);
      
      return (
        <>
          <div className="answer-row">
            <span className="answer-value">{answer}</span>
            <span className="answer-tag">answer</span>
          </div>
          
          {reasoning && <div className="reasoning">{reasoning}</div>}
          
          {Object.keys(votes).length > 0 && (
            <div className="votes">
              {Object.entries(votes).sort((a: [string, unknown], b: [string, unknown]) => (b[1] as number) - (a[1] as number)).map(([k, v]: [string, unknown]) => (
                <div className="vote-row" key={k}>
                  <span className="vote-key">{k}</span>
                  <div className="vote-track">
                    <div className="vote-fill" style={{ width: `${((v as number) / maxVote * 100).toFixed(0)}%` }}></div>
                  </div>
                  <span className="vote-val">{Number(v as number).toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}
        </>
      );
    }
  };

  return (
    <div className="shell">
      <header>
        <div className="brand">GSV<span>·</span>Math</div>
        <nav>
          <a href="https://github.com/shabnam311/GSV_Math" target="_blank" rel="noopener noreferrer">source</a>
        </nav>
      </header>

      <div className="ui-grid">
        <section className="panel">
          <div className="panel-label">Input</div>

          <div 
            className={`dropzone ${isDragging ? "drag" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
            onDrop={onDrop}
            tabIndex={0}
            role="button"
            aria-label="Upload an image"
          >
            <input type="file" ref={fileInputRef} accept="image/*" onChange={onFileChange} />
            {imagePreview ? (
              <img src={imagePreview} alt="Uploaded diagram" />
            ) : (
              <div id="dzContent">
                <div className="dz-mark">+</div>
                <div className="dz-title">Upload an image</div>
                <div className="dz-sub">PNG or JPG, up to 10MB</div>
              </div>
            )}
          </div>

          <div className="samples">
            <button className="chip" onClick={() => loadSample("triangle")}>triangle</button>
            <button className="chip" onClick={() => loadSample("bars")}>bar chart</button>
            <button className="chip" onClick={() => loadSample("angles")}>angles</button>
          </div>

          <div className="field">
            <label className="field-label" htmlFor="question">Question</label>
            <textarea 
              id="question" 
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What is the area of the shaded region?"
            />
          </div>

          <button 
            className="solve" 
            onClick={handleSolve}
            disabled={status === "running"}
          >
            Run
          </button>
          <div className="endpoint-note">
            {notConfigured ? "endpoint not configured" : backendUrl.replace(/^https?:\/\//, '')}
          </div>
        </section>

        <section className="panel">
          <div className="panel-label">Output</div>
          <div id="resultArea">
            {renderResultContent()}
          </div>
        </section>
      </div>

      <footer>
        <span>Qwen2.5-VL · OWL-ViT · MathVista</span>
        <span>{status}</span>
      </footer>
    </div>
  );
}
