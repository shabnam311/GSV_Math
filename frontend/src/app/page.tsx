"use client";

import { useState, useRef, DragEvent, ChangeEvent } from "react";

const SAMPLES = {
  triangle: {
    question: "What is the area of the triangle, in square units?",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" style="background:#fff">
        <polygon points="60,240 340,240 200,40" fill="none" stroke="#1f2430" stroke-width="2.5"/>
        <text x="185" y="270" font-family="monospace" font-size="15" fill="#1f2430">base = 280</text>
        <text x="205" y="140" font-family="monospace" font-size="15" fill="#1f2430">h = 200</text>
      </svg>`
  },
  bars: {
    question: "Which category has the highest value?",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" style="background:#fff">
        <rect x="40" y="180" width="50" height="90" fill="#3a5f7d"/>
        <rect x="130" y="120" width="50" height="150" fill="#3a5f7d"/>
        <rect x="220" y="60" width="50" height="210" fill="#a3372a"/>
        <rect x="310" y="150" width="50" height="120" fill="#3a5f7d"/>
        <line x1="20" y1="270" x2="380" y2="270" stroke="#1f2430" stroke-width="1.5"/>
        <text x="45" y="290" font-family="monospace" font-size="13" fill="#1f2430">A</text>
        <text x="135" y="290" font-family="monospace" font-size="13" fill="#1f2430">B</text>
        <text x="225" y="290" font-family="monospace" font-size="13" fill="#1f2430">C</text>
        <text x="315" y="290" font-family="monospace" font-size="13" fill="#1f2430">D</text>
      </svg>`
  },
  angles: {
    question: "If the two angles are supplementary and one is 65°, what is the other?",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" style="background:#fff">
        <line x1="20" y1="200" x2="380" y2="200" stroke="#1f2430" stroke-width="2.5"/>
        <line x1="200" y1="200" x2="120" y2="60" stroke="#1f2430" stroke-width="2.5"/>
        <text x="140" y="180" font-family="monospace" font-size="15" fill="#1f2430">65°</text>
        <text x="230" y="180" font-family="monospace" font-size="15" fill="#1f2430">?</text>
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
  const [imageUrl, setImageUrl] = useState<string>('');
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<"idle" | "running" | "graded" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [clipScore, setClipScore] = useState<number | null>(null);
  const [sympyPassed, setSympyPassed] = useState<boolean | null>(null);
  const [numSamples, setNumSamples] = useState("3");
  
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
          image_url: imageUrl || null,
          question: question.trim(),
          num_samples: parseInt(numSamples, 10)
        })
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server responded ${resp.status}: ${text.slice(0, 200)}`);
      }
      
      const data = await resp.json();
      if (data.error) throw new Error(data.error);
      
      setResult(data);
      setStatus("graded");
    } catch (err: unknown) {
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  const renderResultContent = () => {
    if (status === "idle") {
      return <div className="placeholder">Nothing graded yet — upload a diagram and ask a question above.</div>;
    }
    
    if (status === "running") {
      return (
        <div className="loading">
          <div className="spinner"></div>
          <span>generating and checking against the diagram</span>
        </div>
      );
    }
    
    if (status === "error") {
      return <div className="error">{errorMsg}</div>;
    }
    
    if (result) {
      const answer = result.answer ?? "—";
      const reasoning = result.reasoning as string | undefined;
      const grounding = typeof result.grounding_confidence === "number" ? result.grounding_confidence : null;
      const votes = (result.vote_distribution as Record<string, unknown>) || {};
      const maxVote = Math.max(1, ...Object.values(votes).map(v => v as number));
      
      return (
        <>
          <div className="grade-row">
            {grounding !== null && (
              <div className="stamp-circle">
                <div className="stamp-pct">{(grounding * 100).toFixed(0)}%</div>
                <div className="stamp-caption">grounded</div>
              </div>
            )}
            <div>
              <div className="answer-value">{answer as string}</div>
              <div className="answer-caption">final answer</div>
            </div>
          </div>
          
          {reasoning && (
            <>
              <div className="reasoning-label">reasoning trace</div>
              <div className="reasoning">{reasoning}</div>
            </>
          )}
          
          {Object.keys(votes).length > 0 && (
            <>
              <div className="votes-label">vote distribution</div>
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
            </>
          )}
        </>
      );
    }
  };

  return (
    <>
      <header>
        <div className="mark">
          <span className="mark-glyph">gsv</span>
          <span className="mark-name">math verification</span>
        </div>
        <div className="links"><a href="https://github.com/shabnam311/GSV_Math" target="_blank" rel="noopener noreferrer">source →</a></div>
      </header>

      <div className="intro">
        <h1>Every answer here is <span className="u">checked</span> against what&apos;s actually drawn on the page.</h1>
      </div>

      <main>
        <div className="sheet">
          <span className="tape left"></span>
          <span className="tape right"></span>

          <div className="sheet-row">
            <div>
              <div className="field-label"><span className="n">1</span>diagram</div>
              
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
                    <div className="dz-title">click or drop an image</div>
                    <div className="dz-sub">PNG / JPG · UP TO 10MB</div>
                  </div>
                )}
              </div>

              <div className="samples">
                <div className="row-label">or try one —</div>
                <div className="chip-row">
                  <button className="chip" onClick={() => loadSample("triangle")}>triangle</button>
                  <button className="chip" onClick={() => loadSample("bars")}>bar chart</button>
                  <button className="chip" onClick={() => loadSample("angles")}>angle pair</button>
                </div>
              </div>
            </div>

            <div>
              <div className="field-label"><span className="n">2</span>question</div>
              <textarea 
                className="qline" 
                id="question" 
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="what is the area of the shaded triangle?"
              />

              <div className="config-row">
                <span>samples per answer</span>
                <select id="numSamples" value={numSamples} onChange={(e) => setNumSamples(e.target.value)}>
                  <option value="1">1</option>
                  <option value="3">3</option>
                  <option value="5">5</option>
                </select>
              </div>

              <button 
                className="solve" 
                onClick={handleSolve}
                disabled={status === "running"}
              >
                Check my work
              </button>
              <div className="endpoint-note">
                {notConfigured ? "endpoint not configured" : backendUrl.replace(/^https?:\/\//, '')}
              </div>
            </div>
          </div>

          <div className="slip" id="slip">
            <div id="resultArea">
              {renderResultContent()}
            </div>
          </div>
        </div>
      </main>

      <footer>
        <span>Qwen2.5-VL-7B · OWL-ViT grounding · MathVista testmini</span>
        <span>{status}</span>
      </footer>
    </>
  );
}




