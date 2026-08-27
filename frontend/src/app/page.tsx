"use client";
import { useState, useRef } from "react";

const SAMPLES = {
  triangle: {
    question: "Find the area of the triangle with base 10 and height 5.",
    svg: <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><polygon points="10,90 90,90 50,10" fill="none" stroke="black" stroke-width="2"/><text x="40" y="95">10</text><line x1="50" y1="10" x2="50" y2="90" stroke="black" stroke-dasharray="2,2"/><text x="55" y="55">5</text></svg>
  },
  bars: {
    question: "What is the difference between group A and group B?",
    svg: <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><line x1="10" y1="90" x2="90" y2="90" stroke="black"/><rect x="20" y="30" width="20" height="60" fill="#333"/><text x="30" y="25" text-anchor="middle">A</text><rect x="60" y="50" width="20" height="40" fill="#777"/><text x="70" y="45" text-anchor="middle">B</text></svg>
  },
  angles: {
    question: "If the two angles are supplementary and one is 65°, what is the other?",
    svg: <svg viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg"><line x1="20" y1="80" x2="180" y2="80" stroke="black" stroke-width="2"/><line x1="90" y1="80" x2="60" y2="20" stroke="black" stroke-width="2"/><text x="65" y="75" font-size="12">65°</text><text x="110" y="75" font-size="12">?</text></svg>
  }
};

const svgToBase64Png = async (svgStr: string): Promise<string> => {
  return new Promise((resolve) => {
    const canvas = document.createElement("canvas");
    canvas.width = 400; canvas.height = 400;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      ctx?.drawImage(img, 0, 0, 400, 400);
      resolve(canvas.toDataURL("image/png"));
    };
    img.src = "data:image/svg+xml;base64," + btoa(svgStr);
  });
};

export default function Home() {
  const [imageBase64, setImageBase64] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [imagePreview, setImagePreview] = useState("");
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<"idle" | "running" | "graded" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [numSamples, setNumSamples] = useState("3");

  const backendUrl = process.env.NEXT_PUBLIC_MODAL_BACKEND_URL || "";
  const notConfigured = !backendUrl;

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImage = (dataUrl: string) => {
    setImagePreview(dataUrl);
    setImageBase64(dataUrl.split(",")[1]);
    setImageUrl("");
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => handleImage(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const onDrop = (e: React.DragEvent) => {
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
        throw new Error(Server responded : );
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
      return <div className="placeholder">Nothing graded yet - upload a diagram and ask a question above.</div>;
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
      const answer = result.answer ?? "-";
      const reasoning = result.reasoning as string | undefined;
      const grounding = typeof result.grounding_confidence === "number" ? result.grounding_confidence : null;
      const votes = (result.vote_distribution as Record<string, unknown>) || {};
      const maxVote = Math.max(1, ...Object.values(votes).map(v => v as number));
      
      const clipScore = typeof result.clip_alignment_score === "number" ? result.clip_alignment_score : null;
      const owlScore = typeof result.owl_grounding_score === "number" ? result.owl_grounding_score : null;
      const sympyPassed = typeof result.symbolic_check_passed === "boolean" ? result.symbolic_check_passed : null;
      
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
                      <div className="vote-fill" style={{ width: ${((v as number) / maxVote * 100).toFixed(0)}% }}></div>
                    </div>
                    <span className="vote-val">{Number(v as number).toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Signal Breakdown */}
          {(clipScore !== null || sympyPassed !== null || owlScore !== null) && (
            <div className="mt-8 border-t border-[var(--paper-line-soft)] pt-6 pb-4">
              <div className="votes-label" style={{marginBottom: "1rem"}}>Signal Breakdown</div>
              <div style={{display: "flex", flexDirection: "column", gap: "12px"}}>
                <div style={{display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.875rem"}}>
                  <span style={{fontFamily: "var(--font-sans)", color: "var(--ink)"}}>OWL-ViT Object Grounding</span>
                  {owlScore !== null ? (
                    <span style={{fontFamily: "var(--font-mono)", fontSize: "0.75rem", backgroundColor: "var(--paper-line-soft)", padding: "4px 8px", borderRadius: "4px", color: "var(--ink-soft)"}}>
                      {(owlScore * 100).toFixed(1)}%
                    </span>
                  ) : (
                    <span style={{fontFamily: "var(--font-mono)", fontSize: "0.75rem", backgroundColor: "var(--paper-line-soft)", padding: "4px 8px", borderRadius: "4px", color: "var(--ink-soft)"}}>ACTIVE</span>
                  )}
                </div>
                {clipScore !== null && (
                  <div style={{display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.875rem"}}>
                    <span style={{fontFamily: "var(--font-sans)", color: "var(--ink)"}}>CLIP Semantic Alignment</span>
                    <span style={{fontFamily: "var(--font-mono)", fontSize: "0.75rem", backgroundColor: "var(--paper-line-soft)", padding: "4px 8px", borderRadius: "4px", color: "var(--ink-soft)"}}>
                      {(clipScore * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                {sympyPassed !== null && (
                  <div style={{display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.875rem"}}>
                    <span style={{fontFamily: "var(--font-sans)", color: "var(--ink)"}}>Sympy Symbolic Check</span>
                    <span style={{
                      fontFamily: "var(--font-mono)", 
                      fontSize: "0.75rem", 
                      padding: "4px 8px", 
                      borderRadius: "4px",
                      backgroundColor: sympyPassed ? '#d1e7dd' : '#f8d7da',
                      color: sympyPassed ? '#0f5132' : '#842029'
                    }}>
                      {sympyPassed ? 'PASSED' : 'FAILED'}
                    </span>
                  </div>
                )}
                {sympyPassed === null && (
                  <div style={{display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.875rem"}}>
                    <span style={{fontFamily: "var(--font-sans)", color: "var(--ink)"}}>Sympy Symbolic Check</span>
                    <span style={{fontFamily: "var(--font-mono)", fontSize: "0.75rem", backgroundColor: "var(--paper-line-soft)", padding: "4px 8px", borderRadius: "4px", color: "var(--ink-soft)"}}>N/A</span>
                  </div>
                )}
              </div>
            </div>
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
        <div className="links"><a href="https://github.com/shabnam311/GSV_Math" target="_blank" rel="noopener noreferrer">source  </a></div>
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
                className={dropzone }
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
                    <div className="dz-sub">PNG / JPG - UP TO 10MB</div>
                  </div>
                )}
              </div>

              <div className="samples">
                <div className="row-label">or try one -</div>
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
        <span>Qwen2.5-VL-7B - OWL-ViT grounding - MathVista testmini</span>
        <span>{status}</span>
      </footer>
    </>
  );
}
