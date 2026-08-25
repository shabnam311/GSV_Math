"use client";

import { useState, useRef } from "react";
import { Upload, Image as ImageIcon, Send, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";

export default function Home() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    if (file.size > 10 * 1024 * 1024) {
      setError("Image must be smaller than 10MB");
      return;
    }

    setImageFile(file);
    setError(null);
    setResult(null);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!imagePreview || !question.trim()) {
      setError("Please provide both an image and a question.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const base64Data = imagePreview.split(',')[1];
      const backendUrl = process.env.NEXT_PUBLIC_MODAL_BACKEND_URL;
      
      if (!backendUrl) {
        throw new Error("Backend URL is not configured. Please set NEXT_PUBLIC_MODAL_BACKEND_URL.");
      }

      const res = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          image_base64: base64Data, 
          question: question.trim() 
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message || "An error occurred connecting to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-20">
      {/* Hero Section */}
      <div className="bg-indigo-600 text-white py-16 px-4 text-center shadow-lg">
        <h1 className="text-4xl md:text-5xl font-bold mb-4 flex items-center justify-center gap-3">
          <span className="text-4xl">??</span> GSV-Math
        </h1>
        <p className="text-xl md:text-2xl font-light text-indigo-100 max-w-2xl mx-auto">
          Grounded Self-Verifying Math VQA
        </p>
      </div>

      <div className="max-w-4xl mx-auto px-4 mt-12 grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Input Panel */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <ImageIcon className="w-5 h-5 text-indigo-500" /> Upload Diagram
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div 
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${imagePreview ? 'border-indigo-300 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400 bg-slate-50'}`}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="Preview" className="mx-auto max-h-48 object-contain rounded-lg shadow-sm" />
              ) : (
                <div className="flex flex-col items-center text-slate-500">
                  <Upload className="w-10 h-10 mb-3 text-slate-400" />
                  <p className="font-medium">Click to upload image</p>
                  <p className="text-sm mt-1">JPEG, PNG up to 10MB</p>
                </div>
              )}
              <input type="file" ref={fileInputRef} className="hidden" accept="image/jpeg, image/png, image/webp" onChange={handleImageChange} />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Question</label>
              <textarea 
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="What is the value of x in the diagram?"
                className="w-full rounded-xl border border-slate-300 p-4 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 min-h-[100px] outline-none transition-all resize-none"
              />
            </div>

            <button 
              type="submit" 
              disabled={loading || !imageFile || !question.trim()}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-4 px-6 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              {loading ? 'Solving (First run may take 60s+ to wake server)...' : 'Solve'}
            </button>
            
            {error && (
              <div className="bg-red-50 text-red-700 p-4 rounded-xl flex items-start gap-3 border border-red-100">
                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <p className="text-sm">{error}</p>
              </div>
            )}
          </form>
        </div>

        {/* Results Panel */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-500" /> Results
          </h2>

          {!result && !loading && (
            <div className="h-[400px] flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-100 rounded-xl">
              <p>Submit a question to see the results here.</p>
            </div>
          )}

          {loading && (
            <div className="h-[400px] flex flex-col items-center justify-center text-indigo-500 space-y-4">
              <Loader2 className="w-10 h-10 animate-spin" />
              <p className="font-medium animate-pulse">Running CISC Voting...</p>
            </div>
          )}

          {result && (
            <div className="space-y-6">
              <div className="bg-emerald-50 border border-emerald-200 p-5 rounded-xl">
                <h3 className="text-sm font-semibold text-emerald-800 uppercase tracking-wider mb-2">Final Answer</h3>
                <p className="text-2xl font-bold text-emerald-950">{result.answer}</p>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">Reasoning Trace</h3>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-slate-700 text-sm whitespace-pre-wrap font-mono h-48 overflow-y-auto">
                  {result.reasoning}
                </div>
              </div>

              {result.vote_distribution && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">Vote Distribution</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(result.vote_distribution).map(([ans, count]: [string, any]) => (
                      <div key={ans} className="bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200 text-sm">
                        <span className="font-semibold text-slate-700">{ans}:</span> <span className="text-slate-500">{count} votes</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {result.note && (
                <p className="text-xs text-slate-400 text-center mt-4">{result.note}</p>
              )}
            </div>
          )}
        </div>
      </div>
      
      <footer className="max-w-4xl mx-auto mt-12 text-center text-slate-400 text-sm pb-8">
        <p>Built with Next.js, Modal, and Qwen2.5-VL.</p>
      </footer>
    </main>
  );
}
