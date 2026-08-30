"""
GSV-Math CPU Backend — FastAPI proxy for llama-server.

This app runs on HF Spaces (CPU Basic, 2 vCPU / 16GB RAM) and forwards
inference requests to llama-server (llama.cpp) running as a subprocess.

The API contract matches the original Modal backend exactly so the
Vercel frontend needs zero code changes.
"""

import os
import re
import base64
import io
import logging
import httpx
from PIL import Image
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLAMA_SERVER = "http://127.0.0.1:8081"

app = FastAPI(title="GSV-Math CPU Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gsv-math.vercel.app", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Auth ----------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set!")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API Key credentials")

# ---------- Answer extraction (ported from pipeline/answer_extraction.py) ----------
FINAL_ANSWER_PATTERNS = [
    r'\\boxed\{([^}]*)\}',
    r'[Ff]inal\s*[Aa]nswer\s*[:\-]?\s*(.{1,80})',
    r'[Tt]herefore[,\s]+(?:the\s+)?(?:answer|value|result)\s+is\s*[:\-]?\s*(.{1,80})',
    r'[Tt]he\s+answer\s+is\s*[:\-]?\s*(.{1,80})',
    r'[Ss]o\s+the\s+answer\s+is\s*[:\-]?\s*(.{1,80})',
    r'=\s*(\S+)\s*$',
]

def extract_answer(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        return str(raw_text)
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, raw_text, re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
    idx = raw_text.lower().rfind("answer is")
    if idx != -1:
        ans = raw_text[idx + 9:].strip().replace(":", "").replace(".", "").strip()
        if ans:
            return ans
    words = raw_text.split()
    if words:
        return words[-1]
    return raw_text[-300:]

def normalize_answer(ans: str) -> str:
    if not ans:
        return ""
    ans = ans.strip().lower()
    for p in ["x=", "y=", "z=", "v=", "a=", "b=", "c="]:
        if ans.startswith(p):
            ans = ans[len(p):].strip()
    try:
        f = float(ans)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return ans

# ---------- Main endpoint ----------
@app.post("/", dependencies=[Depends(verify_api_key)])
async def solve_route(request: Request):
    try:
        data = await request.json()
        image_b64 = data.get("image_base64")
        image_url = data.get("image_url")
        question = data.get("question")

        if not (image_b64 or image_url) or not question:
            return {"error": "Missing image or question in payload"}

        # Resolve image to base64
        if image_url and not image_b64:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
                image_b64 = base64.b64encode(resp.content).decode()

        # Resize image to match training resolution (~396px)
        try:
            img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            if max(img.size) > 396:
                img.thumbnail((396, 396))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Image resize failed, using original: {e}")

        # Build OpenAI-compatible chat request for llama-server
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]

        # Call llama-server (num_samples=1 for CPU speed)
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{LLAMA_SERVER}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "stream": False
                }
            )
            resp.raise_for_status()
            result = resp.json()

        raw_text = result["choices"][0]["message"]["content"]
        answer = normalize_answer(extract_answer(raw_text))

        logger.info(f"Answer: {answer}")

        return {
            "answer": answer,
            "reasoning": raw_text,
            "vote_distribution": {answer: 1},
            "owl_grounding_score": None,
            "clip_alignment_score": None,
            "symbolic_check_passed": None,
            "note": "Running on CPU (llama.cpp) — CLIP/OWL-ViT/SymPy verification disabled for speed."
        }

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok", "message": "GSV-Math CPU backend is reachable."}
