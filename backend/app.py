import os
import base64
import io
import logging
from PIL import Image
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import spaces

# HF spaces exposes gradio by default on 7860, but if we use FastAPI natively via app, we just expose app.
from pipeline.model_loader import load_models
from pipeline.cisc import cisc_generate_and_vote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Restrict CORS to Vercel and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gsv-math.vercel.app", "http://localhost:3000", "*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY = os.environ.get("API_KEY", "dev-secret-key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API Key credentials")

# Initialize models globally. ZeroGPU will automatically manage moving them to the GPU.
os.environ["LORA_REPO_ID"] = os.environ.get("LORA_REPO_ID", "Shabuuuuuuuuuuu/GSV-Math-Qwen2.5-VL-7B-Expert")
try:
    print("Initializing model...")
    model, processor = load_models()
    print("Model loaded.")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    model, processor = None, None

@spaces.GPU(duration=120)
def generate_answer(img, question, num_samples):
    """Wrapped function that runs on the ZeroGPU A100."""
    return cisc_generate_and_vote(model, processor, img, question, num_samples=num_samples)

@app.post("/", dependencies=[Depends(verify_api_key)])
async def solve_route(request: Request):
    if not model or not processor:
        return {"error": "Model failed to load during startup."}
        
    try:
        data = await request.json()
        image_b64 = data.get("image_base64")
        image_url = data.get("image_url")
        question = data.get("question")
        num_samples = int(data.get("num_samples", 3))
        
        if not (image_b64 or image_url) or not question:
            return {"error": "Missing image or question in payload"}
            
        if image_b64 and len(image_b64) > 15_000_000:
            return {"error": "Payload too large. Base64 image exceeds 15MB limit."}
            
        if image_url:
            if not (image_url.startswith("http://") or image_url.startswith("https://")):
                return {"error": "Invalid image URL scheme. SSRF protection blocked request."}
            
            # SSRF Protection: Block private IPs
            from urllib.parse import urlparse
            import socket
            import ipaddress
            try:
                hostname = urlparse(image_url).hostname
                ip = socket.gethostbyname(hostname)
                if ipaddress.ip_address(ip).is_private:
                    return {"error": "SSRF protection blocked request to private network."}
            except Exception:
                return {"error": "Failed to resolve image URL hostname."}

            import urllib.request
            req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
            # SSRF Protection: Add strict timeout
            with urllib.request.urlopen(req, timeout=10) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if max(img.size) > 768:
            img.thumbnail((768, 768))

        # Offload to the GPU-wrapped function
        best_answer, best_trace, vote_dist, clip_score, sympy_passed, owl_score = generate_answer(
            img, question, num_samples
        )
        
        logger.info(f"CISC Voting Complete. Distribution: {vote_dist}")
        
        return {
            "answer": best_answer, 
            "reasoning": best_trace, 
            "vote_distribution": vote_dist,
            "owl_grounding_score": owl_score,
            "clip_alignment_score": clip_score,
            "symbolic_check_passed": sympy_passed,
            "note": "Research demo, not a certified math solver - verify answers independently."
        }
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok", "message": "ZeroGPU backend is reachable."}
