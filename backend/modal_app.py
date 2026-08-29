import modal
import os
import base64
import io
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the Modal App
app = modal.App("gsv-math-backend")

# 2. Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "unsloth",
        "unsloth_zoo",
        "trl",
        "peft",
        "bitsandbytes",
        "accelerate",
        "sympy",
        "spacy",
        "Pillow",
        "huggingface_hub",
        "fastapi[standard]"
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_dir("backend/pipeline", remote_path="/root/pipeline")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# 4. Define the Serverless GPU Class
@app.cls(
    image=image,
    gpu="T4",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=300, # 5 min max per request
    scaledown_window=120, # scale to 0 after 2 mins idle
    concurrency_limit=5, # Limit concurrent containers to prevent cost exhaustion
)
class GSVMathModel:
    @modal.enter()
    def load(self):
        print("Initializing Modal GPU Container...")
        
        # Inject the expected repo ID since we aren't using Space Secrets
        os.environ["LORA_REPO_ID"] = "Shabuuuuuuuuuuu/GSV-Math-Qwen2.5-VL-7B-Expert"
        
        from pipeline.model_loader import load_models
        self.model, self.processor = load_models()
        print("Model and adapter loaded successfully.")

    def _solve_internal(self, data: dict):
        try:
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

            from pipeline.cisc import cisc_generate_and_vote
            
            best_answer, best_trace, vote_dist, clip_score, sympy_passed, owl_score = cisc_generate_and_vote(
                self.model, self.processor, img, question, num_samples=num_samples
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

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        
        web_app = FastAPI()
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], # Allow Vercel domain in production
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @web_app.post("/solve")
        async def solve_route(request: Request):
            data = await request.json()
            return self._solve_internal(data)
            
        return web_app

# 5. Lightweight Health Endpoint (Does not require GPU / keeps costs at $0)
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "message": "Modal backend is reachable."}
