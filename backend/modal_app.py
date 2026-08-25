import modal
import os
import base64
import io
from PIL import Image

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
)

# 3. Persistent Volume for Hugging Face weights to avoid re-downloading on cold starts
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# 4. Define the Serverless GPU Class
@app.cls(
    image=image,
    gpu="T4",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=300, # 5 min max per request
    scaledown_window=120, # scale to 0 after 2 mins idle
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

    @modal.fastapi_endpoint(method="POST")
    def solve(self, data: dict):
        """
        Accepts {"image_base64": "...", "question": "..."}
        """
        try:
            image_b64 = data.get("image_base64")
            question = data.get("question")
            
            if not image_b64 or not question:
                return {"error": "Missing image_base64 or question in payload"}
                
            # Decode the base64 image
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # VRAM protection limit
            if max(img.size) > 768:
                img.thumbnail((768, 768))

            from pipeline.cisc import cisc_generate_and_vote
            
            # Run the voting loop (k=3 to balance accuracy vs Modal credits)
            best_answer, best_trace, vote_dist = cisc_generate_and_vote(
                self.model, self.processor, img, question, num_samples=3
            )
            
            return {
                "answer": best_answer, 
                "reasoning": best_trace, 
                "vote_distribution": vote_dist,
                "note": "Research demo, not a certified math solver - verify answers independently."
            }
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return {"error": str(e)}

# 5. Lightweight Health Endpoint (Does not require GPU / keeps costs at $0)
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "message": "Modal backend is reachable."}
