import os
import torch
from unsloth import FastVisionModel
from peft import PeftModel

# HF_TOKEN is injected by the Space automatically if set in secrets
HF_TOKEN = os.getenv("HF_TOKEN")
# Expecting user to provide this or set it in Space variables
LORA_REPO_ID = os.getenv("LORA_REPO_ID", "YOUR_HF_USERNAME/gsv-math-qwen2.5-vl-lora")

def load_models():
    """Loads the base model in 4-bit and applies the LoRA adapter."""
    print("Loading Base Model (4-bit)...")
    # Base model used in Phase D of the master plan
    model, processor = FastVisionModel.from_pretrained(
        model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        token=HF_TOKEN
    )
    
    if LORA_REPO_ID != "YOUR_HF_USERNAME/gsv-math-qwen2.5-vl-lora":
        print(f"Applying LoRA Adapter from {LORA_REPO_ID}...")
        model = PeftModel.from_pretrained(model, LORA_REPO_ID, token=HF_TOKEN)
        
    FastVisionModel.for_inference(model)
    return model, processor
