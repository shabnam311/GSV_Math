import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# ── Training-matched constants ──
MAX_PIXELS = 156800  # from project_notebooks/Qwen_2.5v_finetuning.ipynb
MIN_PIXELS = 3136    # 56*56

HF_TOKEN = os.getenv("HF_TOKEN")
LORA_REPO_ID = os.getenv("LORA_REPO_ID", "Shabuuuuuuuuuuu/GSV-Math-Qwen2.5-VL-7B-Expert")


def load_models():
    """Loads the base model with BitsAndBytesConfig matching training, then applies LoRA."""
    print("Loading Base Model (Qwen/Qwen2.5-VL-7B-Instruct with live NF4 quantization)...")

    # Match training quantization config exactly
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )

    # Pin processor pixel budget to match training
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        token=HF_TOKEN,
    )

    if LORA_REPO_ID and LORA_REPO_ID != "YOUR_HF_USERNAME/gsv-math-qwen2.5-vl-lora":
        print(f"Applying LoRA Adapter from {LORA_REPO_ID}...")
        model = PeftModel.from_pretrained(model, LORA_REPO_ID, token=HF_TOKEN)

    model.eval()
    return model, processor
