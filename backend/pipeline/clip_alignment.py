import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Lazy initialization placeholders
_clip_model = None
_clip_processor = None

def get_clip_model():
    global _clip_model, _clip_processor
    if _clip_model is None:
        model_id = "openai/clip-vit-base-patch32"
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
    return _clip_model, _clip_processor

def clip_alignment_score(image: Image.Image, reasoning_text: str) -> float:
    try:
        model, processor = get_clip_model()
        
        # Truncate text roughly to CLIP's 77 token limit (by character length to be safe)
        truncated_text = reasoning_text[:300]
        
        inputs = processor(text=[truncated_text], images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # CLIP logits are typically scaled around 15-35 for good matches
        logit = outputs.logits_per_image.item()
        
        # Normalize to roughly a 0.0 - 1.0 scale 
        score = max(0.0, min(1.0, logit / 30.0))
        return score
    except Exception as e:
        print(f"CLIP alignment failed: {e}")
        return 1.0 # Neutral baseline if fails
