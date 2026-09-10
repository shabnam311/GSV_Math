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

def select_salient_text(text: str, max_chars: int = 300) -> str:
    """
    Selects text for CLIP evaluation. If the text is longer than max_chars,
    preserves both the initial context (first ~100 chars) and the concluding
    deduction (last ~200 chars) rather than chopping off the final answer.
    """
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    
    head_len = 100
    tail_len = max_chars - head_len - 5 # allowance for ellipsis
    head = cleaned[:head_len].rsplit(' ', 1)[0]
    tail = cleaned[-tail_len:].split(' ', 1)[-1]
    return f"{head} ... {tail}"

def clip_alignment_score(image: Image.Image, reasoning_text: str) -> float:
    try:
        if not reasoning_text or not reasoning_text.strip():
            return None # Fail open fix: return None if no text to check
            
        model, processor = get_clip_model()
        
        # Smart text selection to preserve the final conclusion
        selected_text = select_salient_text(reasoning_text, max_chars=300)
            
        inputs = processor(text=[selected_text], images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logit = outputs.logits_per_image.item()
        
        # Normalize to roughly a 0.0 - 1.0 scale
        score = max(0.0, min(1.0, logit / 30.0))
        return score
    except Exception as e:
        print(f"CLIP alignment failed: {e}")
        return None # Fail open fix: return None on exception
