import torch
import spacy
from transformers import OwlViTProcessor, OwlViTForObjectDetection

_owl_model = None
_owl_processor = None
_nlp = None

def get_owl_tools():
    global _owl_model, _owl_processor, _nlp
    if _owl_model is None:
        _owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        _owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to("cuda" if torch.cuda.is_available() else "cpu")
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _owl_model, _owl_processor, _nlp

def extract_visual_claims(text, nlp):
    doc = nlp(text)
    
    # Comprehensive visual geometry, math, and diagram components
    visual_terms = {
        # Geometric shapes
        "triangle", "circle", "square", "rectangle", "polygon", "quadrilateral", "trapezoid", "parallelogram",
        "ellipse", "cube", "cylinder", "cone", "sphere", "pyramid",
        # Geometric elements
        "line", "segment", "ray", "angle", "vertex", "vertices", "axis", "axes", "point", "edge", "side",
        "hypotenuse", "leg", "base", "height", "altitude", "radius", "diameter", "circumference",
        "perimeter", "sector", "chord", "tangent", "secant", "arc", "center", "origin", "diagonal",
        # Diagram elements
        "graph", "chart", "bar", "box", "grid", "arrow", "tick", "curve", "region", "shaded region", "overlap",
        # Real-world objects (used in applied math / counting problems)
        "dice", "clock", "watch", "laptop", "car", "apple", "coin", "table", "block", "box"
    }
    
    nouns = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) < 4:
            chunk_lower = chunk.text.lower().strip()
            if any(term in chunk_lower for term in visual_terms):
                nouns.append(chunk_lower)
    
    # Deterministic deduplication
    unique_nouns = list(dict.fromkeys(nouns))
    return unique_nouns[:10]

def owl_grounding_score(image, text):
    try:
        model, processor, nlp = get_owl_tools()
        claims = extract_visual_claims(text, nlp)
        if not claims:
            return None # Fail open fix: return None if unable to verify

        inputs = processor(
            text=[claims],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=16,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.sigmoid(outputs.logits[0])
        max_confidences = probs.max(dim=0).values
        return max_confidences.mean().item()
    except Exception as e:
        print(f"OWL-ViT grounding failed: {e}")
        return None # Fail open fix: return None on exception
